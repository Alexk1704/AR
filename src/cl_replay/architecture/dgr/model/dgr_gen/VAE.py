import time
import numpy as np
import tensorflow as tf

from cl_replay.api.utils import log


class VAE(tf.keras.Model):
    
    
    def __init__(self, **kwargs):
        super().__init__()
        self.dgr_model = kwargs.get('dgr_model')    
        self.encoder = None
        self.decoder = None
        
        self.data_dim       = kwargs.get('input_size')
        self.label_dim      = kwargs.get('num_classes')
        self.batch_size     = kwargs.get('batch_size')
        self.vae_epochs     = kwargs.get('vae_epochs')
        
        self.recon_loss     = kwargs.get('recon_loss')
        self.latent_dim     = kwargs.get('latent_dim')
        
        self.enc_cond_input = kwargs.get('enc_cond_input')
        self.dec_cond_input = kwargs.get('dec_cond_input')
        
        self.vae_beta       = kwargs.get('vae_beta')
        
        self.vae_epsilon    = kwargs.get('vae_epsilon')
        self.adam_beta1     = kwargs.get('adam_beta1')
        self.adam_beta2     = kwargs.get('adam_beta2')
        
        self.vis_path       = kwargs.get('vis_path')

        self.optimizer = tf.keras.optimizers.Adam(self.vae_epsilon, self.adam_beta1, self.adam_beta2)


    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)
    

    def encode(self, xs, ys):
        if self.enc_cond_input == 'yes':
            mean, logvar = self.encoder([xs, ys])
        else:
            mean, logvar = self.encoder(xs)
        
        return mean, logvar


    def reparameterize(self, mean, logvar):
        """ 
        Reparameterization trick.
        - Sample randomly from Gaussian dist., parameterized by eps (mean) & Sigma (std.). 
        """
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean


    def decode(self, zs, ys, apply_sigmoid=False):
        if self.dec_cond_input == 'yes':
            logits = self.decoder([zs, ys])
        else:
            logits = self.decoder(zs)
        if apply_sigmoid:
            return tf.sigmoid(logits)
        else:
            return logits


    def fit(self, *args, **kwargs):
        """ assume that first argument is an iterator object """
        steps_per_epoch = kwargs.get("steps_per_epoch", 1)
        epochs          = kwargs.get("epochs", 1)
        max_steps       = epochs * steps_per_epoch
        
        log_metrics     = kwargs.get("callbacks")[0]
        log_metrics.model = self.dgr_model
        log_metrics.on_train_begin()
        
        log.debug(
            f'TRAINING VAE-GEN FOR {epochs} EPOCHS WITH {steps_per_epoch} STEPS...')
        epoch = 0

        for i, (x, y, sample_weights) in enumerate(args[0], start=1):
            self.train_step(x, y, sample_weights)
            log_metrics.on_batch_end(batch=i)

            if i % steps_per_epoch == 0:
                log.info(
                    f'EPOCH {epoch} STEP {i}\t' + 
                    f'vae_loss\t{self.dgr_model.metrics[0].result()}\t' +
                    f'step_time\t{self.dgr_model.metrics[1].result()}')
                epoch += 1 
                log_metrics.on_epoch_end(epoch)
            
            if i == max_steps: break

        log_metrics.custom_name = 'encoder'
        log_metrics.on_train_end()
        log_metrics.current_task -= 1


    def train_step(self, xs, ys, sample_weights, **kwargs):
        t1 = time.time()

        with tf.GradientTape() as tape:
            mean, logvar    = self.encode(xs, ys)
            z               = self.reparameterize(mean, logvar)
            x_logit         = self.decode(z, ys)

            cross_ent_loss  = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=xs)
            
            logpx_z     = -tf.reduce_sum(cross_ent_loss, axis=[1, 2, 3])
            logpx_z     *= sample_weights
            
            logpz       = self.log_normal_pdf(z, 0., 0.)
            logqz_x     = self.log_normal_pdf(z, mean, logvar)

            loss = -tf.reduce_mean(logpx_z + logpz - logqz_x)

        gradients = tape.gradient(loss, self.trainable_variables) # compute gradients
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables)) # update weights
    
        t2 = time.time()
        delta = (t2 - t1) * 1000.  # ms
        self.dgr_model.metrics[0].update_state(loss)
        self.dgr_model.metrics[1].update_state(delta)


    def sample(self, eps=None, batch_size=100, scalar_classes=None):
        if eps is None:
            eps = tf.random.normal(shape=(batch_size, self.latent_dim))
        if scalar_classes is not None:
            rnd_ys = np.random.choice(scalar_classes, size=batch_size)
            tmp = tf.eye(batch_size, self.label_dim)
            ys = tf.gather(tmp, rnd_ys)
            return self.decode(eps, ys, apply_sigmoid=True)
        else:
            return self.decode(eps, apply_sigmoid=True)


    def save_weights(self, *args, **kwargs): # FIXME: serialization
        pass


    def load_weights(self, *args, **kwargs): # FIXME: serialization
        pass
