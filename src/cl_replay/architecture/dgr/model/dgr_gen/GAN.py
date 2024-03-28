import time
import numpy as np
import tensorflow as tf

from cl_replay.api.utils import log
from cl_replay.api.data.Dataset import visualize_data


class GAN(tf.keras.Model):
    
    
    def __init__(self, **kwargs):
        super().__init__()
        self.dgr_model = kwargs.get('dgr_model')
        self.generator      = None
        self.discriminator  = None

        self.data_dim       = kwargs.get('input_size')
        self.label_dim      = kwargs.get('num_classes')
        self.batch_size     = kwargs.get('batch_size')
        self.gan_epochs     = kwargs.get('gan_epochs')
        
        self.noise_dim      = kwargs.get('noise_dim')
        self.conditional    = kwargs.get('conditional')
        self.wasserstein    = kwargs.get('wasserstein')
        self.gp_weight      = kwargs.get('gp_weight')
        self.wgan_disc_iters= kwargs.get('wgan_disc_iters')
        self.wgan_gen_iters = 1  

        self.gan_epsilon    = kwargs.get('gan_epsilon')
        self.gan_beta1      = kwargs.get('gan_beta1')
        self.gan_beta2      = kwargs.get('gan_beta2')
        
        self.vis_path       = kwargs.get('vis_path')

        self.loss_generator = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.loss_discriminator = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        self.gen_opt = tf.keras.optimizers.Adam(
            self.gan_epsilon, beta_1=self.gan_beta1, beta_2=self.gan_beta2)
        self.disc_opt = tf.keras.optimizers.Adam(
            self.gan_epsilon, beta_1=self.gan_beta1, beta_2=self.gan_beta2)


    def discriminate(self, xs, ys):
        if self.conditional == 'yes':
            return self.discriminator([xs, ys])
        else:
            return self.discriminator(xs)


    def calculate_generator_loss(self, logits):
        if self.wasserstein == 'yes':
            return -tf.reduce_mean(logits)
        else:
            return self.loss_generator(tf.ones_like(logits), logits)


    def calculate_discriminator_loss(self, real_logits, fake_logits):
        if self.wasserstein == 'yes':
            return tf.subtract(
                tf.reduce_mean(fake_logits),
                tf.reduce_mean(real_logits)
            )
        else:
            return tf.add(
                self.loss_discriminator(tf.ones_like(real_logits), real_logits),
                self.loss_discriminator(tf.zeros_like(fake_logits), fake_logits)
            )


    def gradient_penalty(self, real_images, fake_images):
        #alpha = tf.random.normal([tf.shape(real_images)[0], 1, 1, 1], 0., 1.) # FIXME: switch for conv vs. fc mode
        alpha = tf.random.normal([tf.shape(real_images)[0], 1]) # bc flattened...
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator(interpolated)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        #norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3])) # FIXME: see above...
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1]))

        return tf.reduce_mean((norm - 1.0) ** 2)


    def fit(self, *args, **kwargs):
        """ assume that first argument is an iterator object """
        steps_per_epoch = kwargs.get("steps_per_epoch", 1)
        epochs          = kwargs.get("epochs", 1)
        max_steps       = epochs * steps_per_epoch
        
        log_metrics     = kwargs.get("callbacks")[0]
        log_metrics.model = self.dgr_model
        log_metrics.on_train_begin()
        
        log.debug(
            f'TRAINING GAN-GEN FOR {epochs} EPOCHS WITH {steps_per_epoch} STEPS...')
        epoch = 0

        for i, (x, y, sample_weights) in enumerate(args[0], start=1):
            self.train_step(x, y, sample_weights)
            log_metrics.on_batch_end(batch=i)

            if i % steps_per_epoch == 0:
                log.info(
                    f'EPOCH {epoch} STEP {i}\t' + 
                    f'gen_loss\t{self.dgr_model.metrics[0].result()}\t' +
                    f'disc_loss\t{self.dgr_model.metrics[1].result()}\t' + 
                    f'step_time\t{self.dgr_model.metrics[2].result()}')
                epoch += 1 
                log_metrics.on_epoch_end(epoch)
                
                # gen_xs = self.sample(x.shape[0], y)
                # gen_xs = np.reshape(gen_xs, newshape=(x.shape[0], self.data_dim[0], self.data_dim[1], self.data_dim[2]))
                # visualize_data(gen_xs, y, save_path=self.vis_path, filename=f'gen_xs_E{epoch}')
                
            if i == max_steps: break

        log_metrics.custom_name = 'generator'
        log_metrics.on_train_end()
        log_metrics.current_task -= 1


    def train_step(self, xs, ys, sample_weights, **kwargs):
        
        t1 = time.time()
        
        if self.wasserstein == 'yes':
            for _ in range(self.wgan_disc_iters):
                with tf.GradientTape() as disc_tape:
                    fake_xs     = self.sample(xs.shape[0], ys=None)
                    xs_reshaped = tf.reshape(xs, (xs.shape[0], tf.reduce_prod(xs.shape[1:])))
                    real_output = self.discriminator(xs_reshaped)
                    fake_output = self.discriminator(fake_xs)
                    disc_loss   = self.calculate_discriminator_loss(real_output, fake_output)
                    disc_loss   += self.gradient_penalty(xs_reshaped, fake_xs) * self.gp_weight
                    disc_loss   *= sample_weights
                    disc_grads  = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
                    
                self.disc_opt.apply_gradients(zip(disc_grads, self.discriminator.trainable_variables))

            for _ in range(self.wgan_gen_iters):
                with tf.GradientTape() as gen_tape:
                    fake_xs     = self.sample(xs.shape[0], ys=None)
                    gen_output  = self.discriminator(fake_xs)
                    gen_loss    = self.calculate_generator_loss(gen_output)
                    gen_loss    *= sample_weights
                    gen_grads   = gen_tape.gradient(gen_loss, self.generator.trainable_variables)

                self.gen_opt.apply_gradients(zip(gen_grads, self.generator.trainable_variables))
        else:
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                fake_xs     = self.sample(xs.shape[0], ys)
                xs_reshaped = tf.reshape(xs, (xs.shape[0], tf.reduce_prod(xs.shape[1:])))
                real_output = self.discriminate(xs_reshaped, ys)
                fake_output = self.discriminate(fake_xs, ys)
                gen_loss    = self.calculate_generator_loss(fake_output)
                disc_loss   = self.calculate_discriminator_loss(real_output, fake_output)
                gen_loss    *= sample_weights
                disc_loss   *= sample_weights
                gen_grads   = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
                disc_grads  = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

            self.gen_opt.apply_gradients(zip(gen_grads, self.generator.trainable_variables))
            self.disc_opt.apply_gradients(zip(disc_grads, self.discriminator.trainable_variables))
            
        t2 = time.time()
        delta = (t2 - t1) * 1000.  # ms
        self.dgr_model.metrics[0].update_state(gen_loss)
        self.dgr_model.metrics[1].update_state(disc_loss)
        self.dgr_model.metrics[2].update_state(delta)


    def sample(self, batch_size=100, ys=None):
        zs = tf.random.normal([batch_size, self.noise_dim]) # generate a batch of noise vectors
        if self.conditional == 'yes' and ys is not None:
            return self.generator([zs, ys])
        else:
            return self.generator(zs)


    def save_weights(self, *args, **kwargs): # FIXME: serialization
        pass


    def load_weights(self, *args, **kwargs): # FIXME: serialization
        pass
