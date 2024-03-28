import numpy as np
import tensorflow as tf

from cl_replay.api.utils    import log
from cl_replay.api.parsing  import Kwarg_Parser

from .dgr_gen.VAE import VAE
from .dgr_gen.GAN import GAN


class DGR:
    def __init__(self, **kwargs):
        self.parser             = Kwarg_Parser(**kwargs)
        self.name               = 'DGR'
        
        self.vis_path           = self.parser.add_argument('--vis_path',                type=str,   required=True)
        
        self.input_size         = self.parser.add_argument('--input_size',              type=int,   default=[28, 28, 1], help='the input dimensions')
        self.num_classes        = self.parser.add_argument('--num_classes',             type=int,   default=10, help='the output dimensions')
        self.batch_size         = self.parser.add_argument('--batch_size',              type=int,   default=100, help='the size of mini batches for DGR')
        
        self.adam_beta1         = self.parser.add_argument('--adam_beta_1',             type=float, default=0.9, help='ADAM beta1')
        self.adam_beta2         = self.parser.add_argument('--adam_beta_2',             type=float, default=0.999, help='ADAM beta2')
        
        self.solver_epochs      = self.parser.add_argument('--solver_epochs',           type=int,   default=50, help='epochs to train solver structure.')
        
        self.generator_type     = self.parser.add_argument('--generator_type',          type=str,   default="VAE", choices=["GAN", "VAE"])
        if self.generator_type == 'VAE':
            self.recon_loss         = self.parser.add_argument('--recon_loss',          type=str,   default='binary_crossentropy', choices=['binary_crossentropy', 'mean_squared_error'], help='sets the reconstruction loss (BCE/MSE).')
            self.latent_dim         = self.parser.add_argument('--latent_dim',          type=int,   default=25, help='the dimension of the latent vector')
            self.enc_cond_input     = self.parser.add_argument('--enc_cond_input',      type=str,   default='no', choices=['no', 'yes'], help='C-VAE: Latent space encodes label information.')
            self.dec_cond_input     = self.parser.add_argument('--dec_cond_input',      type=str,   default='no', choices=['no', 'yes'], help='C-VAE: Conditionizing latent space.')
            self.vae_beta           = self.parser.add_argument('--vae_beta',            type=float, default=1.0, help='the beta factor for disentangling the VAE')
            self.vae_epsilon        = self.parser.add_argument('--vae_epsilon',         type=float, default=1e-4, help='the learning rate of the VAE')
            self.vae_epochs         = self.parser.add_argument('--vae_epochs',          type=int,   default=100, help='the number of epochs of the VAE')
        elif self.generator_type == 'GAN':
            self.noise_dim          = self.parser.add_argument('--noise_dim',           type=int,   default=100, help='the dimension of the noise vector')
            self.conditional        = self.parser.add_argument('--conditional',         type=str,   default='no', choices=['no', 'yes'], help='if the GAN should be conditional based')
            self.gp_weight          = self.parser.add_argument('--gp_weight',           type=int,   default=10, help='weight for gradient penalty')
            self.wgan_disc_iters    = self.parser.add_argument('--wgan_disc_iters',     type=int,   default=5,  help='how many discriminator iters per gen iters? (WGAN)')
            self.wasserstein        = self.parser.add_argument('--wasserstein',         type=str,   default='no', choices=['no', 'yes'], help='if the GAN should be wasserstein based')
            self.gan_epsilon        = self.parser.add_argument('--gan_epsilon',         type=float, default=1e-4, help='the learning rate of the GAN')
            self.gan_beta1          = self.parser.add_argument('--gan_beta_2',          type=float, default=0.5, help='ADAM beta1')
            self.gan_beta2          = self.parser.add_argument('--gan_beta_2',          type=float, default=0.95, help='ADAM beta2')
            self.gan_epochs         = self.parser.add_argument('--gan_epochs',          type=int,   default=100, help='the number of epochs of the GAN')

        if self.generator_type == 'VAE':
            args = { 
                'dgr_model': self,
                'input_size': self.input_size, 'num_classes': self.num_classes,
                'enc_cond_input': self.enc_cond_input, 'dec_cond_input': self.dec_cond_input,
                'recon_loss': self.recon_loss, 'latent_dim': self.latent_dim, 'vae_beta': self.vae_beta,
                'vae_epsilon': self.vae_epsilon, 'adam_beta1': self.adam_beta1, 'adam_beta2': self.adam_beta2,
                'batch_size': self.batch_size, 'vae_epochs': self.vae_epochs,
                'vis_path': self.vis_path
            }
            self.generator = VAE(
                inputs=None, outputs=None, name='VAE', **args)
            self.generator.compile()
            
            self.generator.encoder = kwargs.get('encoder')
            self.generator.decoder = kwargs.get('decoder')
            
            self.all_metrics = [
                tf.keras.metrics.Mean(name='vae_loss'),
                tf.keras.metrics.Mean(name='step_time')
            ]
                    
        elif self.generator_type == 'GAN':
            args = { 
                'dgr_model': self,
                'input_size': self.input_size, 'num_classes': self.num_classes,
                'conditional': self.conditional,
                'noise_dim': self.noise_dim, 'wasserstein': self.wasserstein,
                'gp_weight': self.gp_weight, 'wgan_disc_iters': self.wgan_disc_iters,
                'gan_epsilon': self.gan_epsilon, 'gan_beta1': self.gan_beta1, 'gan_beta2': self.gan_beta2,
                'batch_size': self.batch_size, 'gan_epochs': self.gan_epochs,
                'vis_path': self.vis_path
            }

            self.generator = GAN(
                inputs=None, outputs=None, name='GAN', **args)
            self.generator.compile()
            
            self.generator.generator        = kwargs.get('generator')
            self.generator.discriminator    = kwargs.get('discriminator')
            
            self.all_metrics = [
                tf.keras.metrics.Mean(name='gen_loss'),
                tf.keras.metrics.Mean(name='disc_loss'),
                tf.keras.metrics.Mean(name='step_time')
            ]

        self.solver = kwargs.get('solver')

        self.dtype_np_float = np.float32
        self.dtype_tf_float = tf.float32
        self.supports_chkpt = False
        self.train_generator_flag = False


    def set_train_generator_flag(self, flag):
        self.train_generator_flag = flag
        
        
    def get_model_weights(self):
        if self.generator_type == 'VAE':
            model_weights = {
                'encoder' : self.generator.encoder.get_weights(),
                'decoder' : self.generator.decoder.get_weights(),
                'solver'  : self.solver.get_weights()
            }
        elif self.generator_type == 'GAN':
            model_weights = {
                'generator' : self.generator.generator.get_weights(),
                'discriminator' : self.generator.discriminator.get_weights(),
                'solver'  : self.solver.get_weights()
            }
            #for w_arr in model_weights['generator']: print(w_arr.mean())
        return model_weights
    
    
    def set_model_weights(self, model, weights):
        model.set_weights(weights)


    def reset_generator(self, initial_weights):
        if self.generator_type == 'VAE':
            self.set_model_weights(self.generator.encoder, initial_weights['encoder'])
            self.set_model_weights(self.generator.decoder, initial_weights['decoder'])
            
        elif self.generator_type == 'GAN':
            self.set_model_weights(self.generator.generator, initial_weights['generator'])
            self.set_model_weights(self.generator.discriminator, initial_weights['discriminator'])
            #for w_arr in initial_weights['generator']: print(w_arr.mean())


    def reset_solver(self, initial_weights):        
        self.set_model_weights(self.solver, initial_weights['solver'])


    def build(self, **kwargs): return self


    def get_model_variables(self):
        if self.generator_type == 'VAE':
            return [net.trainable_variables for net in
                    [self.generator.encoder, self.generator.decoder, self.solver]]
            
        elif self.generator_type == 'GAN':
            return [net.trainable_variables for net in
                    [self.generator.generator, self.generator.discriminator, self.solver]]


    def fit(self, *args, **kwargs):
        # cbs = kwargs["callbacks"]
        # kwargs["callbacks"] = None
        if self.train_generator_flag == True:
            kwargs["epochs"] = self.vae_epochs if self.generator_type == "VAE" else self.gan_epochs
            self.generator.fit(*args, **kwargs)
        kwargs["epochs"] = self.solver_epochs
        log.debug(f'\tTRAINING SOLVER FOR {kwargs["epochs"]} EPOCHS')
        # kwargs["callbacks"] = cbs
        self.solver.fit(*args, **kwargs)


    def evaluate(self, *args, **kwargs):
        self.solver.test_task = self.test_task
        self.solver.evaluate(*args, **kwargs)


    def get_model_params(self):
        if self.generator_type == 'VAE':
            return {
                'input_size': self.input_size, 'num_classes': self.num_classes,
                'enc_cond_input': self.enc_cond_input, 'dec_cond_input': self.dec_cond_input,
                'recon_loss': self.recon_loss, 'latent_dim': self.latent_dim, 'vae_beta': self.vae_beta,
                'vae_epsilon': self.vae_epsilon, 'adam_beta1': self.adam_beta1, 'adam_beta2': self.adam_beta2,
                'batch_size': self.batch_size, 'vae_epochs': self.vae_epochs,
                'vis_path': self.vis_path
            }
        elif self.generator_type == 'GAN':
            return {
                'input_size': self.input_size, 'num_classes': self.num_classes,
                'conditional': self.conditional,
                'noise_dim': self.noise_dim, 'wasserstein': self.wasserstein,
                'gp_weight': self.gp_weight, 'wgan_disc_iters': self.wgan_disc_iters,
                'gan_epsilon': self.gan_epsilon, 'gan_beta1': self.gan_beta1, 'gan_beta2': self.gan_beta2,
                'batch_size': self.batch_size, 'gan_epochs': self.gan_epochs,
                'vis_path': self.vis_path
            }


    def save_weights(self, *args, **kwargs):
        filepath = args[0]
        self.solver.save_weights(filepath + "solver", *args[1:], **kwargs)
        self.generator.save_weights(
            filepath + "generator", *args[1:], **kwargs)


    def load_weights(self, *args, **kwargs):
        filepath = args[0]
        self.solver.load_weights(filepath + "solver", *args[1:], **kwargs)
        self.generator.load_weights(
            filepath + "generator", *args[1:], **kwargs)

    @property
    def metrics(self):
        return self.all_metrics
