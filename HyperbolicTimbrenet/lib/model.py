import tensorflow as tf
from .hyp_model.linear_hyp import LinearHyperbolic, LinearHyperbolicPlusPlus
from .hyp_model.manifold import Poincare

class CVAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.inference_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(128, 1024, 2)),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=1, strides=1, padding='same'),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.AveragePooling2D(
                    pool_size=(2, 2),strides=2,padding='same'),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.AveragePooling2D(
                    pool_size=(2, 2),strides=2,padding='same'),
                
                tf.keras.layers.Conv2D(
                    filters=128, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.Conv2D(
                    filters=128, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.AveragePooling2D(
                    pool_size=(2, 2),strides=2,padding='same'),
                
                tf.keras.layers.Conv2D(
                    filters=256, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.Conv2D(
                    filters=256, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.AveragePooling2D(
                    pool_size=(2, 2),strides=2,padding='same'),
                
                tf.keras.layers.Conv2D(
                    filters=256, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.Conv2D(
                    filters=256, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.AveragePooling2D(
                    pool_size=(2, 2),strides=2,padding='same'),
                
                tf.keras.layers.Conv2D(
                    filters=256, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.Conv2D(
                    filters=256, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.AveragePooling2D(
                    pool_size=(2, 2),strides=2,padding='same'),
                
                tf.keras.layers.Conv2D(
                    filters=256, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.Conv2D(
                    filters=256, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.AveragePooling2D(
                    pool_size=(2, 2),strides=2,padding='same'),
                
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(latent_dim + latent_dim),

          ]
        )
        
        self.generative_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=1*8*256), # * venia con el modelo original


                tf.keras.layers.Reshape(target_shape=(1, 8, 256)),
                
                tf.keras.layers.UpSampling2D(
                    size = 2, interpolation='nearest'),
                tf.keras.layers.Conv2DTranspose(
                    filters=256, kernel_size=(2,16), strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                tf.keras.layers.Conv2DTranspose(
                    filters=256, kernel_size=3, strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                
                tf.keras.layers.UpSampling2D(
                    size = 2, interpolation='nearest'),
                tf.keras.layers.Conv2DTranspose(
                    filters=256, kernel_size=(2,16), strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                tf.keras.layers.Conv2DTranspose(
                    filters=256, kernel_size=3, strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                
                tf.keras.layers.UpSampling2D(
                    size = 2, interpolation='nearest'),
                tf.keras.layers.Conv2DTranspose(
                    filters=256, kernel_size=(2,16), strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                tf.keras.layers.Conv2DTranspose(
                    filters=256, kernel_size=3, strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                
                tf.keras.layers.UpSampling2D(
                    size = 2, interpolation='nearest'),
                tf.keras.layers.Conv2DTranspose(
                    filters=256, kernel_size=(2,16), strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                tf.keras.layers.Conv2DTranspose(
                    filters=256, kernel_size=3, strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                
                tf.keras.layers.UpSampling2D(
                    size = 2, interpolation='nearest'),
                tf.keras.layers.Conv2DTranspose(
                    filters=128, kernel_size=(2,16), strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                tf.keras.layers.Conv2DTranspose(
                    filters=128, kernel_size=3, strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                
                tf.keras.layers.UpSampling2D(
                    size = 2, interpolation='nearest'),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=(2,16), strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=3, strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                
                tf.keras.layers.UpSampling2D(
                    size = 2, interpolation='nearest'),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=(2,16), strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=3, strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                
                tf.keras.layers.Conv2DTranspose(
                    filters=2, kernel_size=1, strides=1, padding="SAME", activation='tanh'),
            
            ]
        )
        
    def sample(self, eps=None, apply_tanh=True):
        if eps is None:
            eps = tf.random.normal(shape=(2, self.latent_dim))
        return self.decode(eps, apply_tanh)

    def encode(self, x):
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_tanh=False):
        logits = self.generative_net(z)
        if apply_tanh:
            probs = tf.math.multiply(tf.math.tanh(logits),0.999999)
            return probs
        else:
            probs = tf.math.multiply(logits,0.999999)
        return logits




class HCVAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(HCVAE, self).__init__()
        self.latent_dim = latent_dim
        self.inference_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(128, 1024, 2)),
                
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=1, strides=1, padding='same'),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.AveragePooling2D(
                    pool_size=(2, 2),strides=2,padding='same'),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.AveragePooling2D(
                    pool_size=(2, 2),strides=2,padding='same'),
                
                tf.keras.layers.Conv2D(
                    filters=128, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.Conv2D(
                    filters=128, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.AveragePooling2D(
                    pool_size=(2, 2),strides=2,padding='same'),
                
                tf.keras.layers.Conv2D(
                    filters=256, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.Conv2D(
                    filters=256, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.AveragePooling2D(
                    pool_size=(2, 2),strides=2,padding='same'),
                
                tf.keras.layers.Conv2D(
                    filters=256, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.Conv2D(
                    filters=256, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.AveragePooling2D(
                    pool_size=(2, 2),strides=2,padding='same'),
                
                tf.keras.layers.Conv2D(
                    filters=256, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.Conv2D(
                    filters=256, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.AveragePooling2D(
                    pool_size=(2, 2),strides=2,padding='same'),
                
                tf.keras.layers.Conv2D(
                    filters=256, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.Conv2D(
                    filters=256, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.AveragePooling2D(
                    pool_size=(2, 2),strides=2,padding='same'),
                
                tf.keras.layers.Flatten(),
                LinearHyperbolicPlusPlus(latent_dim+latent_dim, Poincare(), 1),


          ]
        )
        
        self.generative_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),

                LinearHyperbolicPlusPlus(1*8*256, Poincare(), 1),

                tf.keras.layers.Reshape(target_shape=(1, 8, 256)),
                
                tf.keras.layers.UpSampling2D(
                    size = 2, interpolation='nearest'),
                tf.keras.layers.Conv2DTranspose(
                    filters=256, kernel_size=(2,16), strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                tf.keras.layers.Conv2DTranspose(
                    filters=256, kernel_size=3, strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                
                tf.keras.layers.UpSampling2D(
                    size = 2, interpolation='nearest'),
                tf.keras.layers.Conv2DTranspose(
                    filters=256, kernel_size=(2,16), strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                tf.keras.layers.Conv2DTranspose(
                    filters=256, kernel_size=3, strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                
                tf.keras.layers.UpSampling2D(
                    size = 2, interpolation='nearest'),
                tf.keras.layers.Conv2DTranspose(
                    filters=256, kernel_size=(2,16), strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                tf.keras.layers.Conv2DTranspose(
                    filters=256, kernel_size=3, strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                
                tf.keras.layers.UpSampling2D(
                    size = 2, interpolation='nearest'),
                tf.keras.layers.Conv2DTranspose(
                    filters=256, kernel_size=(2,16), strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                tf.keras.layers.Conv2DTranspose(
                    filters=256, kernel_size=3, strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                
                tf.keras.layers.UpSampling2D(
                    size = 2, interpolation='nearest'),
                tf.keras.layers.Conv2DTranspose(
                    filters=128, kernel_size=(2,16), strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                tf.keras.layers.Conv2DTranspose(
                    filters=128, kernel_size=3, strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                
                tf.keras.layers.UpSampling2D(
                    size = 2, interpolation='nearest'),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=(2,16), strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=3, strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                
                tf.keras.layers.UpSampling2D(
                    size = 2, interpolation='nearest'),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=(2,16), strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=3, strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                
                tf.keras.layers.Conv2DTranspose(
                    filters=2, kernel_size=1, strides=1, padding="SAME", activation='tanh'),
            
            ]
        )
        
    def sample(self, eps=None, apply_tanh=True):
        if eps is None:
            eps = tf.random.normal(shape=(2, self.latent_dim))
        return self.decode(eps, apply_tanh)

    def encode(self, x):
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_tanh=False):
        logits = self.generative_net(z)
        if apply_tanh:
            probs = tf.math.multiply(tf.math.tanh(logits),0.999999)
            return probs
        else:
            probs = tf.math.multiply(logits,0.999999)
        return logits







class HVAE_NEW(tf.keras.Model):
    def __init__(self, latent_dim):
        super(HVAE_NEW, self).__init__()
        self.latent_dim = latent_dim
        self.inference_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(128, 1024, 2)),
                
                tf.keras.layers.Conv2D(
                    filters=16, kernel_size=1, strides=1, padding='same'),
                tf.keras.layers.Conv2D(
                    filters=16, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.Conv2D(
                    filters=16, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.Permute((1,3,2)),

                LinearHyperbolicPlusPlus(128, Poincare(), 1, activation=tf.keras.layers.LeakyReLU(), output_hyp=True),
                tf.keras.layers.Dropout(0.25),

                LinearHyperbolicPlusPlus(64, Poincare(), 1, activation=tf.keras.layers.LeakyReLU(), output_hyp=True,input_hyp=True),
                tf.keras.layers.Dropout(0.25),
                LinearHyperbolicPlusPlus(16, Poincare(), 1, activation=tf.keras.layers.LeakyReLU(),input_hyp=True),
                
                tf.keras.layers.Permute((1,3,2)),



                
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.AveragePooling2D(
                    pool_size=(2, 2),strides=2,padding='same'),
                
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.AveragePooling2D(
                    pool_size=(2, 2),strides=2,padding='same'),
                
                tf.keras.layers.Conv2D(
                    filters=128, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.Conv2D(
                    filters=128, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.AveragePooling2D(
                    pool_size=(2, 2),strides=2,padding='same'),
                
                tf.keras.layers.Conv2D(
                    filters=128, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.Conv2D(
                    filters=128, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.AveragePooling2D(
                    pool_size=(2, 2),strides=2,padding='same'),
                

                tf.keras.layers.Flatten(),

                LinearHyperbolicPlusPlus(latent_dim+latent_dim, Poincare(), 1),

          ]
        )
        
        self.generative_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),

                LinearHyperbolicPlusPlus(8*1*128, Poincare(), 1),

                tf.keras.layers.Reshape(target_shape=(8, 1, 128)),
      
                
                tf.keras.layers.UpSampling2D(
                    size = 2, interpolation='nearest'),
                tf.keras.layers.Conv2DTranspose(
                    filters=128, kernel_size=(2,16), strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                tf.keras.layers.Conv2DTranspose(
                    filters=128, kernel_size=3, strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                
                tf.keras.layers.UpSampling2D(
                    size = 2, interpolation='nearest'),
                tf.keras.layers.Conv2DTranspose(
                    filters=128, kernel_size=(2,16), strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                tf.keras.layers.Conv2DTranspose(
                    filters=128, kernel_size=3, strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                
                tf.keras.layers.UpSampling2D(
                    size = 2, interpolation='nearest'),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=(2,16), strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=3, strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                
                tf.keras.layers.UpSampling2D(
                    size = 2, interpolation='nearest'),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=(2,16), strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                # linear layer
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=3, strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                
                tf.keras.layers.Permute((1,3,2)),
                LinearHyperbolicPlusPlus(64, Poincare(), 1, activation=tf.keras.layers.LeakyReLU(), output_hyp=True),

                tf.keras.layers.Dropout(0.25),
                LinearHyperbolicPlusPlus(128, Poincare(), 1, activation=tf.keras.layers.LeakyReLU(), output_hyp=True,input_hyp=True),

                tf.keras.layers.Dropout(0.25),

                LinearHyperbolicPlusPlus(1024, Poincare(), 1, activation=tf.keras.layers.LeakyReLU(),input_hyp=True),
                tf.keras.layers.Permute((1,3,2)),


                tf.keras.layers.Conv2DTranspose(
                    filters=16, kernel_size=(2,16), strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                tf.keras.layers.Conv2DTranspose(
                    filters=16, kernel_size=3, strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                
                tf.keras.layers.Conv2DTranspose(
                    filters=2, kernel_size=1, strides=1, padding="SAME", activation='tanh'),
            
            ]
        )
        
    def sample(self, eps=None, apply_tanh=True):
        if eps is None:
            eps = tf.random.normal(shape=(2, self.latent_dim))
        return self.decode(eps, apply_tanh)

    def encode(self, x):
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_tanh=False):
        logits = self.generative_net(z)
        if apply_tanh:
            probs = tf.math.multiply(tf.math.tanh(logits),0.999999)
            return probs
        else:
            probs = tf.math.multiply(logits,0.999999)
        return logits




class HVAE_BREGUEL(tf.keras.Model):
    def __init__(self, latent_dim):
        super(HVAE_BREGUEL, self).__init__()
        print("yoooooo")
        self.latent_dim = latent_dim
        self.inference_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(128, 1024, 2)),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=1, strides=1, padding='same'),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.AveragePooling2D(
                    pool_size=(2, 2),strides=2,padding='same'),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.AveragePooling2D(
                    pool_size=(2, 2),strides=2,padding='same'),
                
                tf.keras.layers.Conv2D(
                    filters=128, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.Conv2D(
                    filters=128, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.AveragePooling2D(
                    pool_size=(2, 2),strides=2,padding='same'),
                
                tf.keras.layers.Conv2D(
                    filters=256, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.Conv2D(
                    filters=256, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.AveragePooling2D(
                    pool_size=(2, 2),strides=2,padding='same'),
                
                tf.keras.layers.Conv2D(
                    filters=256, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.Conv2D(
                    filters=256, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.AveragePooling2D(
                    pool_size=(2, 2),strides=2,padding='same'),
                
                tf.keras.layers.Conv2D(
                    filters=256, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.Conv2D(
                    filters=256, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.AveragePooling2D(
                    pool_size=(2, 2),strides=2,padding='same'),
                
                tf.keras.layers.Flatten(),
                LinearHyperbolic(512*2, Poincare(), 1, activation=tf.keras.layers.LeakyReLU(), output_hyp=True),
                tf.keras.layers.Dropout(0.25),
                LinearHyperbolic(128, Poincare(), 1, activation=tf.keras.layers.LeakyReLU(), output_hyp=True,input_hyp=True),
                tf.keras.layers.Dropout(0.25),
                LinearHyperbolic(32, Poincare(), 1, activation=tf.keras.layers.LeakyReLU(), output_hyp=True,input_hyp=True),
                tf.keras.layers.Dropout(0.25),

                LinearHyperbolic(latent_dim+latent_dim, Poincare(), 1,input_hyp=True),

          ]
        )
        
        self.generative_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                LinearHyperbolic(32, Poincare(), 1, activation=tf.keras.layers.LeakyReLU(), output_hyp=True),
                tf.keras.layers.Dropout(0.25),
                LinearHyperbolic(128, Poincare(), 1, activation=tf.keras.layers.LeakyReLU(), output_hyp=True,input_hyp=True),
                tf.keras.layers.Dropout(0.25),
                LinearHyperbolic(512*2, Poincare(), 1, activation=tf.keras.layers.LeakyReLU(), output_hyp=True,input_hyp=True),
                tf.keras.layers.Dropout(0.25),
                LinearHyperbolic(2*16*256, Poincare(), 1,input_hyp=True),


                tf.keras.layers.Reshape(target_shape=(2, 16, 256)),
                
                
                tf.keras.layers.UpSampling2D(
                    size = 2, interpolation='nearest'),
                tf.keras.layers.Conv2DTranspose(
                    filters=256, kernel_size=(2,16), strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                tf.keras.layers.Conv2DTranspose(
                    filters=256, kernel_size=3, strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                
                tf.keras.layers.UpSampling2D(
                    size = 2, interpolation='nearest'),
                tf.keras.layers.Conv2DTranspose(
                    filters=256, kernel_size=(2,16), strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                tf.keras.layers.Conv2DTranspose(
                    filters=256, kernel_size=3, strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                
                tf.keras.layers.UpSampling2D(
                    size = 2, interpolation='nearest'),
                tf.keras.layers.Conv2DTranspose(
                    filters=256, kernel_size=(2,16), strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                tf.keras.layers.Conv2DTranspose(
                    filters=256, kernel_size=3, strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                
                tf.keras.layers.UpSampling2D(
                    size = 2, interpolation='nearest'),
                tf.keras.layers.Conv2DTranspose(
                    filters=128, kernel_size=(2,16), strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                tf.keras.layers.Conv2DTranspose(
                    filters=128, kernel_size=3, strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                
                tf.keras.layers.UpSampling2D(
                    size = 2, interpolation='nearest'),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=(2,16), strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=3, strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                
                tf.keras.layers.UpSampling2D(
                    size = 2, interpolation='nearest'),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=(2,16), strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=3, strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                
                tf.keras.layers.Conv2DTranspose(
                    filters=2, kernel_size=1, strides=1, padding="SAME", activation='tanh'),
            ]
        )
        
    def sample(self, eps=None, apply_tanh=True):
        if eps is None:
            eps = tf.random.normal(shape=(2, self.latent_dim))
        return self.decode(eps, apply_tanh)

    def encode(self, x):
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_tanh=False):
        logits = self.generative_net(z)
        if apply_tanh:
            probs = tf.math.multiply(tf.math.tanh(logits),0.999999)
            return probs
        else:
            probs = tf.math.multiply(logits,0.999999)
        return logits       








class ECVAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(ECVAE, self).__init__()
        self.latent_dim = latent_dim
        
        self.conv = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(128, 1024, 2)),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=1, strides=1, padding='same'),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
        ])
        # * division

        self.embedding = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(128, 1024, 32)),
                tf.keras.layers.Permute((1,3,2)),
                LinearHyperbolic(256, Poincare(), 1, activation=tf.keras.layers.LeakyReLU(), output_hyp=True),
                tf.keras.layers.Dropout(0.25),
                LinearHyperbolic(64, Poincare(), 1, activation=tf.keras.layers.LeakyReLU(),  input_hyp=True),
                tf.keras.layers.Permute((1,3,2)),
                tf.keras.layers.Reshape((128, 16, 128)),
                tf.keras.layers.Permute((2,1,3)),
        ])

        self.conv_inter = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(128, 1024, 32)),
                tf.keras.layers.AveragePooling2D(
                    pool_size=(2, 2),strides=2,padding='same'),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.AveragePooling2D(
                    pool_size=(2, 2),strides=2,padding='same'),
                
                tf.keras.layers.Conv2D(
                    filters=128, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.Conv2D(
                    filters=128, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.AveragePooling2D(
                    pool_size=(2, 2),strides=2,padding='same'),
                
        ])

        # * union


        self.inference_net = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(16, 128, 256)),
                
                
                tf.keras.layers.Conv2D(
                    filters=256, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.Conv2D(
                    filters=256, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.AveragePooling2D(
                    pool_size=(2, 2),strides=2,padding='same'),
                
                tf.keras.layers.Conv2D(
                    filters=256, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.Conv2D(
                    filters=256, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.AveragePooling2D(
                    pool_size=(2, 2),strides=2,padding='same'),
                
                tf.keras.layers.Conv2D(
                    filters=256, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.Conv2D(
                    filters=256, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.AveragePooling2D(
                    pool_size=(2, 2),strides=2,padding='same'),
                
                tf.keras.layers.Conv2D(
                    filters=256, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.Conv2D(
                    filters=256, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.AveragePooling2D(
                    pool_size=(2, 2),strides=2,padding='same'),
                
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(latent_dim + latent_dim),

          ]
        )
        
        self.generative_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=1*8*256), # * venia con el modelo original


                tf.keras.layers.Reshape(target_shape=(1, 8, 256)),
                
                tf.keras.layers.UpSampling2D(
                    size = 2, interpolation='nearest'),
                tf.keras.layers.Conv2DTranspose(
                    filters=256, kernel_size=(2,16), strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                tf.keras.layers.Conv2DTranspose(
                    filters=256, kernel_size=3, strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                
                tf.keras.layers.UpSampling2D(
                    size = 2, interpolation='nearest'),
                tf.keras.layers.Conv2DTranspose(
                    filters=256, kernel_size=(2,16), strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                tf.keras.layers.Conv2DTranspose(
                    filters=256, kernel_size=3, strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                
                tf.keras.layers.UpSampling2D(
                    size = 2, interpolation='nearest'),
                tf.keras.layers.Conv2DTranspose(
                    filters=256, kernel_size=(2,16), strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                tf.keras.layers.Conv2DTranspose(
                    filters=256, kernel_size=3, strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                
                tf.keras.layers.UpSampling2D(
                    size = 2, interpolation='nearest'),
                tf.keras.layers.Conv2DTranspose(
                    filters=256, kernel_size=(2,16), strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                tf.keras.layers.Conv2DTranspose(
                    filters=256, kernel_size=3, strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                
                tf.keras.layers.UpSampling2D(
                    size = 2, interpolation='nearest'),
                tf.keras.layers.Conv2DTranspose(
                    filters=128, kernel_size=(2,16), strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                tf.keras.layers.Conv2DTranspose(
                    filters=128, kernel_size=3, strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                
                tf.keras.layers.UpSampling2D(
                    size = 2, interpolation='nearest'),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=(2,16), strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=3, strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                
                tf.keras.layers.UpSampling2D(
                    size = 2, interpolation='nearest'),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=(2,16), strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=3, strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                
                tf.keras.layers.Conv2DTranspose(
                    filters=2, kernel_size=1, strides=1, padding="SAME", activation='tanh'),
            
            ]
        )
        
    def sample(self, eps=None, apply_tanh=True):
        if eps is None:
            eps = tf.random.normal(shape=(2, self.latent_dim))
        return self.decode(eps, apply_tanh)

    def encode(self, x):
        first_conv = self.conv(x)
        embedding = self.embedding(first_conv)
        conv2 = self.conv_inter(first_conv)
        concat = tf.concat([embedding, conv2], axis=3)

        mean, logvar = tf.split( self.inference_net(concat), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape) 
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_tanh=False):
        logits = self.generative_net(z)
        if apply_tanh:
            probs = tf.math.multiply(tf.math.tanh(logits),0.999999)
            return probs
        else:
            probs = tf.math.multiply(logits,0.999999)
        return logits


