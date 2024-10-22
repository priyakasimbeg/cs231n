import tensorflow as tf
import numpy as np
import os
from gan.utils import *
from gan.metrics import *
from tensorflow.keras.applications.vgg19 import VGG19
# import tensorflow.contrib.gan.losses.wargs as wargs

# Constants
NUM_RESIDUAL_BLOCKS = 7
DIM = 32
NOISE_DIM = DIM * DIM
INPUT = DIM * DIM


## Loss functions
def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss described above.
    
    Inputs:
    - logits_real: Tensor of shape (N, 1) giving scores for the real data.
    - logits_fake: Tensor of shape (N, 1) giving scores for the fake data.
    
    Returns:
    - loss: Tensor containing (scalar) the loss for the discriminator.
    """
    loss = None
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    real_loss = cross_entropy(tf.ones_like(logits_real), logits_real)
    fake_loss = cross_entropy(tf.zeros_like(logits_fake), logits_fake)
    loss = real_loss + fake_loss

    return loss

def generator_loss(logits_fake):
    """
    Computes the generator loss described above.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    loss = None
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    loss = cross_entropy(tf.ones_like(logits_fake), logits_fake)

    return loss

def ls_discriminator_loss(scores_real, scores_fake):
    """
    Compute the Least-Squares GAN loss for the discriminator.
    
    Inputs:
    - scores_real: Tensor of shape (N, 1) giving scores for the real data.
    - scores_fake: Tensor of shape (N, 1) giving scores for the fake data.
    
    Outputs:
    - loss: A Tensor containing the loss.
    """
    loss = None
    loss = 1/2 * (tf.reduce_mean(tf.math.square(scores_real - 1)) +  tf.reduce_mean(tf.math.square(scores_fake)))

    return loss

def ls_generator_loss(scores_fake):
    """
    Computes the Least-Squares GAN loss for the generator.
    
    Inputs:
    - scores_fake: Tensor of shape (N, 1) giving scores for the fake data.
    
    Outputs:
    - loss: A Tensor containing the loss.
    """
    loss = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    loss = 1/2 * tf.reduce_mean(tf.math.square(scores_fake - 1))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss

def tv_loss(img, tv_weight):
    """
    Compute total variation loss.
    
    Inputs:
    - img: Tensor of shape (1, H, W, 3) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    
    Returns:
    - loss: Tensor holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    _, H, W, C = img.shape
    shift_h = tf.roll(img, shift=-1, axis=1)
    diff = tf.reshape(tf.transpose(shift_h - img, perm=(1,0,2,3)), (H, -1))[:-1,:]
    loss = tf.reduce_sum(tf.math.square(diff))
    shift_v = tf.roll(img, shift=-1, axis=2)
    diff = tf.reshape(tf.transpose(shift_h - img, perm=(2,0,1,3)), (W, -1))[:-1,:]
    loss += tf.reduce_sum(tf.math.square(diff))
    return tv_weight * loss

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

def get_solvers(dlr=1e-4, glr=1e-3, beta1=0.5):
    """Create solvers for GAN training.
    
    Inputs:
    - learning_rate: learning rate to use for both solvers
    - beta1: beta1 parameter for both solvers (first moment decay)
    
    Returns:
    - D_solver: instance of tf.optimizers.Adam with correct learning_rate and beta1
    - G_solver: instance of tf.optimizers.Adam with correct learning_rate and beta1
    """
    D_solver = None
    G_solver = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    D_solver = tf.optimizers.RMSprop(learning_rate=dlr)
    G_solver = tf.optimizers.Adam(learning_rate=glr, beta_1=beta1)


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return D_solver, G_solver

# a giant helper function
def run_a_gan(GAN, D_solver, G_solver, L, L_validation,
              show_every=20, print_every=20, batch_size=50, num_epochs=10, 
              input_size=NOISE_DIM, 
              late_dlr=4e-7, late_glr=5e-7, 
              gen_steps_per_discrimination=1, model_name='gan'):
    """Train a GAN for a certain number of epochs.
    Inputs:
    - D: Discriminator model
    - G: Generator model
    - D_solver: an Optimizer for Discriminator
    - G_solver: an Optimizer for Generator
    - generator_loss: Generator loss
    - discriminator_loss: Discriminator loss
    Returns:
        Nothing
    """
    D = GAN.D
    G = GAN.G

    epid = EPID(batch_size=batch_size)
    phantom = PHANTOM(batch_size=batch_size, fake=True) 

    g_errors = []
    d_errors = []
    iter_count = 0
    
    ## Setup storing stuff
    if not os.path.exists('train/{}'.format(model_name)):
        os.makedirs('train/{}'.format(model_name))
    
    if not os.path.exists('train/{}/images'.format(model_name)):
        os.makedirs('train/{}/images'.format(model_name))
    
    for epoch in range(num_epochs):

        #Decrease learning rate for later epochs
        if epoch == 150:
            D_solver, G_solver = get_solvers(dlr=late_dlr, glr=late_glr)

        for x, y in zip(epid, phantom):

            # Update Discriminator
            with tf.GradientTape() as tape:
                real_data = x
                intermediate_real, logits_real = D(preprocess_img(real_data))

#                 g_fake_seed = sample_noise(batch_size, noise_size)
                g_fake_seed = y
                fake_images = G(g_fake_seed)

                intermediate_fake, logits_fake = D(tf.reshape(fake_images, [batch_size, INPUT]))
                
                #Compute discriminator loss
                L.update_discriminator_loss(logits_real, logits_fake)
                d_total_error = L.get_discriminator_loss()
                
                
                d_gradients = tape.gradient(d_total_error, D.trainable_variables)      
                D_solver.apply_gradients(zip(d_gradients, D.trainable_variables))
                
                ## Validation
                intermediate_real_val, logits_real_val = D(preprocess_img(epid.val))
                fake_images_val = G(phantom.val)
                intermediate_fake_val, logits_fake_val = D(tf.reshape(fake_images_val, [batch_size, INPUT]))
                L_validation.update_discriminator_loss(logits_real_val, logits_fake_val)
                _ = L_validation.get_discriminator_loss()
                                                           
                       
            for j in range(gen_steps_per_discrimination):
                with tf.GradientTape() as tape:
                    g_fake_seed = y
                    fake_images = G(g_fake_seed)

                    gen_intermediate_fake, gen_logits_fake = D(tf.reshape(fake_images, [batch_size, INPUT]))

                    #Compute generator loss
                    L.update_generator_loss(real_data, fake_images, gen_logits_fake, 
                                            intermediate_real, gen_intermediate_fake, GAN.VGG)
                    g_error = L.get_generator_loss()
                    
                    #Decrease generator learning steps when error is low
                    if g_error < 1: 
                        gen_steps_per_discrimination = max(2, gen_steps_per_discrimination)
                    
                    g_gradients = tape.gradient(g_error, G.trainable_variables)      
                    G_solver.apply_gradients(zip(g_gradients, G.trainable_variables))
                    
                    ## Validation
                    g_fake_seed_val = phantom.val
                    fake_images_val = G(phantom.val)
                    gen_intermediate_fake_val, gen_logits_fake_val = D(tf.reshape(fake_images_val, [batch_size, INPUT]))
                    L_validation.update_generator_loss(epid.val, fake_images_val, gen_logits_fake_val, intermediate_real_val,
                                                       gen_intermediate_fake_val, GAN.VGG)
                    _ = L_validation.get_generator_loss()
                    

            if (iter_count % show_every == 0):
                print('Epoch: {}, Iter: {}, D: {:.4}, G:{:.4}'.format(epoch, iter_count,d_total_error,g_error))
                imgs_numpy = fake_images.cpu().numpy()
                show_images(imgs_numpy[0:16])
                plt.savefig('train/{}/images/{}.png'.format(model_name, iter_count))
                plt.show()

            iter_count += 1
            g_errors.append(g_error)
            d_errors.append(d_total_error)
            
    G.save_weights('train/{}/{}.h5'.format(model_name, model_name))
    
    # random noise fed into our generator
    phantom = PHANTOM(batch_size, shuffle=True)
    # generated images
    for x in phantom: 
        G_sample = G(x)
        break
    print('Final images')
    show_images(G_sample[:16])
    plt.show()
    plt.savefig('train/{}/images/final_images.png'.format(model_name))
    
    return g_errors, d_errors

class Loss():
    def __init__(self, gen_reg, tv_reg, feat_reg, l1_reg=True, vgg_reg=False, 
                 ls_disc=False):
        self.losses = {}
        self.losses['Generator loss'] = []
        self.losses['Total Variation loss'] = []
        self.losses['L1 content loss'] = []
        self.losses['MSE loss'] = []
        self.losses['Feature content loss'] = []
        self.losses['VGG content loss'] = []
        self.losses['Total loss'] = []
        self.losses['Discriminator loss'] = []
        self.l1_reg = l1_reg
        self.gen_reg = gen_reg
        self.tv_reg = tv_reg
        self.feat_reg = feat_reg
        self.vgg_reg = vgg_reg
        self.ls_disc = ls_disc

    def update_generator_loss(self, real_data, fake_images, gen_logits_fake, 
                    intermediate_real, gen_intermediate_fake, perceptual_feature_extractor):
        
        self.losses['Generator loss'].append(generator_loss(gen_logits_fake))
        self.losses['Total Variation loss'].append(tv_loss(fake_images))
        self.losses['L1 content loss'].append(l1_content_loss(real_data, fake_images))
        self.losses['MSE loss'].append(mse_loss(real_data, fake_images)) 
        self.losses['Feature content loss'].append(feature_mapping_loss(intermediate_real, gen_intermediate_fake))
        self.losses['VGG content loss'].append(vgg_content_loss(real_data, fake_images,
                                                                perceptual_feature_extractor))
    
    def update_discriminator_loss(self, logits_real, logits_fake):
        if self.ls_disc:
            loss = discriminator_loss(logits_real, logits_fake)
        else:
            loss = ls_discriminator_loss(logits_real, logits_fake)
        
        #Record iteration loss
        self.losses['Discriminator loss'].append(loss)
        
    def get_generator_loss(self):
        loss = self.gen_reg * self.losses['Generator loss'][-1]
        loss += self.tv_reg * self.losses['Total Variation loss'][-1] 
        #Use branching to avoid extraneous gradient computations
        if self.l1_reg:
            loss += self.losses['L1 content loss'][-1]
        else:
            loss += self.losses['MSE loss'][-1]
        if self.vgg_reg:
            loss += self.feat_reg * self.losses['VGG content loss'][-1]
        else:
            loss += self.feat_reg * self.losses['Feature content loss'][-1]
        
        #Record total iteration loss
        self.losses['Total loss'].append(loss)
        return loss
    
    def get_discriminator_loss(self):
        return self.losses['Discriminator loss'][-1]
    
    def get_loss_plots(self):
        plots = {}
        for loss in self.losses:
            fig = plt.figure()
            plt.plot(self.losses[loss], figure=fig)
            plt.xlabel('Iteration number')
            plt.ylabel('Loss')
            plt.title(loss)
            plots[loss] = fig
        return plots

def vgg_content_loss(real_data, gen_data, feature_extractor):
        N, D = tf.shape(real_data)
        real_data = tf.reshape(real_data, (N, DIM, DIM, 1))
        gen_data = tf. reshape(gen_data, (N, DIM, DIM, 1))

        real_data = tf.image.grayscale_to_rgb(real_data)
        gen_data = tf.image.grayscale_to_rgb(gen_data)
        loss = tf.reduce_mean(tf.norm(feature_extractor(real_data) - feature_extractor(gen_data)))
        return loss
    
def tv_loss(fake_data):
    fake_data = tf.reshape(fake_data, (-1, DIM, DIM, 1))
    return tf.reduce_sum(tf.image.total_variation(fake_data))

def l1_content_loss(real_data, fake_data):
    return tf.norm(real_data - fake_data, ord=1) 

def mse_loss(real_data, fake_data):
    return tf.square(tf.reduce_sum(real_data - fake_data)) / tf.cast(tf.size(real_data), tf.float32)

def feature_mapping_loss(intermediate_real, gen_intermediate_fake):
    return tf.reduce_sum(tf.square(tf.reduce_mean(intermediate_real, axis=0) 
                                   - tf.reduce_mean(gen_intermediate_fake, axis=0)))
    
def effective_generator_loss(real_data, fake_data, gen_logits_fake, batch_size, 
                             tv_reg, gen_reg, feat_reg,
                             intermediate_real, gen_intermediate_fake):
    
    gen_loss = generator_loss(gen_logits_fake)
    l1_loss = tf.norm(real_data - tf.reshape(fake_data, [batch_size, INPUT]), ord=1) 
#     mse_loss = mse(real_data, tf.reshape(fake_data, [batch_size, INPUT])) 
    tv_loss = tv_reg * tf.reduce_sum(tf.image.total_variation(fake_data))
    fm_loss = feature_mapping_loss(intermediate_real, gen_intermediate_fake)
    
    return gen_reg * gen_loss + l1_loss + feat_reg * fm_loss + tv_reg * tv_loss

class DCGAN():
    def __init__(self):
        self.D = self.discriminator()
        self.G = self.generator()
  
    def discriminator(self):
        """Compute discriminator score for a batch of input images.

        Inputs:
        - x: TensorFlow Tensor of flattened input images, shape [batch_size, 784]

        Returns:
        TensorFlow Tensor with shape [batch_size, 1], containing the score 
        for an image being real for each input image.
        """
        model = tf.keras.models.Sequential([
            # TODO: implement architecture
            tf.keras.layers.Reshape((DIM, DIM, 1), input_shape=(INPUT, )),
            tf.keras.layers.Conv2D(32, 5, padding='valid',),
            tf.keras.layers.LeakyReLU(alpha=0.01),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
            tf.keras.layers.Conv2D(64, 5, padding='valid'),
            tf.keras.layers.LeakyReLU(alpha=0.01),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4 * 4 * 64),
            tf.keras.layers.LeakyReLU(alpha=0.01),
            tf.keras.layers.Dense(1)
        ])
        return model
    
    def generator(self, noise_dim=NOISE_DIM):
        """Generate images from a random noise vector.

        Inputs:
        - z: TensorFlow Tensor of random noise with shape [batch_size, noise_dim]

        Returns:
        TensorFlow Tensor of generated images, with shape [batch_size, 784].
        """
        model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1024, activation='relu', input_shape=(noise_dim,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(7 * 7 * 128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Reshape((7, 7, 128)),
        tf.keras.layers.Conv2DTranspose(64, 4, strides=(2, 2), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2DTranspose(1, 4, strides=(2, 2), activation='tanh', padding='same')
        ])
        return model
    
    
class SRGAN():
    
    def __init__(self, num_residual_blocks, intermediate_layer=6):
        self.l = intermediate_layer
        self.D = self.discriminator()
        self.G = self.generator(num_residual_blocks)
        self.VGG = self.setup_vgg()


    def discriminator(self):
        """Compute discriminator score for a batch of input images.

        Inputs:
        - x: TensorFlow Tensor of flattened input images, shape [batch_size, 784]

        Returns:
        TensorFlow Tensor with shape [batch_size, 1], containing the score 
        for an image being real for each input image.
        """
        model = tf.keras.models.Sequential([
            tf.keras.layers.Reshape(target_shape=(DIM,DIM,1), input_shape=(INPUT,)),
            tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=1,
                                  padding='same'),
            tf.keras.layers.LeakyReLU(alpha=0.01),

            tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=2,
                                  padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.01),

            tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=1,
                                  padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.01),

            tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=2,
                                  padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.01),

            tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=1,
                          padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.01),

            tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=2,
                          padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.01),

            tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=1,
                          padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.01),

            tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=2,
                          padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.01),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=1024),
            tf.keras.layers.LeakyReLU(alpha=0.01),
            tf.keras.layers.Dense(units=1)
        ])
        
        ## Add feature mapping
        features_list_all = [layer.output for layer in model.layers]
        features_list = [features_list_all[self.l], features_list_all[-1]]
    
        feat_extraction_model = tf.keras.Model(inputs=model.input, outputs=features_list)
        
        return feat_extraction_model

    def generator(self, num_residual_blocks):
        model = ResNetGenerator(num_residual_blocks)
        return model

    def setup_vgg(self, ):
        image_shape = (32, 32, 3) # minimum shape
        vgg = VGG19(include_top=False, weights='imagenet', input_shape=image_shape)
        loss_model = tf.keras.Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
        loss_model.trainable = False
        return loss_model




class ResidualBlock(tf.keras.Model):
    def __init__(self, kernels, filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filters[0], 
                                            kernel_size=kernels[0], 
                                            strides=1, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.prelu1 = tf.keras.layers.LeakyReLU()
        self.conv2 = tf.keras.layers.Conv2D(filters=filters[1], 
                                            kernel_size=kernels[1], 
                                            strides=1, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.prelu2 = tf.keras.layers.LeakyReLU()

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.prelu2(x)
        return x  
    
class ResBlockSequence(tf.keras.Model):
    def __init__(self, kernels, filters, num_blocks):
        super(ResBlockSequence, self).__init__()
        self.kernels = kernels
        self.filters = filters
        self.num_blocks = num_blocks
        
        self.conv = tf.keras.layers.Conv2D(filters=64, kernel_size=[3,3],
                                           strides=1, padding='same')
        self.bn = tf.keras.layers.BatchNormalization()
        self.rb = {}
        for block in range(num_blocks):
            self.rb[block] = ResidualBlock(self.kernels, self.filters)
        
    def call(self, x):
        for block in range(self.num_blocks):
            x += self.rb[block](x)
        x = self.conv(x)
        x = self.bn(x)
        return x
              
class SmoothingBlock(tf.keras.Model):
        def __init__(self, kernels, filters):
            super(SmoothingBlock, self).__init__()
            self.conv1 = tf.keras.layers.Conv2D(filters=filters[0], 
                                                kernel_size=kernels[0], 
                                                strides=1, padding='same')
            #TODO: pixel shuffler
            self.prelu1 = tf.keras.layers.LeakyReLU()
            self.conv2 = tf.keras.layers.Conv2D(filters=filters[1], 
                                                kernel_size=kernels[1], 
                                                strides=1, padding='same')
            self.prelu2 = tf.keras.layers.LeakyReLU()
            
        def call(self, x):
            x = self.conv1(x)
            #TODO: implement shuffle
            x = self.prelu1(x)
            x = self.conv2(x)
            #TODO: implement shuffle
            x = self.prelu2(x)
            return x

class ResNetGenerator(tf.keras.Model):
    def __init__(self, num_blocks):
        super(ResNetGenerator, self).__init__() 
        self.shapein = tf.keras.layers.Reshape((DIM, DIM, 1), input_shape=(INPUT,))
        self.conv1 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=[9,9],
                                           strides=1, padding='same')
        self.prelu1 = tf.keras.layers.LeakyReLU()
        
        self.conv2 = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=[9,9], 
                                           strides=1, padding='same',
                                           activation='tanh')
        self.shapeout = tf.keras.layers.Reshape((INPUT,))
        self.num_blocks = num_blocks
        kernels, filters = [[3, 3], [3, 3]], [64, 64]
        self.rbs = ResBlockSequence(kernels, filters, self.num_blocks)
        self.sb = SmoothingBlock(kernels, [256, 256])
        
    def call(self, x, training=False):
        x = self.shapein(x)
        x = self.conv1(x)
        x = self.prelu1(x)
        x += self.rbs(x)
        x = self.sb(x)
        x = self.conv2(x)
        x = self.shapeout(x)
        return x

