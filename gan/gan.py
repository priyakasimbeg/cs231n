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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    real_loss = cross_entropy(tf.ones_like(logits_real), logits_real)
    fake_loss = cross_entropy(tf.zeros_like(logits_fake), logits_fake)
    loss = real_loss + fake_loss

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    loss = cross_entropy(tf.ones_like(logits_fake), logits_fake)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    loss = 1/2 * (tf.reduce_mean(tf.math.square(scores_real - 1)) +  tf.reduce_mean(tf.math.square(scores_fake)))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
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
    # Your implementation should be vectorized and not require any loops!
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    _, H, W, C = img.shape
    shift_h = tf.roll(img, shift=-1, axis=1)
    diff = tf.reshape(tf.transpose(shift_h - img, perm=(1,0,2,3)), (H, -1))[:-1,:]
    loss = tf.reduce_sum(tf.math.square(diff))
    shift_v = tf.roll(img, shift=-1, axis=2)
    diff = tf.reshape(tf.transpose(shift_h - img, perm=(2,0,1,3)), (W, -1))[:-1,:]
    loss += tf.reduce_sum(tf.math.square(diff))
    return tv_weight * loss

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

def get_solvers(learning_rate=1e-3, beta1=0.5):
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

    D_solver = tf.optimizers.Adam(1e-3, 0.3)
    G_solver = tf.optimizers.Adam(1e-3, 0.5)


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return D_solver, G_solver

# a giant helper function
def run_a_gan(D, G, D_solver, G_solver, discriminator_loss, generator_loss,\
              show_every=20, print_every=20, batch_size=128, num_epochs=10, noise_size=NOISE_DIM):
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
    epid = EPID(batch_size=batch_size)
    phantom = PHANTOM(batch_size=batch_size) 

    g_errors = []
    d_errors = []
    iter_count = 0
    for epoch in range(num_epochs):
        for x, y in zip(epid, phantom):
            with tf.GradientTape() as tape:
                real_data = x
                logits_real = D(preprocess_img(real_data))

#                 g_fake_seed = sample_noise(batch_size, noise_size)
            
                g_fake_seed = y
                fake_images = G(g_fake_seed)

                logits_fake = D(tf.reshape(fake_images, [batch_size, INPUT]))

                d_total_error = discriminator_loss(logits_real, logits_fake)
                d_gradients = tape.gradient(d_total_error, D.trainable_variables)      
                D_solver.apply_gradients(zip(d_gradients, D.trainable_variables))
            
            with tf.GradientTape() as tape:
                g_fake_seed = sample_noise(batch_size, noise_size)
                fake_images = G(g_fake_seed)

                gen_logits_fake = D(tf.reshape(fake_images, [batch_size, INPUT]))
                g_error = generator_loss(gen_logits_fake)
                g_gradients = tape.gradient(g_error, G.trainable_variables)      
                G_solver.apply_gradients(zip(g_gradients, G.trainable_variables))

            if (iter_count % show_every == 0):
                print('Epoch: {}, Iter: {}, D: {:.4}, G:{:.4}'.format(epoch, iter_count,d_total_error,g_error))
                imgs_numpy = fake_images.cpu().numpy()
                show_images(imgs_numpy[0:16])
                plt.show()
            iter_count += 1
            
            g_errors.append(g_error)
            d_errors.append(d_total_error)
    
    # random noise fed into our generator
    z = sample_noise(batch_size, noise_size)
    # generated images
    G_sample = G(z)
    print('Final images')
    show_images(G_sample[:16])
    plt.show()
    
    return g_errors, d_errors

class DCGAN():
    def __init__(self):
        self.discriminator = discriminator()
        self.generator = generator()
  
    def discriminator():
        """Compute discriminator score for a batch of input images.

        Inputs:
        - x: TensorFlow Tensor of flattened input images, shape [batch_size, 784]

        Returns:
        TensorFlow Tensor with shape [batch_size, 1], containing the score 
        for an image being real for each input image.
        """
        model = tf.keras.models.Sequential([
            # TODO: implement architecture
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ])
        return model
    
    def generator(noise_dim=NOISE_DIM):
        """Generate images from a random noise vector.

        Inputs:
        - z: TensorFlow Tensor of random noise with shape [batch_size, noise_dim]

        Returns:
        TensorFlow Tensor of generated images, with shape [batch_size, 784].
        """
        model = tf.keras.models.Sequential([
        # TODO: implement architecture
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        tf.keras.layers.Dense(1024, activation='relu', input_shape=(noise_dim,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(7 * 7 * 128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Reshape((7, 7, 128)),
        tf.keras.layers.Conv2DTranspose(64, 4, strides=(2, 2), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2DTranspose(1, 4, strides=(2, 2), activation='tanh', padding='same')

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            ])
        return model
    
    
class SRGAN():
    
    def __init__(self):
        self.discriminator = discriminator()
        self.generator = generator()
    
    def discriminator():
        """Compute discriminator score for a batch of input images.

        Inputs:
        - x: TensorFlow Tensor of flattened input images, shape [batch_size, 784]

        Returns:
        TensorFlow Tensor with shape [batch_size, 1], containing the score 
        for an image being real for each input image.
        """
        model = tf.keras.models.Sequential([
            # TODO: implement architecture
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            tf.keras.layers.Reshape(target_shape=(28,28,1), input_shape=(784,)),
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

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ])
        return model

    
    def generator():
        model = ResNetGenerator(NUM_RESIDUAL_BLOCKS)
        return model


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
        
    def call(self, x):
        for block in range(self.num_blocks):
            rb = ResidualBlock(self.kernels, self.filters)
            x += rb(x)
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
        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=[9,9],
                                           strides=1, padding='same')
        self.prelu1 = tf.keras.layers.LeakyReLU()
        
        self.conv2 = tf.keras.layers.Conv2D(filters=1, kernel_size=[9,9], 
                                           strides=1, padding='same',
                                            activation='tanh')
        self.num_blocks = num_blocks
        
    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.prelu1(x)
        kernels, filters = [[3, 3], [3, 3]], [64, 64]
        rbs = ResBlockSequence(kernels, filters, self.num_blocks)
        x += rbs(x)
        sb = SmoothingBlock(kernels, [256, 256])
        x = sb(x)
        x = self.conv2(x) 
        return x

