import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import gan.gan as gan
import gan.utils as utils
import gan.metrics as metrics

mpl.rcParams['lines.linewidth'] = 5
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['legend.fontsize'] = 16
mpl.rcParams['axes.titlesize'] = 18

N_RES = 7 
BATCH_SIZE = 50

def plot_train_val_loss(train_loss, val_loss, model_name):

    train_losses_g = np.array(train_loss.losses['Total loss'])
    val_losses_g = np.array(val_loss.losses['Total loss'])
    
    train_losses_d = np.array(train_loss.losses['Discriminator loss'])
    val_losses_d = np.array(val_loss.losses['Discriminator loss'])

    iterations_g = range(0, len(train_losses_g))
    iterations_d = range(0, len(train_losses_d))

    # G Loss
    plt.figure(figsize=(8, 6))
    plt.plot(iterations_g, train_losses_g, label= 'Train Loss')
    plt.plot(iterations_g, val_losses_g, label = 'Validation Loss')
    
    plt.ylabel('Loss')
    plt.xlabel('Iterations')
    plt.legend()
    
    plt.title('Generator Losses')
    
    plt.savefig('train/{}/{}_G_loss.png'.format(model_name, model_name))
    plt.show()
    
    # D loss
    plt.figure(figsize=(8, 6))
    plt.plot(iterations_d, train_losses_d, label= 'Train Loss')
    plt.plot(iterations_d, val_losses_d, label = 'Validation Loss')
    
    plt.ylabel('Loss')
    plt.xlabel('Iterations')
    plt.legend()
    
    plt.title('Discriminator Losses')
    plt.savefig('train/{}/{}_D_loss.png'.format(model_name, model_name))
    plt.show()
    
    
def load_gan(model_name):
    trained_gan = gan.SRGAN(N_RES)
    
    for x in utils.PHANTOM(batch_size=BATCH_SIZE):
        trained_gan.G(x)
        break
        
    trained_gan.G.load_weights('train/{}/{}.h5'.format(model_name, model_name))
    
    return trained_gan.G


def runTest(model_name, show_n=9, labels=['(a)', '(b)']):
    
    model = load_gan(model_name)

    epid = utils.EPID(batch_size=BATCH_SIZE)
    test_in = epid.test
    test_out = model(test_in)
    
    fig = utils.show_images_set([test_in[:show_n], test_out[:show_n]], labels = labels)
    fig.savefig('train/{}/{}_test.png'.format(model_name, model_name))
    
    return test_in, test_out, fig


def runEvaluation(model_name, i = 4):
    
    test_in, test_out, fig = runTest(model_name)
    test_in_tensor = utils.array_batch_to_tensor(test_in)
    test_out_tensor = utils.array_batch_to_tensor(test_out)

    psnr_stats = metrics.psnr(test_in_tensor, test_out_tensor)
    ssim_stats = metrics.ssim(test_in_tensor, test_out_tensor)
    
    #Plot single illustrative example
    label = 'PSNR: {:.2f}     SSIM: {:.2f}'.format(psnr_stats[i], ssim_stats[i])
    image_set = np.array([[test_in[i]], [test_out[i]]])
    
    fig, axes = plt.subplots(1, 2)
    plt.axis('off')

    for j, images in enumerate(image_set):
        image = images[0]
        side = int(np.sqrt((image.shape)[0]))
        image = np.reshape(image, (side, side))
        

        ax = axes[j]
        
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        
        ax.imshow(image,)
   
    fig.suptitle(label, fontsize=20)
    plt.tight_layout()
    plt.savefig('train/{}/{}_example.png'.format(model_name, model_name))
    plt.show()
  
    return np.mean(psnr_stats), np.mean(ssim_stats)


