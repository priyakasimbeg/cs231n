## Summary

Review of GAN-based architechtures for medical imagge processing applications including:
- synthesis
- de-noising
- reconstruction

Classification of GANS:
- latent space to image
- image to image

Section 4.5: Image denoising.

Overview of GAN architectures:
1. GAN: Generator G, Discriminator D. Min-max optimization function. 
Challenges
- mode colllapsing (When G collapses to map all latent space inputs to the same data).
- instability: generation of different outputs for same input. Use batchnormalizaton.
- data distribution compared using Jensen-Shannon divergence.
GAN-based deep networks proposed specifically for medical imaging projects (architechtures and loss functions to enhance reliability).
2. DCGAN
- Extract hierarchical features by learning down/up-sampling. Extracted features used to generate new ones.
- Must use batch normalization and leaky-ReLu for stability.
- prone to mode collapse.
3. cGAN (conditional GAN)
- Generator receives noise z and information c jointly. 
- Improves training and stability
- Generator and Discriminator follow U-net and MGAN, combine l1 loss and adverserial loss to make images similar to ground truth images.
4. MGAN *
- Style transfer
- uses pre-trained VGG19 network with fixed weights to extract high-level features. 
- High level features preserve image content.
- G uses feature maps to produce image with target texture
- discriminator uses FCN to discriminate VGG19 features
5. cycleGAN:
- relationship between image domains learned from unpaired data
6. AC-GAN:
- Discriminator splits into discriminator and auxillary classifier 
7. WGAN:
- uses EARTH Mover or Wasserstein-1 distance instead of JS divergence.
- slow optimization process.
8. LSGAN
- parameters added in loss function to avoid gradient vanishing
- Loss function has squared error terms.


## Applicatons
1. Conditional Image Synthesis : CT from MR, MR from CT, Retinal Image Synthesis, PET from CT, PET from MRI, Ultrasound.
X-ray. 
2. De-noising: Low dose results in low Signal-to-Noise Ratio. 
- Wolterink: learn texture info from small amount of paired data using a combination of voxel-wise ME and adversarial loss. 
- Wang: cGAN to remove artifacts from images
- Yang: combination of a perceputal loss and Wasserstein loss for numerical stability.
- Yi: sharpness of denoised image: combines pixel-wise loss, adverserial loss and sharpness mapping loss.
Commmon metrics for denoising (PSNR, MSE, SSIM, SD and mean).


Relevant papers:
- Szegedy et al [5] : adverserial training
- Goodfellow [2]: GAN's
- Radford [6]: DCGAN 
- Mirza [7]: cGAN
- 8 : U-net
- 9 : MGAN
- 101 : Wolterink voxel-wise MSE and adversarial loss
- 102: Wang cGAN to remove artifacts
- 103: Yang 
- 104: Yi

