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
GAN-based deep networks proposed specifically for medical imaging projects (architechtures and loss functions to enhance reliability).
2. DCGAN
- Extract hierarchical features by learning down/up-sampling. Extracted features used to generate new ones.
- Must use batch normalization and leaky-ReLu for stability.
- prone to mode collapse.
3. cGAN (conditional GAN)
- Generator receives noise z and information c jointly. 
- Improves training and stability
- Generator and Discriminator follow U-net and MGAN, combine l1 loss and adverserial loss to make images similar to ground truth images.
4. MGAN
- Style transfer
- uses pre-trained VGG19 network with fixed weights to extract high-level features. 
- High level features preserve image content.



Relevant papers:
-Szegedy et al [5] : adverserial training
-Goodfellow [2]: GAN's
-Radford [6]: DCGAN 
-Mirza [7]: cGAN
-8 : U-net
-9 : MGAN

