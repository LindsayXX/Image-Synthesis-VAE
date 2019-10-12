# Image Synthesis Using Variational Autoencoders with Adversarial Learning

## Reproduction & comparison study
In this project, we analyze the adversarial generator encoder ([AGE](https://arxiv.org/pdf/1807.06358.pdf)) and the introspective autoencoder ([IntroVAE](https://arxiv.org/pdf/1807.06358.pdf)). Here is the structure and training flow of 1)VAE, 2)AGE, and 3)IntroVAE:

And the detailed network architecture for encoding images of 128\*128 resolution in IntroVAE(top) and AGE(bottom).

We implement the models from scratch and apply them to CIFAR-10 and CelebA image datasets for unconditional image generation and reconstruction tasks. For CIFAR-10, we evaluate AGE quantitatively and our model reaches an Inception score of 2.90. AGE does not converge on the higher resolution dataset CelebA, whereas IntroVAE reaches stable training but suffer from blurriness and sometimes mode collapse.

## Dataset

## Experiments

## Result

Here are some examples of our results:



### Useful link:


