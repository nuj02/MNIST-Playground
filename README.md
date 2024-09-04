# MNIST Playground

Exploration of MNIST dataset

## Dimension Reduction

### PCA

Linear PCA is insufficient in differentiating different handwritten digits when using only two principal components.
![MNIST Analysis using 2-Component PCA](./PCA/figures/PCA_Classification.png)

## Autoencoders

Inspiration by ["Reducing the Dimensionality of Data with Neural Networks" by Hinton and Salakuhtdinov](https://www.science.org/doi/10.1126/science.1127647)

An autoencoder with an encoder architecture of 768-1000-500-250-2 is used with a symmetric decoder. With a schedule learning rate of 0.1, 0.01, 0.001 for 50 epochs each. The projection onto the 2-dimensional latent space is used as a non-linear PCA.
![MNIST Analysis using 2-dimensional latent space](./Autoencoder/figures/output.png)