# Digit-Recognition

Exploration of classification of handwritten digits leveraging MNIST dataset using PyTorch.

## 1. PCA

PCA is insufficient in handwritten digit classification of the MNIST dataset: 
![Classification of digits using only 2 component PCA](https://github.com/nuj02/Digit-Recognition/blob/main/1%20PCA/figures/PCA_Classification.png?raw=true)

## 2. Dimension Reduction Using Autoencoders

Inspiration by ["Reducing the Dimensionality of Data with Neural Networks" by Hintonand Salakuhtdinov](https://www.science.org/doi/10.1126/science.1127647)

Therefore nonlinear PCA is explored using an autoencoder: encoding high-dimensional data to lower dimensions and recovering high-dimensional information from the dimension-reduced representation.