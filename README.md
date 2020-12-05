# Autoencoders

This repo contains simple explorations of autoencoders in different settings using Tensorflow and Keras.

- [Denoising autoencoders](https://github.com/danieltsoukup/autoencoders/blob/master/denoising_autoencoders.ipynb)

<img src="assets/denoising_vector_field.png" 
alt="denoising vector field"/>

- [Convolutional variational autoencoders for dimension reduction](https://github.com/danieltsoukup/autoencoders/blob/master/variational_autoencoders.ipynb)

<img src="assets/tsne_latent.png" 
alt="tsne clustering"/>

- [Randomized Autoencoder Ensembles for Outlier Detection](https://github.com/danieltsoukup/autoencoders/blob/master/outlier_detection_with_autoencoders.ipynb)

<img src="assets/wordline_features.png" 
alt="wordline features"/>

The notebooks contain
- useful utility functions for tensorboard logging,
- constructing custom Keras models such as with weight masking,
- custom training loops for variational autoencoders,
- custom data loaders that change the input per epoch for adaptive learning.

Resources:

- [Awesome anomaly detection resources](https://github.com/yzhao062/anomaly-detection-resources)