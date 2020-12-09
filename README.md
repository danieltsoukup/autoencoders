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

The main experiment with the RandNet architecture [(see Chang et at)](https://paperswithcode.com/paper/randnet-deep-learning-with-compressed)  and 
adaptive learning can be called by the `train_ensemble.py` script.
The Wordline file needs to be downloaded by hand to a `data` folder (the script should run with other data sets if you change the path and input shape).

The notebooks contain
- useful utility functions for tensorboard logging (both metrics and images),
- constructing custom Keras models: weight masking and custom training loops for variational autoencoders,
- custom data loaders that change the model input per epoch for adaptive sampling. 
We experiment with a batch-adaptive method that increases the batch size over the epochs. This
results in covering more of the training data as we progress in learning.

More ideas to explore:
- Sparse autoencoders
- Semantic hashing for texts
- Layer-wise pretraining of deep autoencoders

Resources:

- [Awesome anomaly detection resources](https://github.com/yzhao062/anomaly-detection-resources)