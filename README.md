# Autoencoders

This repo contains explorations of autoencoders in different settings using Tensorflow and Keras.

The main experiment focuses on the RandNet architecture [(see Chen et at)](https://saketsathe.net/downloads/autoencode.pdf) for unsupervised anomaly detection; the model training can be run by calling the `train_ensemble.py` script.
The [Wordline](https://www.kaggle.com/mlg-ulb/creditcardfraud) file needs to be downloaded by hand to a `data` folder (the script should run ok with other data sets if you change the `DATA_PATH` and `INPUT_SHAPE` params). Other training parameters can be changed by modifying the appropriate capitalized variables.

The notebooks contain
- useful utility functions for tensorboard logging (both metrics and images),
- constructing custom Keras models with weight masking and custom training steps (e.g., variational autoencoders),
- custom data loaders that change the model input per epoch for adaptive sampling.

We experiment with a **batch-adaptive sampling method** that increases the batch size over the epochs. This
results in covering more of the training data as we progress in learning.

- [Denoising autoencoders](https://github.com/danieltsoukup/autoencoders/blob/master/denoising_autoencoders.ipynb)

<img src="assets/denoising_vector_field.png" 
alt="denoising vector field"/>

- [Convolutional variational autoencoders for dimension reduction](https://github.com/danieltsoukup/autoencoders/blob/master/variational_autoencoders.ipynb)

<img src="assets/tsne_latent.png" 
alt="tsne clustering"/>

- [Randomized Autoencoder Ensembles for Outlier Detection](https://github.com/danieltsoukup/autoencoders/blob/master/outlier_detection_with_autoencoders.ipynb)

<img src="assets/wordline_features.png" 
alt="wordline features"/>

More ideas to explore:
- Sparse autoencoders
- Semantic hashing for texts
- Layer-wise pretraining of deep autoencoders

Resources (more links in the notebooks):

- [Keras AE tutorial](https://blog.keras.io/building-autoencoders-in-keras.html)
- [Kereas generative VAE example](https://keras.io/examples/generative/vae/)
- [Deeplearning book (by Goodfellow et al) on AEs](https://www.deeplearningbook.org/contents/autoencoders.html)
- [Hands-on ML (by Geron) on AEs](https://github.com/ageron/handson-ml2/blob/master/17_autoencoders_and_gans.ipynb)
- [Awesome anomaly detection resources](https://github.com/yzhao062/anomaly-detection-resources)
- [Wordline post](https://blog.worldline.tech/2018/09/26/anomaly-detection-for-predictive-monitoring.html)
- [PYOD outlier detection library](https://pyod.readthedocs.io/en/latest/index.html)
