import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle


#######################################################################
# Based on  Jinghui Chen, Saket Sathe, Charu Aggarwal, Deepak Turaga. #
# Outlier Detection with Autoencoder Ensembles. SIAM SDM, 2017.       #
#######################################################################

class RandAE(tf.keras.Sequential):
    def __init__(self, input_dim, hidden_dims, drop_ratio=0.5, **kwargs):
        super(RandAE, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.drop_ratio = drop_ratio

        self.layer_masks = dict()

        self.build_model()

    def build_model(self) -> None:
        """
        Adds the layers and records masks.
        """

        self.add(layers.Input(self.input_dim, name="input"))

        for i, dim in enumerate(self.hidden_dims):
            layer_name = f"hidden_{i}"
            layer = layers.Dense(
                dim, activation="relu" if i > 0 else "sigmoid", name=layer_name
            )
            self.add(layer)

            # add layer mask
            self.layer_masks[layer_name] = self.get_mask(layer)

        layer_name = "output"
        output_layer = layers.Dense(
            self.input_dim, activation="sigmoid", name=layer_name
        )
        self.add(output_layer)
        self.layer_masks[layer_name] = self.get_mask(output_layer)

    def get_mask(self, layer, mode="replacement") -> np.ndarray:
        """
        Build mask for a layer.
        """

        shape = layer.input_shape[1], layer.output_shape[1]

        if mode == "by_ratio":
            mask = np.random.choice(
                [0.0, 1.0], size=shape, p=[self.drop_ratio, 1 - self.drop_ratio]
            )
        elif mode == "replacement":
            mask = np.ones(shape=(shape[0] * shape[1],))
            zero_idx = np.random.choice(mask.shape[0], size=mask.shape[0], replace=True)
            zero_idx = np.unique(zero_idx)
            mask[zero_idx] = 0
            mask = mask.reshape(shape)
        else:
            raise NotImplementedError(
                f"Mode {mode} not implemented, choose from 'replacement' (original implementation) or 'by_ratio'."
            )

        return mask

    def load_masks(self, mask_pickle_path) -> None:
        """
        Load the masks from a pickled dictionary.
        """

        with open(mask_pickle_path, "rb") as handle:
            self.layer_masks = pickle.load(handle)

    def get_encoder(self) -> keras.Sequential:
        """
        Get the encoder from the full model.
        """

        n_layers = (len(self.hidden_dims) + 1) // 2
        encoder_layers = [layers.Input(self.input_dim)] + self.layers[:n_layers]

        return keras.Sequential(encoder_layers)

    def mask_weights(self) -> None:
        """
        Apply the masks to each layer in the encoder and decoder.
        """

        for layer in self.layers:
            layer_name = layer.name
            if layer_name in self.layer_masks:
                masked_w = layer.weights[0].numpy() * self.layer_masks[layer_name]
                b = layer.weights[1].numpy()
                layer.set_weights((masked_w, b))

    def call(self, data, training=True) -> tf.Tensor:
        # mask the weights before original forward pass
        self.mask_weights()

        return super().call(data)


class BatchAdaptiveDataGenerator(keras.utils.Sequence):
    def __init__(
        self,
        x,
        start_batch_size,
        epochs,
        subsample=0.3,
        start_data_ratio=0.5,
        shuffle=True,
        verbose=False,
    ):
        self.x = x
        self.subsample = subsample
        self.verbose = verbose

        if self.subsample:
            sample_idx = np.random.choice(
                self.x.shape[0], size=int(self.subsample * self.x.shape[0])
            )
            self.x = self.x[sample_idx]

        # initial training params
        self.epochs = epochs
        self.start_batch_size = start_batch_size
        self.start_data_ratio = start_data_ratio
        self.steps_per_epoch = int(
            self.start_data_ratio * self.x.shape[0] / self.start_batch_size
        )

        # adaptive learning param to increase batch_size after each epoch
        self.alpha = np.exp(np.log(1 / self.start_data_ratio) / self.epochs)
        self.shuffle = shuffle

        # per epoch variables
        self.epoch = 0
        self.current_x = None
        self.current_batch_size = self.start_batch_size
        self.on_epoch_end()

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx):
        """
        Return a batch for autoencoder training.
        """

        batch = self.current_x[
            idx * self.current_batch_size : (idx + 1) * self.current_batch_size
        ]

        return batch, batch

    def on_epoch_end(self):
        """
        Called before training and after every epoch to include more and more training data.
        """

        # update training data by slicing and shuffling
        current_x_size = int(self.current_batch_size * self.steps_per_epoch)
        self.current_x = self.x[:current_x_size]

        # shuffle rows to mix data in different batches
        if self.shuffle:
            rand_idx = np.arange(self.current_x.shape[0])
            np.random.shuffle(rand_idx)
            self.current_x = self.current_x[rand_idx]

        if self.verbose:
            print(
                f"Epoch {self.epoch + 1} -- {self.current_x.shape[0] / self.x.shape[0] * 100}% data"
            )

        # update batch size
        self.current_batch_size = int(self.start_batch_size * self.alpha ** self.epoch)
        self.epoch += 1
