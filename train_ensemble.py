import numpy as np
import pandas as pd
from typing import Tuple
from tqdm import tqdm
from datetime import datetime
import pickle
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler, StandardScaler
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    cohen_kappa_score,
)
from tensorflow import keras
from randnet import RandAE, BatchAdaptiveDataGenerator


class MyRobustScaler(TransformerMixin):
    def __init__(self, low_q=0.1, high_q=0.9):
        self.low_q = low_q
        self.high_q = high_q

        self.feature_low_bound = None
        self.feature_high_bound = None

    def fit(self, X):
        """
        Record per-feature quantiles.
        """
        self.feature_low_bound, self.feature_high_bound = np.quantile(X, [self.low_q, self.high_q], axis=0)

        return self

    def transform(self, X):
        """
        Limit the data between the quantiles.
        """
        X_transformed = np.where(X <= self.feature_low_bound, self.feature_low_bound, X)
        X_transformed = np.where(X_transformed >= self.feature_high_bound, self.feature_high_bound, X_transformed)

        return X_transformed


def load_data(path: str) -> pd.DataFrame:
    """
    Load the data into dataframe.
    """

    data = pd.read_csv(path)

    return data


def preprocess_data(data, target_col, drop_cols=None, test_size=0.2) -> Tuple:
    """
    Scale and train-test split the data.
    """

    scaler = Pipeline(
        [
          #  ("own_robust", MyRobustScaler()),
            ("minmax", MinMaxScaler()),
        ]
    )

    if drop_cols:
        drop_cols += [target_col]
    else:
        drop_cols = [target_col]

    X = data.drop(drop_cols, axis=1).values
    y = data[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=1
    )

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    return X_train, X_test, y_train, y_test


def eval_ensemble(ensemble, input_data, input_labels, contamination):

    results = dict()

    # make prediction by each ensemble component
    predictions = [model.predict(input_data) for model in ensemble]

    # SSE based on reconstruction for each component
    reconstruction_loss = np.stack(
        [np.square(pred - input_data).sum(axis=1) for pred in predictions], axis=1
    ) # could be better ways to aggregate other then sum...

    results["reconstruction_loss"] = reconstruction_loss

    # scale the std to account for different levels of overfitting
    scaler = StandardScaler(with_mean=False)
    reconstruction_loss = scaler.fit_transform(reconstruction_loss)

    # find the median loss for each sample
    median_loss = np.median(reconstruction_loss, axis=1)

    # calibrate the threshold by training contamination ratio
    threshold = np.quantile(median_loss, contamination)

    # make hard predicition
    test_outliers = np.where(median_loss > threshold, 1, 0)

    results["classification_report"] = classification_report(
        input_labels, test_outliers
    )

    # min-max scaling the reconstruction loss to calculate PR-curve
    min_ = median_loss.min()
    max_ = median_loss.max()
    scaled_loss = (median_loss - min_) / (max_ - min_)

    results["precision_recall_curve"] = precision_recall_curve(
        input_labels, scaled_loss
    )

    results["cohen_kappa"] = cohen_kappa_score(input_labels, test_outliers)

    return results


# CHANGE if other data used
DATA_PATH = "data/creditcard.csv"
DROP_COLS = [] #["Time", "Amount"]
TARGET_COL = "Class"
INPUT_SHAPE = 30 - len(DROP_COLS)

MODEL_PARAMS = {
    "input_dim": INPUT_SHAPE,
    "hidden_dims": [25, 12, 25],  # could benefit from layer-wise pretraining
    "drop_ratio": 0.2,
}

COMPILE_PARAMS = {
    "optimizer": "adam",
    "loss": "mse",  # can try binary cross entropy?
    "run_eagerly": True,  # for the layer-masking to work
}

EPOCHS = 50  # needs to be increased if layer-wise pre-training

DATA_GEN_PARAMS = {
    "start_batch_size": 32,
    "epochs": EPOCHS,
    "subsample": 0.1,
}

FIT_PARAMS = {"epochs": EPOCHS, "verbose": 1, "workers": -1}

N_MODELS = 25  # there was no significant gain from more components in the paper

all_params = {
    "model_params": MODEL_PARAMS,
    "compile_params": COMPILE_PARAMS,
    "epochs": EPOCHS,
    "data_gen_params": DATA_GEN_PARAMS,
    "fit_params": FIT_PARAMS,
    "n_models": N_MODELS,
}

if __name__ == "__main__":
    # load the data
    print("Loading the data...")
    data = load_data(DATA_PATH)

    # scale and train-test split
    print("Processing the data...")
    X_train, X_test, y_train, y_test = preprocess_data(data, TARGET_COL, drop_cols=DROP_COLS)

    # fit models and save results
    print("Fitting the models...")
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    with open(f"models/all_params_{timestamp}.pickle", "wb") as handle:
        pickle.dump(all_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ensemble = []
    for i in tqdm(range(N_MODELS)):
        model = RandAE(**MODEL_PARAMS)
        model.compile(**COMPILE_PARAMS)

        data_generator = BatchAdaptiveDataGenerator(X_train, **DATA_GEN_PARAMS)
        history = model.fit(data_generator, **FIT_PARAMS)

        # save the weights and mask
        model.save_weights(f"models/model_{i}_{timestamp}")

        with open(f"models/model_{i}_masks_{timestamp}.pickle", "wb") as handle:
            pickle.dump(model.layer_masks, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f"models/model_{i}_history_{timestamp}.pickle", "wb") as handle:
            pickle.dump(history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

        ensemble.append(model)

    # evaluate the ensemble on the test set
    print("Evaluating...")
    eval_results = eval_ensemble(ensemble, X_test, y_test, contamination=y_train.mean())

    with open(f"models/eval_results_{timestamp}.pickle", "wb") as handle:
        pickle.dump(eval_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
