import numpy as np
import pandas as pd
from typing import Tuple
from tqdm import tqdm
from datetime import datetime
import pickle
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    cohen_kappa_score,
)
from randnet import RandAE, BatchAdaptiveDataGenerator


def load_data(path: str) -> pd.DataFrame:
    """
    Load the data into dataframe.
    """

    data = pd.read_csv(path)

    return data


def preprocess_data(data, target_col, test_size=0.2) -> Tuple:
    """
    Scale and train-test split the data.
    """

    scaler = Pipeline(
        [
            ("quantile_transform", QuantileTransformer(output_distribution="normal")),
            ("minmax", MinMaxScaler()),
        ]
    )

    X = data.drop(target_col, axis=1).values
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

    # make ensemble predictions
    predictions = [model.predict(input_data) for model in ensemble]

    reconstruction_loss = np.stack(
        [np.square((pred - input_data)).mean(axis=1) for pred in predictions], axis=1
    )
    median_loss = np.median(reconstruction_loss, axis=1)

    threshold = np.quantile(median_loss, contamination)
    test_outliers = np.where(median_loss > threshold, 1, 0)

    results["classification_report"] = classification_report(
        input_labels, test_outliers
    )

    # min-max scaling the reconstruction loss
    min_ = median_loss.min()
    max_ = median_loss.max()
    scaled_loss = (median_loss - min_) / (max_ - min_)

    results["precision_recall_curve"] = precision_recall_curve(
        input_labels, scaled_loss
    )

    results["cohen_kappa"] = cohen_kappa_score(input_labels, test_outliers)

    return results


DATA_PATH = "data/creditcard.csv"
TARGET_COL = "Class"
INPUT_SHAPE = 30

MODEL_PARAMS = {"input_dim": INPUT_SHAPE, "hidden_dims": [16, 8, 16], "drop_ratio": 0.2}

COMPILE_PARAMS = {
    "optimizer": "adam",
    "loss": "mse",
    "run_eagerly": True,
}

EPOCHS = 25

DATA_GEN_PARAMS = {
    "start_batch_size": 128,
    "epochs": EPOCHS,
    "subsample": 0.3,
}

FIT_PARAMS = {"epochs": EPOCHS, "verbose": 1}

N_MODELS = 50

all_params = (
    MODEL_PARAMS,
    COMPILE_PARAMS,
    EPOCHS,
    DATA_GEN_PARAMS,
    FIT_PARAMS,
    N_MODELS,
)

if __name__ == "__main__":
    # load the data
    print("Loading the data...")
    data = load_data(DATA_PATH)

    # scale and train-test split
    print("Processing the data...")
    X_train, X_test, y_train, y_test = preprocess_data(data, TARGET_COL)

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
        model.fit(data_generator, **FIT_PARAMS)

        # save the weights and mask
        model.save_weights(f"models/randae_model_{i}_{timestamp}")

        with open(f"models/model_{i}_masks_{timestamp}.pickle", "wb") as handle:
            pickle.dump(model.layer_masks, handle, protocol=pickle.HIGHEST_PROTOCOL)

        ensemble.append(model)

    # evaluate the ensemble on the test set
    print("Evaluating...")
    eval_results = eval_ensemble(ensemble, X_test, y_test, contamination=y_train.mean())

    with open(f"models/eval_results_{timestamp}.pickle", "wb") as handle:
        pickle.dump(eval_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
