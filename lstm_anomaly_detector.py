import numpy as np
import pandas as pd
import sklearn.metrics
from numpy.lib.stride_tricks import sliding_window_view
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def make_windows(data, window_size: int) -> np.ndarray:
    arr = data.to_numpy() if hasattr(data, "to_numpy") else np.asarray(data)
    windows = sliding_window_view(arr, window_shape=window_size, axis=0)
    return np.moveaxis(windows, -1, 1)


class LSTMAnomalyDetector:
    """Reconstruction-error anomaly detector for one sensor stream.

    Train on normal-only windows, calibrate a threshold on a labeled tuning
    set, then score/predict on new windows. One instance per sensor stream --
    each sensor gets its own baseline, architecture, and threshold.

    Callers always pass windows shaped (N, timesteps, total_features); feature
    inclusion/exclusion (see set_features/exclude_features) is applied
    internally, so the caller never needs to pre-slice arrays.
    """

    METRICS = ("precision", "recall", "specificity", "f1", "f2")

    def __init__(
        self,
        timesteps: int,
        n_features: int,
        latent_dim: int = 16,
        encoder_units=(128, 64),
        decoder_units=None,
        activation: str = "tanh",
        dropout: float = 0.2,
        optimizer: str = "adam",
        loss: str = "mse",
        name: str = "lstm_anomaly_detector",
        feature_names=None,
    ):
        self.timesteps = timesteps
        self.total_features = n_features
        self.feature_names = list(feature_names) if feature_names is not None else [str(i) for i in range(n_features)]
        if len(self.feature_names) != n_features:
            raise ValueError("feature_names must have length n_features")
        self.feature_mask = np.ones(n_features, dtype=bool)

        self.latent_dim = latent_dim
        self.encoder_units = list(encoder_units)
        self.decoder_units = list(decoder_units) if decoder_units is not None else list(reversed(self.encoder_units))
        self.activation = activation
        self.dropout = dropout
        self.optimizer = optimizer
        self.loss = loss
        self.name = name

        self.model = self.encoder = None
        self.threshold_ = None
        self.tuning_report_ = None
        self.history_ = None
        self._rebuild()

    @property
    def n_features(self) -> int:
        """Number of currently-included features (<= total_features)."""
        return int(self.feature_mask.sum())

    @property
    def active_feature_indices(self):
        return np.where(self.feature_mask)[0]

    def _apply_feature_mask(self, X):
        return X[..., self.feature_mask]

    def _rebuild(self):
        """(Re)build the keras model from the current architecture/feature config.

        Called automatically by set_architecture / set_features / exclude_features
        / include_features. Discards any trained weights, threshold, and tuning
        report -- the model needs to be re-fit and re-tuned after this.
        """
        self.model, self.encoder = self._build(
            self.timesteps, self.n_features, self.latent_dim, self.encoder_units,
            self.decoder_units, self.activation, self.dropout, self.optimizer, self.loss, self.name,
        )
        self.threshold_ = None
        self.tuning_report_ = None
        self.history_ = None

    @staticmethod
    def _build(timesteps, n_features, latent_dim, encoder_units, decoder_units,
               activation, dropout, optimizer, loss, name):
        if decoder_units is None:
            decoder_units = list(reversed(encoder_units))

        inputs = keras.Input(shape=(timesteps, n_features))
        x = inputs
        for units in encoder_units:
            x = layers.LSTM(units, activation=activation, return_sequences=True, dropout=dropout)(x)
        encoded = layers.LSTM(latent_dim, activation=activation, return_sequences=False,
                               dropout=dropout, name=f"{name}_bottleneck")(x)

        x = layers.RepeatVector(timesteps)(encoded)
        for units in decoder_units:
            x = layers.LSTM(units, activation=activation, return_sequences=True, dropout=dropout)(x)
        outputs = layers.TimeDistributed(layers.Dense(n_features))(x)

        autoencoder = keras.Model(inputs, outputs, name=name)
        encoder = keras.Model(inputs, encoded, name=f"{name}_encoder")
        autoencoder.compile(optimizer=optimizer, loss=loss)
        return autoencoder, encoder

    # ---- architecture: display / edit --------------------------------

    def get_architecture(self) -> dict:
        return {
            "timesteps": self.timesteps,
            "n_features": self.n_features,
            "latent_dim": self.latent_dim,
            "encoder_units": list(self.encoder_units),
            "decoder_units": list(self.decoder_units),
            "activation": self.activation,
            "dropout": self.dropout,
            "optimizer": self.optimizer,
            "loss": self.loss,
        }

    def set_architecture(self, latent_dim=None, encoder_units=None, decoder_units=None,
                          activation=None, dropout=None, optimizer=None, loss=None):
        """Edit any subset of architecture hyperparameters and rebuild the model.

        Passing encoder_units without decoder_units re-mirrors the decoder
        automatically (same convention as __init__). Rebuilding discards any
        trained weights, threshold, and tuning report -- fit/tune again after.
        """
        if encoder_units is not None:
            self.encoder_units = list(encoder_units)
            self.decoder_units = list(decoder_units) if decoder_units is not None else list(reversed(self.encoder_units))
        elif decoder_units is not None:
            self.decoder_units = list(decoder_units)
        if latent_dim is not None:
            self.latent_dim = latent_dim
        if activation is not None:
            self.activation = activation
        if dropout is not None:
            self.dropout = dropout
        if optimizer is not None:
            self.optimizer = optimizer
        if loss is not None:
            self.loss = loss
        self._rebuild()

    def summary(self):
        print(f"{self.name}: {self.get_architecture()}")
        self.model.summary()

    # ---- features: display / edit -------------------------------------

    def list_features(self) -> pd.DataFrame:
        return pd.DataFrame({
            "feature_index": range(self.total_features),
            "feature_name": self.feature_names,
            "included": self.feature_mask,
        })

    def _resolve_indices(self, items):
        return [item if isinstance(item, (int, np.integer)) else self.feature_names.index(item) for item in items]

    def set_features(self, included):
        """Keep only the given features (indices or names) active. Rebuilds the model."""
        mask = np.zeros(self.total_features, dtype=bool)
        mask[self._resolve_indices(included)] = True
        if not mask.any():
            raise ValueError("must include at least one feature")
        self.feature_mask = mask
        self._rebuild()

    def exclude_features(self, excluded):
        """Drop the given features (indices or names) from an otherwise-included set. Rebuilds the model."""
        mask = self.feature_mask.copy()
        mask[self._resolve_indices(excluded)] = False
        if not mask.any():
            raise ValueError("cannot exclude every feature")
        self.feature_mask = mask
        self._rebuild()

    def include_features(self, included):
        """Add the given features (indices or names) back into an otherwise-excluded set. Rebuilds the model."""
        mask = self.feature_mask.copy()
        mask[self._resolve_indices(included)] = True
        self.feature_mask = mask
        self._rebuild()

    # ---- training / scoring / evaluation -------------------------------

    def fit(self, fit_X, val_X, epochs=200, patience=15, lr_patience=5, lr_factor=0.5, verbose=1):
        fit_X = self._apply_feature_mask(fit_X)
        val_X = self._apply_feature_mask(val_X)
        self.history_ = self.model.fit(
            fit_X, fit_X,
            validation_data=(val_X, val_X),
            epochs=epochs,
            verbose=verbose,
            callbacks=[
                EarlyStopping(patience=patience, restore_best_weights=True),
                ReduceLROnPlateau(patience=lr_patience, factor=lr_factor),
            ],
        )
        return self.history_

    def score(self, X, verbose=0):
        X = self._apply_feature_mask(X)
        pred = self.model.predict(X, verbose=verbose)
        return np.mean((X - pred) ** 2, axis=(1, 2))

    def tune_threshold(self, tuning_X, tuning_y, metric="f2", cutoffs=None, verbose=0):
        if metric not in self.METRICS:
            raise ValueError(f"metric must be one of {self.METRICS}, got {metric!r}")

        mses = self.score(tuning_X, verbose=verbose)
        if cutoffs is None:
            cutoffs = [c / 2 for c in range(0, 200)]

        rows = []
        for cutoff in cutoffs:
            threshold = np.percentile(mses, cutoff)
            y_pred = (mses >= threshold).astype(int)
            rows.append({
                "cutoff": cutoff,
                "threshold": threshold,
                "precision": sklearn.metrics.precision_score(tuning_y, y_pred, zero_division=0),
                "recall": sklearn.metrics.recall_score(tuning_y, y_pred, zero_division=0),
                "specificity": sklearn.metrics.recall_score(tuning_y, y_pred, pos_label=0, zero_division=0),
                "f1": sklearn.metrics.f1_score(tuning_y, y_pred, zero_division=0),
                "f2": sklearn.metrics.fbeta_score(tuning_y, y_pred, beta=2, zero_division=0),
            })

        self.tuning_report_ = pd.DataFrame(rows)
        best_row = self.tuning_report_.loc[self.tuning_report_[metric].idxmax()]
        self.threshold_ = best_row["threshold"]
        return best_row.to_dict()

    def predict(self, X, verbose=0):
        if self.threshold_ is None:
            raise RuntimeError("call tune_threshold(...) before predict(...)")
        return (self.score(X, verbose=verbose) >= self.threshold_).astype(int)

    def evaluate(self, eval_X, eval_y, verbose=0):
        y_pred = self.predict(eval_X, verbose=verbose)
        return {
            "precision": sklearn.metrics.precision_score(eval_y, y_pred, zero_division=0),
            "recall": sklearn.metrics.recall_score(eval_y, y_pred, zero_division=0),
            "specificity": sklearn.metrics.recall_score(eval_y, y_pred, pos_label=0, zero_division=0),
            "f1": sklearn.metrics.f1_score(eval_y, y_pred, zero_division=0),
            "confusion_matrix": sklearn.metrics.confusion_matrix(eval_y, y_pred),
        }

    def feature_breakdown(self, X, y, verbose=0):
        """Per-(currently active) feature reconstruction-error gap between
        anomalous and normal windows. feature_index/feature_name refer back to
        the original column layout, even when some features are excluded."""
        X_active = self._apply_feature_mask(X)
        pred = self.model.predict(X_active, verbose=verbose)
        per_window_feature_mse = np.mean((X_active - pred) ** 2, axis=1)
        y = np.asarray(y)
        normal_mse = per_window_feature_mse[y == 0].mean(axis=0)
        anomalous_mse = per_window_feature_mse[y == 1].mean(axis=0)
        active_indices = self.active_feature_indices
        return pd.DataFrame({
            "feature_index": active_indices,
            "feature_name": [self.feature_names[i] for i in active_indices],
            "normal_mse": normal_mse,
            "anomalous_mse": anomalous_mse,
            "gap": anomalous_mse - normal_mse,
        }).sort_values("gap", ascending=False)
