from tensorflow import keras
from tensorflow.keras import layers

from detector_base import BaseAnomalyDetector, make_windows  # noqa: F401  (re-exported)


class LSTMAnomalyDetector(BaseAnomalyDetector):
    """Reconstruction-error LSTM autoencoder for one sensor stream.

    Stacked LSTM encoder -> bottleneck -> RepeatVector -> mirrored LSTM
    decoder. Train on normal-only windows, calibrate a threshold on a labeled
    tuning set, then score/predict on new windows.
    """

    ARCH_PARAMS = ("latent_dim", "encoder_units", "decoder_units", "activation",
                   "dropout", "optimizer", "loss")

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
        self.latent_dim = latent_dim
        self.encoder_units = list(encoder_units)
        self.decoder_units = list(decoder_units) if decoder_units is not None else list(reversed(self.encoder_units))
        self.activation = activation
        self.dropout = dropout
        self.optimizer = optimizer
        self.loss = loss
        self.encoder = None
        super().__init__(timesteps, n_features, name=name, feature_names=feature_names)

    def set_architecture(self, **kwargs):
        # keep the decoder mirrored when only the encoder is edited
        if "encoder_units" in kwargs and "decoder_units" not in kwargs:
            kwargs["decoder_units"] = list(reversed(list(kwargs["encoder_units"])))
        super().set_architecture(**kwargs)

    def _build(self):
        inputs = keras.Input(shape=(self.timesteps, self.n_features))
        x = inputs
        for units in self.encoder_units:
            x = layers.LSTM(units, activation=self.activation, return_sequences=True, dropout=self.dropout)(x)
        encoded = layers.LSTM(self.latent_dim, activation=self.activation, return_sequences=False,
                               dropout=self.dropout, name=f"{self.name}_bottleneck")(x)

        x = layers.RepeatVector(self.timesteps)(encoded)
        for units in self.decoder_units:
            x = layers.LSTM(units, activation=self.activation, return_sequences=True, dropout=self.dropout)(x)
        outputs = layers.TimeDistributed(layers.Dense(self.n_features))(x)

        autoencoder = keras.Model(inputs, outputs, name=self.name)
        self.encoder = keras.Model(inputs, encoded, name=f"{self.name}_encoder")
        autoencoder.compile(optimizer=self.optimizer, loss=self.loss)
        return autoencoder
