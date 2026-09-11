# `LSTMAnomalyDetector`

Reconstruction-error LSTM autoencoder for anomaly detection on a single
sensor stream. One instance per stream. Works on plain numpy arrays shaped
`(N_windows, timesteps, total_features)`; labels are `0`/`1` arrays, one per
window.

Also exported: `make_windows(data, window_size)` — turns a `(rows, features)`
array (or 1D label series) into a sliding-window array shaped
`(N_windows, window_size, features)`.

## Constructor

```python
LSTMAnomalyDetector(
    timesteps, n_features,
    latent_dim=16, encoder_units=(128, 64), decoder_units=None,
    activation="tanh", dropout=0.2, optimizer="adam", loss="mse",
    name="lstm_anomaly_detector", feature_names=None,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `timesteps` | `int` | — | Window length. |
| `n_features` | `int` | — | Total input columns. All included by default. |
| `latent_dim` | `int` | `16` | Bottleneck size. |
| `encoder_units` | `Sequence[int]` | `(128, 64)` | Stacked LSTM sizes before the bottleneck. |
| `decoder_units` | `Sequence[int] \| None` | `None` | Stacked LSTM sizes after the bottleneck. `None` → `encoder_units` reversed. |
| `activation` | `str` | `"tanh"` | Activation for all LSTM layers. |
| `dropout` | `float` | `0.2` | Dropout inside each LSTM layer. |
| `optimizer` | `str` | `"adam"` | Passed to `keras.Model.compile`. |
| `loss` | `str` | `"mse"` | Passed to `keras.Model.compile`. |
| `name` | `str` | `"lstm_anomaly_detector"` | Name for the underlying keras models. |
| `feature_names` | `list[str] \| None` | `None` | Column names, length `n_features`. Enables name-based lookups in the feature methods. |

## Properties

| Property | Type | Description |
|---|---|---|
| `n_features` | `int` | Number of currently-*included* features (≤ `total_features`). |
| `active_feature_indices` | `np.ndarray` | Indices of currently-included features, in original column order. |
| `threshold_` | `float \| None` | Current decision threshold. `None` until `tune_threshold` is called. |
| `tuning_report_` | `pd.DataFrame \| None` | Full cutoff sweep from the last `tune_threshold` call. |
| `history_` | `keras.callbacks.History \| None` | Result of the last `fit` call. |

## Architecture

| Method | Description |
|---|---|
| `get_architecture()` | Returns the current config as a `dict` (timesteps, n_features, latent_dim, encoder/decoder units, activation, dropout, optimizer, loss). |
| `set_architecture(latent_dim=None, encoder_units=None, decoder_units=None, activation=None, dropout=None, optimizer=None, loss=None)` | Edits any subset of the config and rebuilds the model. `encoder_units` without `decoder_units` re-mirrors the decoder. |
| `summary()` | Prints the config dict and `keras.Model.summary()`. |

## Feature selection

| Method | Description |
|---|---|
| `list_features()` | Returns a `DataFrame` of every column: `feature_index`, `feature_name`, `included`. |
| `set_features(items)` | Sets the active feature set to exactly `items` (names or indices). Rebuilds the model. |
| `exclude_features(items)` | Deactivates `items` (names or indices) from the current set. Rebuilds the model. |
| `include_features(items)` | Reactivates `items` (names or indices). Rebuilds the model. |
| `feature_breakdown(X, y)` | Returns a `DataFrame` (`feature_index`, `feature_name`, `normal_mse`, `anomalous_mse`, `gap`) ranking currently-active features by how much their reconstruction error separates normal from anomalous windows. |

All methods below take full, un-sliced `total_features`-wide arrays — the
active feature mask is applied internally.

> Rebuilding (via `set_architecture` or any feature-set change) discards
> trained weights, `threshold_`, and `tuning_report_`. `fit`/`tune_threshold`
> must be called again afterward.

## Training / scoring / evaluation

| Method | Description |
|---|---|
| `fit(fit_X, val_X, epochs=200, patience=15, lr_patience=5, lr_factor=0.5, verbose=1)` | Trains the autoencoder to reconstruct `fit_X`, with `EarlyStopping` (restores best weights) and `ReduceLROnPlateau` on `val_X`. Returns the `History` object. |
| `score(X)` | Returns one MSE per window (`mean((X - reconstruction)**2)` over timesteps and features). |
| `tune_threshold(tuning_X, tuning_y, metric="f2", cutoffs=None)` | Sweeps percentile cutoffs over `score(tuning_X)`, scoring each against `tuning_y` on precision/recall/specificity/F1/F2. Sets `threshold_` to the cutoff maximizing `metric`. Returns that best row as a `dict`. `metric` ∈ `{"precision","recall","specificity","f1","f2"}`. |
| `predict(X)` | Returns `int` array of `0`/`1`: `score(X) >= threshold_`. Raises `RuntimeError` if `tune_threshold` hasn't been called. |
| `evaluate(eval_X, eval_y)` | Returns a `dict`: `precision`, `recall`, `specificity`, `f1`, `confusion_matrix`. |

## Usage

```python
from lstm_anomaly_detector import make_windows, LSTMAnomalyDetector

detector = LSTMAnomalyDetector(timesteps=10, n_features=16, name="door_lstm",
                                feature_names=list(features.columns))
detector.fit(fit_X, val_X, epochs=200)                    # fit_X/val_X: normal only
detector.tune_threshold(tuning_X, tuning_y, metric="f2")  # tuning_X/y: normal + anomalous
metrics = detector.evaluate(eval_X, eval_y)                # eval_X/y: held out, touched once
is_anomalous = detector.predict(new_windows)
```

`fit_X`/`val_X` must be normal-only. `tuning_X`/`eval_X` must be disjoint
labeled sets — `tune_threshold` should only ever see the tuning set, and
`eval_X`/`eval_y` should only ever be passed to `evaluate`, so the reported
metrics reflect genuine generalization rather than a threshold fit to the
same data it's scored on.

Multiple sensors = multiple independent instances; nothing is shared between
them.
