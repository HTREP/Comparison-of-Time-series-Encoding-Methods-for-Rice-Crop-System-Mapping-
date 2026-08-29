# Comparison of Time-series Encoding Methods for Rice Crop System Mapping

Reference implementation for the experiment reported in *Comparison of
Time-series Encoding Methods for Rice Crop System Mapping from Sentinel-2 EVI
Time Series Using Convolutional Neural Networks*.

Six image encodings and an unencoded one-dimensional baseline are trained on
the same reconstructed Enhanced Vegetation Index (EVI) profiles and evaluated
against the same independent reference set, so that the representation is the
only quantity that differs between arms.

| Arm | Description |
|-----|-------------|
| `cwt` | Continuous Wavelet Transform scalogram, Morlet wavelet |
| `sst` | Synchrosqueezing Transform of the same wavelet coefficients |
| `stft` | Short-Time Fourier Transform spectrogram |
| `gaf` | Gramian Angular Summation Field |
| `mtf` | Markov Transition Field |
| `rp` | Recurrence Plot, unthresholded |
| `1d` | Reconstructed profile passed directly to a one-dimensional CNN |

The Sentinel-2 imagery, the field reference points, and the rice cultivation
mask used in the paper are not redistributed here. The repository ships a
generator that produces synthetic profiles in the required format so that the
pipeline can be run end to end without them.

---

## Requirements

### Software

| Package | Version used |
|---------|--------------|
| Python | 3.10 or later |
| PyTorch | 2.0 or later |
| NumPy | 1.24 or later |
| pandas | 2.0 or later |
| SciPy | 1.11 or later |
| scikit-learn | 1.3 or later |
| ssqueezepy | 0.6.4 or later |
| pyts | 0.13 or later |
| opencv-python | 4.8 or later |

```
pip install torch numpy pandas scipy scikit-learn ssqueezepy pyts opencv-python
```

`ssqueezepy` replaces `scipy.signal.cwt` and `scipy.signal.morlet2`, which were
removed in SciPy 1.15.

### Hardware

The figures below describe one arm trained on 200 profiles of 726 daily steps
under five-fold cross-validation.

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| CPU | 4 cores | 8 cores or more |
| System memory | 8 GB | 16 GB |
| GPU | not required | CUDA device with 6 GB of memory or more |
| Disk | 2 GB | 10 GB if prediction rasters are written |

A CUDA device is optional for training at this sample size but becomes
necessary when the trained model is applied to a full scene, where the encoding
is computed for every pixel of the rice mask. The pairwise encodings (`gaf`,
`mtf`, `rp`) produce a matrix whose side equals the length of their input, so
memory use grows with the square of the record length; the series is reduced to
128 points before encoding for this reason.

---

## Input data format

Two CSV files are required, one for training and one for evaluation. Both use
the same layout: one row per reference point, with the reconstructed daily EVI
profile stored across the `t*` columns.

| Column | Type | Description |
|--------|------|-------------|
| `point_id` | string | Identifier of the reference point |
| `class` | string | One of `SRC`, `DRC`, `HRC`, `TRC` |
| `x`, `y` | float | Projected coordinates; retained for reference, not used in training |
| `t0` … `tN` | float | Reconstructed daily EVI, one column per day |

```
point_id,class,x,y,t0,t1,t2,...,t725
SRC_000,SRC,583555.0,1650265.0,0.181,0.183,0.186,...,0.204
DRC_000,DRC,612340.0,1621880.0,0.094,0.098,0.101,...,0.315
```

Requirements on the profile columns:

- Every row must have the same number of `t*` columns, and the columns must be
  ordered by day index. The paper uses 726 columns, corresponding to the window
  from 1 April 2023 to 26 March 2025.
- Values must already be reconstructed. No missing values are permitted.
- Values are expected in the range 0 to 1.

The four classes are single rice crop (SRC, one cycle per year), double rice
crop (DRC, two cycles per year), two-and-a-half rice crop (HRC, five cycles in
two years), and triple rice crop (TRC, three cycles per year).

### Producing the profile columns from raw observations

`preprocess.py` implements the reconstruction used in the paper. The order of
operations is fixed: invalid values are removed, the remaining observations are
interpolated onto a uniform daily grid, and the interpolated sequence is then
smoothed with a Savitzky–Golay filter using a 31-day window and a third-order
polynomial. Applying the filter before interpolation would define the window
over an irregular sampling interval and distort the reconstructed profile.

```python
from preprocess import preprocess_daily
import numpy as np

obs_days = np.array([0.0, 5.0, 15.0, 20.0])
values   = np.array([0.21, 0.24, 0.31, 0.35])
target   = np.arange(726, dtype=float)

daily = preprocess_daily(values, obs_days, target)
```

---

## Usage

Generate synthetic data in the required format:

```
python make_mock_data.py --n-per-class 50
```

This writes `data/train.csv` and `data/eval.csv`, each containing 200 profiles
of 726 daily steps, balanced across the four classes. The generator applies the
same reconstruction as the real pipeline, including a simulated cloud gap rate
of 25 percent on a five-day revisit schedule.

Train all seven arms:

```
python train.py --train data/train.csv --eval data/eval.csv --outdir results
```

Train a subset:

```
python train.py --arms cwt gaf 1d --outdir results
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--train` | `data/train.csv` | Training split |
| `--eval` | `data/eval.csv` | Independent evaluation split |
| `--outdir` | `results` | Output directory |
| `--arms` | all seven | Subset of arms to run |
| `--seed` | `0` | Seed for weight initialisation and shuffling |
| `--device` | `cuda` if available | `cuda` or `cpu` |

---

## Output

| File | Content |
|------|---------|
| `summary.csv` | One row per arm: cross-validation macro F1, evaluation overall accuracy, macro F1, kappa, parameter count |
| `confusion_<arm>.csv` | Confusion matrix on the evaluation set |
| `report_<arm>.csv` | Per-class precision, recall, and F1-score |
| `cv_<arm>.csv` | Per-fold macro F1 and the epoch at which each fold stopped |
| `pred_<arm>.npy` | Predicted class index for each evaluation point |
| `model_<arm>.pth` | Weights of the final model |
| `config.json` | Training configuration for the run |

---

## Configuration

Encoding parameters are defined at the top of `image_encoding.py`.

| Encoding | Parameter | Value |
|----------|-----------|-------|
| CWT | Mother wavelet | Morlet, ω₀ = 6 |
| CWT | Period band | 100 to 700 days |
| SST | Base transform | CWT with the same wavelet and band |
| STFT | Window | Hann, 512 days |
| STFT | Hop | 64 days, 87.5 percent overlap |
| STFT | FFT length | 1024 |
| GAF | Variant | Summation |
| MTF | Quantile bins | 4 |
| RP | Threshold | None, continuous distance matrix |
| RP | Embedding | m = 1, τ = 1 |
| All | Output image | 128 × 128, single channel, scaled to [0, 1] |

Training parameters are defined at the top of `train.py`.

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | 1 × 10⁻⁴ |
| Weight decay | 1 × 10⁻² |
| Batch size | 128 |
| Maximum epochs | 500 |
| Early stopping | Validation macro F1, patience 100 |
| Cross-validation | Stratified five-fold, `random_state = 42` |
| Loss | Cross-entropy |
| Augmentation, learning-rate schedule, class weighting | None |

The two-dimensional network has 422,212 trainable parameters and the
one-dimensional network 188,676. Both use four convolution blocks with 32, 64,
128, and 256 filters, batch normalisation, ReLU activation, max pooling after
the first three blocks, adaptive average pooling, and a fully connected head of
256 → 128 → 4 with dropout of 0.3. The two-dimensional network uses 3 × 3
kernels throughout; the one-dimensional network uses kernels of 7, 7, 5, and 3.

---

## Repository contents

| File | Purpose |
|------|---------|
| `preprocess.py` | Reconstruction of irregular observations into daily profiles |
| `image_encoding.py` | The six image encodings and the scaling applied to the 1D input |
| `models.py` | Two-dimensional and one-dimensional CNN definitions |
| `train.py` | Cross-validation, final training, and evaluation |
| `make_mock_data.py` | Synthetic data generator in the required input format |

---

## Notes on reproduction

Results in the paper were obtained from field reference points collected in
Suphan Buri province with the Thailand Rice Science Institute. Running the
pipeline on the synthetic data supplied here will not reproduce those figures;
the generator is intended only to demonstrate the input format and to confirm
that the pipeline executes.

Each encoding was evaluated at a single parameter setting. The values above
describe the configuration reported in the paper and are not the outcome of a
search over the parameter space of each encoding.
