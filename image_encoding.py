import numpy as np
import cv2
from scipy import signal as sp_signal
from ssqueezepy import ssq_cwt
from pyts.image import GramianAngularField, MarkovTransitionField, RecurrencePlot

IMAGE_SIZE = (128, 128)
MORLET_MU = 6
FS = 1.0

PMIN_PERIOD = 100.0
PMAX_PERIOD = 700.0

STFT_NPERSEG = 512
STFT_OVERLAP_FRAC = 0.875
STFT_NFFT = 1024
STFT_CROP_PERIOD = True

GAF_METHOD = "summation"
GAF_IMG_LEN = 128
MTF_N_BINS = 4
MTF_STRATEGY = "quantile"
MTF_IMG_LEN = 128
RP_THRESHOLD = None
RP_IMG_LEN = 128

TRANSFORMS = ["cwt", "sst", "stft", "gaf", "mtf", "rp"]


def wavelet_pair(sig):
    Tx, Wx, freqs, _ = ssq_cwt(sig, wavelet=("morlet", {"mu": MORLET_MU}), fs=FS)
    f = np.abs(np.asarray(freqs, dtype=np.float64))
    A_cwt, A_sst = np.abs(Wx), np.abs(Tx)
    if f[0] > f[-1]:
        f = f[::-1].copy()
        A_cwt = A_cwt[::-1].copy()
        A_sst = A_sst[::-1].copy()
    return A_cwt, A_sst, 1.0 / f


def band_rows(period, pmin=PMIN_PERIOD, pmax=PMAX_PERIOD):
    keep = np.where((period >= pmin) & (period <= pmax))[0]
    if keep.size == 0:
        raise ValueError(f"no wavelet rows inside {pmin}-{pmax} d")
    return int(keep[0]), int(keep[-1] + 1)


def encode_one(kind, sig, cache=None):
    H, W = IMAGE_SIZE

    if kind in ("cwt", "sst"):
        A_cwt, A_sst, period = cache if cache is not None else wavelet_pair(sig)
        lo, hi = band_rows(period)
        A = (A_cwt if kind == "cwt" else A_sst)[lo:hi]
        im = cv2.resize(A, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
        return im / (im.max() + 1e-8)

    if kind == "stft":
        nperseg = min(STFT_NPERSEG, len(sig))
        noverlap = int(nperseg * STFT_OVERLAP_FRAC)
        nfft = max(STFT_NFFT, nperseg)
        f, _, Z = sp_signal.stft(sig, fs=1.0, nperseg=nperseg,
                                 noverlap=noverlap, nfft=nfft)
        A = np.abs(Z)
        if STFT_CROP_PERIOD:
            with np.errstate(divide="ignore"):
                per = np.where(f > 0, 1.0 / np.maximum(f, 1e-12), np.inf)
            k = np.where((per >= PMIN_PERIOD) & (per <= PMAX_PERIOD))[0]
            if k.size:
                A = A[k.min():k.max() + 1]
        im = A
    else:
        L = {"gaf": GAF_IMG_LEN, "mtf": MTF_IMG_LEN, "rp": RP_IMG_LEN}[kind]
        xs = np.linspace(0, len(sig) - 1, L)
        s_re = np.interp(xs, np.arange(len(sig)), sig)[None, :]
        if kind == "gaf":
            im = GramianAngularField(image_size=L, method=GAF_METHOD).fit_transform(s_re)[0]
        elif kind == "mtf":
            im = MarkovTransitionField(image_size=L, n_bins=MTF_N_BINS,
                                       strategy=MTF_STRATEGY).fit_transform(s_re)[0]
        elif kind == "rp":
            im = RecurrencePlot(threshold=RP_THRESHOLD).fit_transform(s_re)[0]
        else:
            raise ValueError(kind)

    im = np.asarray(im, np.float32)
    if im.shape != (H, W):
        im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
    mn, mx = float(im.min()), float(im.max())
    return ((im - mn) / (mx - mn + 1e-8)).astype(np.float32)


def encode_batch(kind, series):
    out = np.empty((len(series), *IMAGE_SIZE), dtype=np.float32)
    for i, s in enumerate(series):
        cache = wavelet_pair(s) if kind in ("cwt", "sst") else None
        out[i] = encode_one(kind, s, cache)
    return out[:, None, :, :]


def scale_1d(series):
    out = np.asarray(series, dtype=np.float32).copy()
    mn = out.min(axis=1, keepdims=True)
    mx = out.max(axis=1, keepdims=True)
    return ((out - mn) / (mx - mn + 1e-8))[:, None, :]
