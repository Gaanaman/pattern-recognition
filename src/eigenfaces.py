"""Support code for the Eigenface recognition study.

Holds the pieces that would otherwise clutter the notebook: data splitting,
image perturbation, evaluation and plotting helpers. The PCA itself is
implemented in the notebook, since it is the object of study.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

IMG_SHAPE = (64, 64)


# --------------------------------------------------------------------------
# Data splitting
# --------------------------------------------------------------------------
def subject_split(y, n_train, rng):
    """Split indices so each subject contributes exactly `n_train` images.

    Every identity must appear in the training set: a face recogniser cannot
    be asked to name a person it has never seen. A plain random split does
    not guarantee that, so the split is built per subject.
    """
    train, test = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0].copy()
        rng.shuffle(idx)
        train.extend(idx[:n_train])
        test.extend(idx[n_train:])
    return np.array(sorted(train)), np.array(sorted(test))


# --------------------------------------------------------------------------
# Image perturbations (used in the critical-analysis section)
# --------------------------------------------------------------------------
def illumination_ramp(X, strength, shape=IMG_SHAPE):
    """Multiply each image by a left-to-right linear intensity ramp.

    Models a directional light source. `strength` 0 leaves the image
    untouched; 1.0 gives a ramp from 0 to 2x across the width.
    """
    h, w = shape
    ramp = np.linspace(1.0 - strength, 1.0 + strength, w)[None, :]
    ramp = np.repeat(ramp, h, axis=0).ravel()
    return np.clip(X * ramp[None, :], 0.0, 1.0)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def score_all(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": p,
        "recall_macro": r,
        "f1_macro": f,
    }


def reconstruction_metrics(X, X_hat):
    """RMSE and PSNR between originals and reconstructions (pixels in [0, 1])."""
    mse = np.mean((X - X_hat) ** 2, axis=1)
    rmse = np.sqrt(mse)
    psnr = 10 * np.log10(1.0 / np.maximum(mse, 1e-12))
    return rmse.mean(), psnr.mean()


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def image_grid(images, titles=None, ncols=5, figsize=None, cmap="gray",
               suptitle=None, shape=IMG_SHAPE, normalise=False, path=None):
    n = len(images)
    nrows = int(np.ceil(n / ncols))
    figsize = figsize or (1.7 * ncols, 2.1 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= n:
            continue
        img = images[i].reshape(shape)
        kw = {} if normalise else {"vmin": 0.0, "vmax": 1.0}
        ax.imshow(img, cmap=cmap, **kw)
        if titles is not None:
            ax.set_title(titles[i], fontsize=9)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, y=1.0)
    fig.tight_layout(h_pad=1.6)
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path
