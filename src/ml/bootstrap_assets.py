"""Create minimal ML artifacts required by the Streamlit app.

The Streamlit app expects these files under `src/ml/`:
- ml/models/model.pkl
- ml/models/vectorizer.pkl
- ml/data/X_test.pkl
- ml/data/y_test.pkl

If you haven't trained/exported your real model yet, this module can generate a
small TF‑IDF + LogisticRegression model on a tiny synthetic dataset so the app
runs end-to-end.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass

import joblib


@dataclass(frozen=True)
class MlAssetPaths:
    model_path: str
    vectorizer_path: str
    x_test_path: str
    y_test_path: str


def get_asset_paths(src_dir: str) -> MlAssetPaths:
    ml_dir = os.path.join(src_dir, "ml")
    models_dir = os.path.join(ml_dir, "models")
    data_dir = os.path.join(ml_dir, "data")

    return MlAssetPaths(
        model_path=os.path.join(models_dir, "model.pkl"),
        vectorizer_path=os.path.join(models_dir, "vectorizer.pkl"),
        x_test_path=os.path.join(data_dir, "X_test.pkl"),
        y_test_path=os.path.join(data_dir, "y_test.pkl"),
    )


def _ensure_dirs(paths: MlAssetPaths) -> None:
    os.makedirs(os.path.dirname(paths.model_path), exist_ok=True)
    os.makedirs(os.path.dirname(paths.x_test_path), exist_ok=True)


def bootstrap_ml_assets(src_dir: str, *, force: bool = False) -> MlAssetPaths:
    """Create artifacts expected by the Streamlit app.

    Args:
        src_dir: Absolute path to the repo's `src/` directory.
        force: If True, overwrite any existing artifacts.

    Returns:
        Paths for the created (or existing) artifacts.
    """

    paths = get_asset_paths(src_dir)
    _ensure_dirs(paths)

    if not force and all(
        os.path.exists(p)
        for p in (paths.model_path, paths.vectorizer_path, paths.x_test_path, paths.y_test_path)
    ):
        return paths

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    real_examples = [
        "The official report confirms the event occurred on Tuesday.",
        "According to the government statistics office, inflation fell 0.5%.",
        "The court filing shows the company paid a settlement in 2022.",
        "The agency published the full dataset and methodology online.",
        "Reuters reported the announcement after the press conference.",
        "The peer-reviewed study was published in a scientific journal.",
    ]

    fake_examples = [
        "BREAKING: Secret cure they don't want you to know!!!",
        "SHOCKING: You won't believe what happened next!!!",
        "Doctors hate him for revealing this one weird trick.",
        "This is 100% proven and guaranteed to work overnight.",
        "Miracle product exposes the truth mainstream media hides.",
        "Click here to see the unbelievable truth they erased.",
    ]

    texts = real_examples + fake_examples
    labels = [1] * len(real_examples) + [0] * len(fake_examples)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=5000,
    )

    X = vectorizer.fit_transform(texts)

    model = LogisticRegression(
        max_iter=1000,
        solver="liblinear",
        random_state=42,
    )
    model.fit(X, labels)

    joblib.dump(model, paths.model_path)
    joblib.dump(vectorizer, paths.vectorizer_path)

    with open(paths.x_test_path, "wb") as f:
        pickle.dump(X, f)
    with open(paths.y_test_path, "wb") as f:
        pickle.dump(labels, f)

    return paths
