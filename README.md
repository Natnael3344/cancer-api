# Cancer API (breast-cancer-api)

A small Flask REST API that serves a pre-trained Support Vector Machine (SVM)
model to classify a tumor as **Malignant** or **Benign** from a set of 30
numeric input features.

The repository does not include the training script or dataset, so the exact
provenance of the data isn't documented in code. What can be verified directly
from the shipped model artifacts:

- The `StandardScaler` (`scaler.pkl`) was fit on **30 input features**.
- The `PCA` transformer (`pca.pkl`) reduces those 30 features to **10
  principal components**.
- The classifier (`svm_model.pkl`) is a scikit-learn `SVC` with an **RBF
  kernel** trained on the 10 PCA components, predicting one of two classes
  (`0` or `1`).

A 30-feature input with a binary malignant/benign outcome is consistent with
the well-known Breast Cancer Wisconsin (Diagnostic) dataset (features such as
radius, texture, perimeter, area, smoothness, compactness, concavity,
concave points, symmetry, and fractal dimension, each reported as a mean,
standard error, and "worst" value). This is an inference based on the shape
of the data, not something asserted in the code or documented anywhere in the
repository — no feature names or training data are included.

## Tech stack

- **Python 3**
- **Flask** — HTTP API / routing
- **scikit-learn** — `StandardScaler`, `PCA`, and `SVC` (loaded from
  pre-trained `.pkl` files; no training code is included in this repo)
- **NumPy** — reshaping incoming request data into a feature array
- **joblib** — loading the serialized model/scaler/PCA objects
- **Gunicorn** — production WSGI server (used by the Render deployment)
- **Render** (`render.yaml`) — deployment configuration for Render.com, as a
  web service named `breast-cancer-api`

## Project structure

```
cancer-api/
├── app.py            # Flask app; defines the single POST /predict endpoint
├── svm_model.pkl     # Pre-trained scikit-learn SVC (RBF kernel) classifier
├── scaler.pkl        # Pre-fit StandardScaler (expects 30 input features)
├── pca.pkl           # Pre-fit PCA (reduces 30 features -> 10 components)
├── requirements.txt  # Python dependencies
└── render.yaml       # Render.com deployment configuration
```

There is no training script, dataset, or test suite in this repository — only
the inference API and the already-trained model artifacts.

## Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/Natnael3344/cancer-api.git
cd cancer-api
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` lists: `flask`, `numpy`, `scikit-learn`, `joblib`,
`gunicorn` (no versions are pinned).

> **Note:** the `.pkl` files in this repo were serialized with
> scikit-learn 1.6.1. If you install a newer/older scikit-learn version,
> loading them will print an `InconsistentVersionWarning` at startup. In
> practice the models still load and predict correctly, but this is worth
> knowing about since versions aren't pinned in `requirements.txt`.

## Running

Development server (as invoked directly in `app.py`, with Flask's debug mode
enabled):

```bash
python app.py
```

This starts the API on `http://127.0.0.1:5000`.

Production-style server, matching the `startCommand` in `render.yaml`:

```bash
gunicorn app:app
```

See [API_REFERENCE.md](API_REFERENCE.md) for the endpoint contract and
[GUIDE.md](GUIDE.md) for a walkthrough with real example requests.
