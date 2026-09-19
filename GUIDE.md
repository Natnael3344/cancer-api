# Developer Guide

A practical walkthrough for running the Cancer API locally and calling it.

## 1. Run it locally

```bash
git clone https://github.com/Natnael3344/cancer-api.git
cd cancer-api
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

You should see Flask start up and log something like:

```
 * Running on http://127.0.0.1:5000
```

Leave that process running in one terminal and use another terminal (or any
HTTP client) for the requests below.

To run it the way `render.yaml` runs it in deployment (via Gunicorn instead
of Flask's dev server):

```bash
gunicorn app:app
```

## 2. Make a sample request

The API has a single endpoint, `POST /predict`, which expects a JSON body
with a `features` array of exactly **30** numbers (see
[API_REFERENCE.md](API_REFERENCE.md) for the full contract).

### curl

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      20.57, 17.77, 132.9, 1326, 0.08474,
      0.07864, 0.0869, 0.07017, 0.1812, 0.05667,
      0.5435, 0.7339, 3.398, 74.08, 0.005225,
      0.01308, 0.0186, 0.0134, 0.01389, 0.003532,
      24.99, 23.41, 158.8, 1956, 0.1238,
      0.1866, 0.2416, 0.186, 0.275, 0.08902
    ]
  }'
```

Response (verified by running this exact request against the app):

```json
{
  "prediction": "Malignant"
}
```

### Python (using `requests`)

```python
import requests

payload = {
    "features": [
        20.57, 17.77, 132.9, 1326, 0.08474,
        0.07864, 0.0869, 0.07017, 0.1812, 0.05667,
        0.5435, 0.7339, 3.398, 74.08, 0.005225,
        0.01308, 0.0186, 0.0134, 0.01389, 0.003532,
        24.99, 23.41, 158.8, 1956, 0.1238,
        0.1866, 0.2416, 0.186, 0.275, 0.08902
    ]
}

response = requests.post("http://127.0.0.1:5000/predict", json=payload)
print(response.status_code, response.json())
# 200 {'prediction': 'Malignant'}
```

Sending 30 zeros instead produces a `"Benign"` prediction — this was also
verified directly against the running app; the exact boundary between the two
classes depends entirely on the trained `svm_model.pkl`, which isn't
documented beyond what's observable by calling the endpoint.

## 3. Common mistakes (and what they return)

These were all confirmed by hitting the running app directly — see
[API_REFERENCE.md](API_REFERENCE.md) for the full table:

- Omitting `"features"` entirely -> `500` (`KeyError: 'features'`)
- Sending the wrong number of values (e.g. 5 instead of 30) -> `500`
  (`ValueError: X has 5 features, but StandardScaler is expecting 30
  features as input.`)
- Sending an invalid/empty JSON body -> `400 Bad Request`
- Using `GET` instead of `POST` -> `405 Method Not Allowed`

There is no input validation in `app.py`, so it's on the caller to always
send exactly 30 numeric values in `features`.

## 4. Notes on the model

- The repository ships only the trained artifacts (`svm_model.pkl`,
  `scaler.pkl`, `pca.pkl`) and the inference code (`app.py`). There is no
  training script, notebook, or dataset in the repo, so the exact training
  data, feature order, and evaluation metrics are **not documented** and
  cannot be reconstructed from this repository alone.
- What can be verified by inspecting the pickled objects: the scaler expects
  30 input features, the PCA step reduces them to 10 components, and the
  classifier is a scikit-learn `SVC` with an RBF kernel predicting a binary
  label (`0`/`1`, mapped to `"Benign"`/`"Malignant"` in `app.py`). This shape
  (30 features, binary malignant/benign outcome) matches the well-known
  Breast Cancer Wisconsin (Diagnostic) dataset, but this repo does not state
  that anywhere, so treat it as a reasonable inference rather than a
  documented fact.
- The pickled objects were created with scikit-learn 1.6.1. `requirements.txt`
  does not pin a scikit-learn version, so installing a different version may
  print `InconsistentVersionWarning` messages on startup (this was observed
  first-hand while testing locally). Predictions still worked correctly in
  that scenario, but for guaranteed compatibility it's safest to install
  scikit-learn 1.6.1 specifically:

  ```bash
  pip install scikit-learn==1.6.1
  ```

- `app.py` calls `app.run(debug=True)`. Do not run it this way in production
  — Flask's debug mode exposes interactive stack traces (including local
  file paths) to anyone who can trigger a server error. The `render.yaml` in
  this repo correctly avoids this by using Gunicorn (`gunicorn app:app`)
  instead of `python app.py` for deployment.
