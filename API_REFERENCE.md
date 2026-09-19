# API Reference

This document describes the HTTP interface exposed by `app.py`. There is
exactly one route.

Base URL (local development): `http://127.0.0.1:5000`

All behavior below (status codes, error bodies) was verified by running the
app locally with `python app.py` and issuing real requests against it.

---

## `POST /predict`

Runs the saved preprocessing pipeline (`StandardScaler` -> `PCA`) and SVM
classifier on a feature vector and returns a Malignant/Benign prediction.

### Request

- **Method:** `POST`
- **Path:** `/predict`
- **Content-Type:** `application/json`

**Body:**

```json
{
  "features": [/* 30 numbers */]
}
```

| Field      | Type            | Required | Notes                                                                                                 |
|------------|-----------------|----------|---------------------------------------------------------------------------------------------------------|
| `features` | array of number | Yes      | Must contain exactly **30** numeric values. `app.py` reads `data['features']` and reshapes it into a `(1, 30)` array before scaling. The expected order/meaning of the 30 values is not documented or validated in code. |

The route reads the body with `request.get_json(force=True)`, so it will
attempt to parse the body as JSON even without a correct `Content-Type`
header, but the body must still be valid JSON.

### Processing pipeline (from `app.py`)

1. `features = np.array(data['features']).reshape(1, -1)`
2. `features_scaled = scaler.transform(features)` — the 30 raw values are
   standardized by the pre-fit `StandardScaler`.
3. `features_pca = pca.transform(features_scaled)` — reduced from 30 to 10
   components by the pre-fit `PCA`.
4. `prediction = model.predict(features_pca)` — the `SVC` (RBF kernel)
   predicts class `0` or `1`.
5. `'Malignant' if prediction[0] == 1 else 'Benign'`

### Success response

**`200 OK`**

```json
{
  "prediction": "Benign"
}
```

or

```json
{
  "prediction": "Malignant"
}
```

`prediction` is always one of the two literal strings `"Benign"` or
`"Malignant"`, derived from the underlying model's binary output (class `1`
-> `"Malignant"`, class `0` -> `"Benign"`).

### Error responses

`app.py` has no explicit input validation or exception handling, so errors
surface as Flask/Werkzeug's default error pages. Verified behavior:

| Condition                                              | Status | Body (dev server, `debug=True`)                                                                 |
|---------------------------------------------------------|--------|---------------------------------------------------------------------------------------------------|
| `features` key missing from the JSON body                | `500`  | Werkzeug interactive debugger page (dev only) showing `KeyError: 'features'`                     |
| `features` array has a length other than 30               | `500`  | Werkzeug debugger page showing `ValueError: X has N features, but StandardScaler is expecting 30 features as input.` |
| Body is not valid JSON / empty body                       | `400`  | `400 Bad Request` (Werkzeug's standard JSON-parsing error page)                                  |
| Wrong HTTP method (e.g. `GET /predict`)                    | `405`  | `405 Method Not Allowed`                                                                          |

**Important:** `app.py` calls `app.run(debug=True)` when run directly with
`python app.py`. Debug mode is what produces the full interactive traceback
page shown above for the `500` cases — it also means stack traces (including
file paths) are exposed to the client, which is not suitable for production.
When the app is served through Gunicorn instead (`gunicorn app:app`, as
`render.yaml` does), Flask's debug/reloader behavior from `app.run()` does not
apply, so those same errors would still return `500`/`400`/`405` but without
the debugger HTML — just a generic error response.

No other endpoints, authentication, rate limiting, or content negotiation are
implemented in this API.
