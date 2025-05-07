from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load saved model and preprocessors
model = joblib.load('C:/Users/natna/OneDrive/Desktop/Cancer Research/svm_model.pkl')
scaler = joblib.load('C:/Users/natna/OneDrive/Desktop/Cancer Research/scaler.pkl')
pca = joblib.load('C:/Users/natna/OneDrive/Desktop/Cancer Research/pca.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    # Convert input data to NumPy array
    features = np.array(data['features']).reshape(1, -1)
    
    # Preprocess (scale + PCA)
    features_scaled = scaler.transform(features)
    features_pca = pca.transform(features_scaled)

    # Predict
    prediction = model.predict(features_pca)
    result = 'Malignant' if prediction[0] == 1 else 'Benign'
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
