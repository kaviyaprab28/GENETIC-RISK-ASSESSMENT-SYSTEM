
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model and preprocessing pipeline
model = joblib.load("genetic_disorder_model.joblib")
preprocessor = joblib.load("genetic_disorder_preprocessor.joblib")

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Genetic Disorder Risk Assessment API is running"
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        patient_data = request.get_json()

        if not patient_data:
            return jsonify({"error": "No input data provided"}), 400

        input_df = pd.DataFrame([patient_data])
        transformed_data = preprocessor.transform(input_df)
        prediction = model.predict(transformed_data)[0]

        return jsonify({"prediction": str(prediction)})

    except Exception as error:
        return jsonify({"error": str(error)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
