from flask import Flask, render_template, request
import pickle
import numpy as np
import MySQLdb

# Load trained model and scaler
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)
with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# MySQL database configuration
db = MySQLdb.connect(
    host="localhost",
    user="root",
    passwd="22875777",
    db="heart_disease_db"
)
cursor = db.cursor()

# app.py

app = Flask(__name__)

@app.route('/')
def hello_world():
    return "Hello, World!"


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input values from the form
        features = [
            int(request.form['age']),
            int(request.form['male']),
            int(request.form['cigsPerDay']),
            int(request.form['totChol']),
            int(request.form['sysBP']),
            int(request.form['glucose'])
        ]

        # Scale input data
        features_array = np.array([features])
        scaled_features = scaler.transform(features_array)

        # Predict using the model
        prediction = model.predict(scaled_features)[0]  # 0 = Low Risk, 1 = High Risk

        print("DEBUG - Model Prediction:", prediction)  # Check output in terminal

        result_text = "High Risk of Heart Disease" if prediction == 1 else "Low Risk of Heart Disease"

        return render_template('result.html', prediction=result_text,
                               age=features[0], male=features[1], cigsPerDay=features[2],
                               totChol=features[3], sysBP=features[4], glucose=features[5])

    except Exception as e:
        return f"Error: {str(e)}"



if __name__ == "__main__":
    app.run(debug=True)
