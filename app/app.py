# from flask import Flask, render_template, request
# import pickle
# import numpy as np
# import mysql.connector
# import os
# import logging

# # Initialize Flask app
# app = Flask(__name__)

# # Load trained model and scaler with proper error handling
# model, scaler = None, None

# # Check if model.pkl exists
# if os.path.exists("model.pkl"):
#     try:
#         with open("model.pkl", "rb") as model_file:
#             model = pickle.load(model_file)
#         print("✅ DEBUG: Model loaded successfully!")
#     except Exception as e:
#         print(f"❌ ERROR: Failed to load model - {str(e)}")
# else:
#     print("❌ ERROR: model.pkl file is missing!")

# # Check if scaler.pkl exists
# if os.path.exists("scaler.pkl"):
#     try:
#         with open("scaler.pkl", "rb") as scaler_file:
#             scaler = pickle.load(scaler_file)
#         print("✅ DEBUG: Scaler loaded successfully!")
#     except Exception as e:
#         print(f"❌ ERROR: Failed to load scaler - {str(e)}")
# else:
#     print("❌ ERROR: scaler.pkl file is missing!")

# # MySQL database configuration
# DB_CONFIG = {
#     "host": "localhost",
#     "user": "root",
#     "password": "22875777",  # Replace with a secure method (e.g., env variables)
#     "database": "heart_disease_db"
# }

# def get_db_connection():
#     """Establish a connection to the database."""
#     try:
#         return mysql.connector.connect(**DB_CONFIG)
#     except mysql.connector.Error as e:
#         logging.error(f"❌ Database connection failed: {str(e)}")
#         return None

# # Home route
# @app.route("/")
# def home():
#     return render_template('index.html')

# # Prediction route
# @app.route('/predict', methods=['POST'])
# def predict():
#     if model is None or scaler is None:
#         return "❌ Error: Model or Scaler not loaded. Please check the files and restart the app."

#     try:
#         # Extract input values safely
#         features = [
#             int(request.form.get('age', 0)),
#             int(request.form.get('male', 0)),
#             int(request.form.get('cigsPerDay', 0)),
#             int(request.form.get('totChol', 0)),
#             int(request.form.get('sysBP', 0)),
#             int(request.form.get('glucose', 0))
#         ]

#         print(f"✅ DEBUG: Received input - {features}")

#         # Ensure correct shape for model
#         features_array = np.array([features]).reshape(1, -1)

#         # Scale input
#         scaled_features = scaler.transform(features_array)

#         # Predict using the model
#         prediction = model.predict(scaled_features)[0]  # 0 = Low Risk, 1 = High Risk
#         result_text = "High Risk of Heart Disease" if prediction == 1 else "Low Risk of Heart Disease"

#         print(f"✅ DEBUG: Prediction result - {result_text}")

#         return render_template('result.html', prediction=result_text,
#                                age=features[0], male=features[1], cigsPerDay=features[2],
#                                totChol=features[3], sysBP=features[4], glucose=features[5])

#     except Exception as e:
#         logging.error(f"❌ Prediction error: {str(e)}")
#         return f"❌ Error: {str(e)}"

# if __name__ == "__main__":
#     app.run(debug=True, port=2100)
 
 
from flask import Flask, render_template, request
import pickle
import numpy as np
import mysql.connector
import os
import logging

# Initialize Flask app
app = Flask(__name__)

# Load trained model and scaler
model, scaler = None, None

# Check if model.pkl exists
if os.path.exists("model.pkl"):
    try:
        with open("model.pkl", "rb") as model_file:
            model = pickle.load(model_file)
        print("✅ DEBUG: Model loaded successfully!")
    except Exception as e:
        print(f"❌ ERROR: Failed to load model - {str(e)}")
else:
    print("❌ ERROR: model.pkl file is missing!")

# Check if scaler.pkl exists
if os.path.exists("scaler.pkl"):
    try:
        with open("scaler.pkl", "rb") as scaler_file:
            scaler = pickle.load(scaler_file)
        print("✅ DEBUG: Scaler loaded successfully!")
    except Exception as e:
        print(f"❌ ERROR: Failed to load scaler - {str(e)}")
else:
    print("❌ ERROR: scaler.pkl file is missing!")

# MySQL database configuration
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "22875777",  # Replace with a secure method (e.g., env variables)
    "database": "heart_disease_db"
}

def get_db_connection():
    """Establish a connection to the database."""
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except mysql.connector.Error as e:
        logging.error(f"❌ Database connection failed: {str(e)}")
        return None

# Home route
@app.route("/")
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return "❌ Error: Model or Scaler not loaded. Please check the files and restart the app."

    try:
        # Extract input values safely
        features = [
            int(request.form.get('age', 0)),
            int(request.form.get('male', 0)),
            int(request.form.get('cigsPerDay', 0)),
            int(request.form.get('totChol', 0)),
            int(request.form.get('sysBP', 0)),
            int(request.form.get('glucose', 0))
        ]

        print(f"✅ DEBUG: Received input - {features}")

        # Ensure correct shape for model
        features_array = np.array([features]).reshape(1, -1)

        # Scale input
        scaled_features = scaler.transform(features_array)

        # Predict using the model
        prediction = model.predict(scaled_features)[0]  # 0 = Low Risk, 1 = High Risk
        result_text = "High Risk of Heart Disease" if prediction == 1 else "Low Risk of Heart Disease"

        print(f"✅ DEBUG: Prediction result - {result_text}")

        # ✅ Store the prediction data in the database
        db_connection = get_db_connection()
        if db_connection:
            cursor = db_connection.cursor()
            sql_query = """
    INSERT INTO patient_data (age, male, cigsPerDay, totChol, sysBP, glucose, result)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
"""


            cursor.execute(sql_query, (features[0], features[1], features[2], features[3], features[4], features[5], result_text))
            db_connection.commit()
            cursor.close()
            db_connection.close()
            print("✅ DEBUG: Data successfully inserted into MySQL!")

        return render_template('result.html', prediction=result_text,
                               age=features[0], male=features[1], cigsPerDay=features[2],
                               totChol=features[3], sysBP=features[4], glucose=features[5])

    except Exception as e:
        logging.error(f"❌ Prediction error: {str(e)}")
        return f"❌ Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True, port=12210)
