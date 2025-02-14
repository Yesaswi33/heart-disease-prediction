# 🏥 Heart Disease Prediction using Machine Learning

This is a **Flask-based web application** that predicts the risk of heart disease based on patient health data.  
It uses **Random Forest Classifier** and applies **SMOTE** to handle data imbalance.

---

## 🚀 Features
- **Machine Learning Model:** Uses **Random Forest Classifier** for accurate predictions.
- **Flask Web App:** Interactive UI for users to input health data.
- **Data Balancing:** Applies **SMOTE (Synthetic Minority Over-sampling Technique)** to handle imbalanced data.
- **MySQL Database Integration:** Stores patient predictions in a database.
- **Deployment-Ready:** Can be hosted on **Render, Heroku, or AWS**.

---

## 🛠️ Tech Stack
- **Backend:** Flask, Python
- **Frontend:** HTML, CSS
- **Machine Learning:** Scikit-Learn, Pandas, NumPy
- **Database:** MySQL
- **Deployment:** Render (Optional)

---

## 📂 Project Structure
📦 heart-disease-prediction ┣ 📂 ml_model ┃ ┣ 📜 train_model.py # ML Model Training ┃ ┣ 📜 model.pkl # Trained Model ┃ ┣ 📜 scaler.pkl # Scaler for Preprocessing ┣ 📂 templates ┃ ┣ 📜 index.html # User Input Page ┃ ┣ 📜 result.html # Prediction Result Page ┣ 📜 app.py # Flask App ┣ 📜 requirements.txt # Dependencies for Deployment ┣ 📜 README.md # Project Documentation


---

## ⚙️ Installation & Setup

### 🔹 **1️⃣ Clone the Repository**
```bash
git clone https://github.com/Yesaswi33/heart-disease-prediction.git
cd heart-disease-prediction
🔹 2️⃣ Create a Virtual Environment & Install Dependencies
python3 -m venv venv
source venv/bin/activate  # For Mac/Linux
venv\Scripts\activate     # For Windows

pip install -r requirements.txt
🔹 3️⃣ Train the Model
python ml_model/train_model.py
🔹 4️⃣ Run the Flask Web App
python app.py
🌍 Open http://127.0.0.1:5001/ in your browser.

