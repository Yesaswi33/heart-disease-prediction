import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

# Loading dataset
df = pd.read_csv("framingham.csv")

# removing rows with missing values
df.dropna(inplace=True)

# Check class distribution before balancing to chech no of zeroes are eql with no of ones
print("Original class distribution:")
print(df["TenYearCHD"].value_counts())

# Select only the features used in the UI
selected_features = ["age", "male", "cigsPerDay", "totChol", "sysBP", "glucose"]
X = df[selected_features]
y = df["TenYearCHD"]

# Apply SMOTE to balance the dataset to smoothing then data set
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Check class distribution after SMOTE
print("New class distribution after SMOTE:")
print(pd.Series(y_resampled).value_counts())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)





#ml

# Train Random Forest model also a decession tree
model = RandomForestClassifier(
    n_estimators=200,  # More trees for better learning
    max_depth=10,  # Allow deeper learning
    min_samples_split=5,  # Avoid overfitting
    random_state=42
)
model.fit(X_train, y_train)





# Predictions
y_pred = model.predict(X_test)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print evaluation results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")





# Save model and scaler
with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)
with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)






print("New class distribution after SMOTE:")
print(pd.Series(y_resampled).value_counts())  # Should print roughly equal 0s and 1s after smoothing the dataset




# for chcking  


test_cases = [
    [30, 0, 0, 180, 110, 90],  # Low risk
    [65, 1, 20, 250, 160, 150],  # High risk
    [50, 1, 5, 200, 130, 110],  # Moderate risk similarly with low risk
]

# Scale inputs and predict
for test in test_cases:
    scaled_test = scaler.transform([test])
    prediction = model.predict(scaled_test)
    print(f"Input: {test} --> Prediction: {prediction[0]}")

