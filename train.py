import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

# Define paths
DATA_PATH = "sonar_data.csv"
MODEL_PATH = "model.pkl"

print("Loading dataset...")
sonar_data = pd.read_csv(DATA_PATH, header=None)

# Separate features and target
X = sonar_data.drop(columns=[60])
Y = sonar_data[60]

print("Training Logistic Regression model...")
model = LogisticRegression(solver='lbfgs', max_iter=1000)
model.fit(X, Y)

print(f"Saving model to {MODEL_PATH}...")
joblib.dump(model, MODEL_PATH)
print("Training completed successfully!")
