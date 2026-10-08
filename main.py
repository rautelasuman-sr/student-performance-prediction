import pandas as pd
from sklearn.model_selection import train_test_split

# --------------------------------
# 1. Load the dataset
# --------------------------------
data = pd.read_csv("dataset/student-mat.csv", sep=";")

print("Dataset loaded successfully!")
print("Shape:", data.shape)


# --------------------------------
# 2. Select input features
# --------------------------------
features = [
    "age",
    "studytime",
    "failures",
    "absences",
    "G1",
    "G2"
]

X = data[features]


# --------------------------------
# 3. Select target
# --------------------------------
y = data["G3"]


# --------------------------------
# 4. Split data into training and testing
# --------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# --------------------------------
# 5. Display dataset sizes
# --------------------------------
print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

print("\nTraining target:", y_train.shape)
print("Testing target:", y_test.shape)

from sklearn.linear_model import LinearRegression

# Create the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

print("\nModel trained successfully!")

# Make predictions on testing data
y_pred = model.predict(X_test)

print("\nPredictions made successfully!")

# Show first 5 predictions
print("\nActual vs Predicted:")
for actual, predicted in zip(y_test.head(), y_pred[:5]):
    print("Actual:", actual, "Predicted:", round(predicted, 2))


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("MAE:", round(mae, 2))
print("MSE:", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R² Score:", round(r2, 2))


import matplotlib.pyplot as plt

# Actual vs Predicted values
plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Final Grade (G3)")
plt.ylabel("Predicted Final Grade (G3)")
plt.title("Actual vs Predicted Final Grades")

# Perfect prediction line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.tight_layout()
plt.savefig("visualizations/actual_vs_predicted.png")
plt.show()


# Predict final grade for a new student

print("\n--- New Student Prediction ---")

age = float(input("Enter age: "))
studytime = float(input("Enter study time (1-4): "))
failures = float(input("Enter number of past failures: "))
absences = float(input("Enter number of absences: "))
G1 = float(input("Enter first period grade (G1): "))
G2 = float(input("Enter second period grade (G2): "))

new_student = [[
    age,
    studytime,
    failures,
    absences,
    G1,
    G2
]]

predicted_grade = model.predict(new_student)

print("\nPredicted Final Grade (G3):", round(predicted_grade[0], 2))
