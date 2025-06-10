
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import os

# Resolve path to cars.csv
script_dir = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(script_dir, '..', 'cars.csv'))

# Features and target
X = df.drop('sale_price', axis=1).select_dtypes(include=['number'])
y = df['sale_price']

# Train-test split
_, X_test, _, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Save test set to CSV
test_data = X_test.copy()
test_data['actual_sale_price'] = y_test
test_data.to_csv('test_data.csv', index=False)

# Load the trained model
model = joblib.load('model.joblib')

# Make predictions
y_pred = model.predict(X_test)

# Compute metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print metrics
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# Save metrics to a file
with open('metrics.txt', 'w') as f:
    f.write(f"Mean Absolute Error: {mae:.2f}\n")
    f.write(f"Mean Squared Error: {mse:.2f}\n")
    f.write(f"R^2 Score: {r2:.2f}\n")
