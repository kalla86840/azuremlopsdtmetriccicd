
import pandas as pd
from sklearn.linear_model import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

# Resolve path to cars.csv
script_dir = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(script_dir, '..', 'cars.csv'))

# Features and target

# Create multi-class labels based on mpg
# 0 = low, 1 = medium, 2 = high
df['mpg_class'] = pd.qcut(df['mpg'], q=3, labels=[0, 1, 2]).astype(int)


median_mpg = df['mpg'].median()
df['mpg_class'] = (df['mpg'] > median_mpg).astype(int)


median_price = df['mpg'].median()
df['mpg_class'] = (df['mpg'] > median_price).astype(int)

X = df.drop(['mpg', 'mpg_class'], axis=1).select_dtypes(include=['number'])
y = df['mpg_class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'model.joblib')

print("Training completed and model saved to 'model.joblib'")
