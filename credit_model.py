import pandas as pd

# Load the dataset
df = pd.read_csv("credit_data.csv")

# Display first 5 rows
print("First 5 rows of the dataset:")
print(df.head())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Summary of the dataset
print("\nData info:")
print(df.info())
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Drop missing values (we already checked this, but it's safe to include)
df = df.dropna()

# Separate features and target
X = df.drop("Default", axis=1)  # All columns except 'Default'
y = df["Default"]               # The target column

# Scale features (standardize)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("\nData has been preprocessed and split into training and testing sets.")
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Initialize the models
lr_model = LogisticRegression()
dt_model = DecisionTreeClassifier()
rf_model = RandomForestClassifier()

# Train the models on the training data
lr_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

print("\nModels have been trained successfully!")
from sklearn.metrics import classification_report, accuracy_score

# Predict on test data
lr_preds = lr_model.predict(X_test)
dt_preds = dt_model.predict(X_test)
rf_preds = rf_model.predict(X_test)

# Evaluate models
print("\n📊 Logistic Regression Performance:")
print(classification_report(y_test, lr_preds))

print("📊 Decision Tree Performance:")
print(classification_report(y_test, dt_preds))

print("📊 Random Forest Performance:")
print(classification_report(y_test, rf_preds))
