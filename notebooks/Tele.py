# Import libraries needed to work with data, train the model, test it, and save it
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# Read the customer churn CSV file
df = pd.read_csv("data/churn.csv")


# Show the first 5 rows of the dataset
print("\nFirst 5 rows:")
print(df.head())


# Show the number of rows and columns
print("\nDataset shape:")
print(df.shape)


# Show all column names
print("\nColumn names:")
print(df.columns.tolist())


# Show information about each column
print("\nDataset information:")
df.info()


# Check how many missing values are present in each column
print("\nMissing values:")
print(df.isnull().sum())


# Check how many duplicate rows are present
print("\nDuplicate rows:")
print(df.duplicated().sum())


# Remove duplicate rows
df = df.drop_duplicates()


# Show how many customers stayed (0) and exited (1)
print("\nExited counts:")
print(df["Exited"].value_counts())


# Show the percentage of customers who stayed and exited
print("\nExited percentages:")
print(df["Exited"].value_counts(normalize=True) * 100)


# X contains the customer information that will be used to make predictions
X = df.drop(
    columns=[
        "CustomerId",
        "Surname",
        "Exited"
    ]
)


# y contains the answer we want the model to predict
y = df["Exited"]


# Show the columns that will actually be used for training
print("\nFeatures used for training:")
print(X.columns.tolist())


# Use 80% of the data for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Show how many rows are being used for training and testing
print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))


# Find all columns that contain numbers
numerical_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()


# Find all columns that contain text
categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()


# Show the numerical columns
print("\nNumerical columns:")
print(numerical_columns)


# Show the categorical columns
print("\nCategorical columns:")
print(categorical_columns)


# Fill missing numbers with the median and standardize their values
numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# Fill missing text values and convert text categories into numbers
categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)


# Apply the correct preprocessing to numerical and categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_pipeline,
            numerical_columns
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_columns
        )
    ]
)


# Create one pipeline that preprocesses the data and then uses Logistic Regression
model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)


# Train the model using the training data
model.fit(X_train, y_train)

print("\nModel trained successfully!")


# Predict whether each customer in the test data stayed or exited
y_pred = model.predict(X_test)


# Find the predicted exit probability for each test customer
y_probability = model.predict_proba(X_test)[:, 1]


# Calculate the percentage of total predictions that were correct
accuracy = accuracy_score(
    y_test,
    y_pred
)


# Calculate how often an exit prediction was actually correct
precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)


# Calculate how many actual exiting customers the model found
recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)


# Calculate how well the model separates exiting and staying customers
roc_auc = roc_auc_score(
    y_test,
    y_probability
)


# Display the main Day 1 model results
print("\n==============================")
print("DAY 1 BASELINE MODEL RESULTS")
print("==============================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


# Show the numbers of correct and incorrect predictions
print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# Show detailed precision, recall and F1-score results
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Stayed",
            "Exited"
        ],
        zero_division=0
    )
)


# Create the models folder if it does not exist
os.makedirs(
    "models",
    exist_ok=True
)


# Choose the location where the trained model will be saved
model_path = "models/day1_churn_pipeline.joblib"


# Save the complete trained pipeline
joblib.dump(
    model,
    model_path
)

print("\nModel saved successfully!")
print("Saved at:", model_path)


# Load the saved model to make sure the file works
loaded_model = joblib.load(
    model_path
)


# Create one example customer for testing our model
sample_customer = pd.DataFrame(
    [
        {
            "CreditScore": 619,
            "Geography": "France",
            "Gender": "Female",
            "Age": 42,
            "Tenure": 2,
            "Balance": 0.0,
            "NumOfProducts": 1,
            "HasCrCard": 1,
            "IsActiveMember": 1,
            "EstimatedSalary": 101348.88
        }
    ]
)


# Predict whether the example customer will stay or exit
sample_prediction = loaded_model.predict(
    sample_customer
)[0]


# Find the predicted probability that the example customer will exit
sample_probability = loaded_model.predict_proba(
    sample_customer
)[0][1]


# Convert the 0 or 1 prediction into simple text
if sample_prediction == 1:
    prediction_text = "Customer may exit"
else:
    prediction_text = "Customer may stay"


# Show the final prediction for our example customer
print("\n==============================")
print("SAMPLE CUSTOMER PREDICTION")
print("==============================")

print("Prediction:", prediction_text)

print(
    "Exit probability:",
    round(sample_probability * 100, 2),
    "%"
)