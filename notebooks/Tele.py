# Import libraries for reading data, training the model, checking results, and saving the model
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Use the same random value every time so the train-test split stays the same
RANDOM_STATE = 42


# Find the main RetailMind project folder automatically
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Create the paths for the dataset, model folder, and report folder
DATA_PATH = PROJECT_ROOT / "data" / "churn.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"


# Create the model and report folders if they do not already exist
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


# Read the bank customer churn dataset
df = pd.read_csv(DATA_PATH)


# Show basic information about the dataset
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())


# Remove completely duplicate rows
df = df.drop_duplicates()


# Check that the target column exists
if "Exited" not in df.columns:
    raise ValueError("The dataset must contain a column named Exited.")


# Show how many customers stayed and exited
print("\nTarget counts:")
print(df["Exited"].value_counts())

print("\nTarget percentages:")
print(df["Exited"].value_counts(normalize=True) * 100)


# Remove identifiers and the answer column from the model inputs
columns_to_remove = [
    column
    for column in [
        "RowNumber",
        "CustomerId",
        "Surname",
        "Exited"
    ]
    if column in df.columns
]

X = df.drop(columns=columns_to_remove)


# Store the answer that the model needs to predict
y = df["Exited"].astype(int)


# Split the data before fitting any preprocessing steps
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)


# Show the size of the training and testing data
print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))


# Find numerical columns using only the training data
numerical_columns = X_train.select_dtypes(
    include=["number"]
).columns.tolist()


# Find text columns using only the training data
categorical_columns = X_train.select_dtypes(
    exclude=["number"]
).columns.tolist()


# Show the columns found in each group
print("\nNumerical columns:")
print(numerical_columns)

print("\nCategorical columns:")
print(categorical_columns)


# Fill missing numerical values and standardize numerical columns
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


# Fill missing text values and convert categories into numerical columns
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


# Apply the correct preprocessing to numerical and text columns
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


# Create one pipeline containing preprocessing and Logistic Regression
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
                random_state=RANDOM_STATE
            )
        )
    ]
)


# Train the complete pipeline using only the training data
model.fit(X_train, y_train)


# Predict the class of every customer in the test data
y_pred = model.predict(X_test)


# Predict the exit probability of every customer in the test data
y_probability = model.predict_proba(X_test)[:, 1]


# Calculate the main model evaluation scores
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


# Display the Day 1 baseline results
print("\n================================")
print("DAY 1 LOGISTIC REGRESSION")
print("================================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


# Show the correct and incorrect prediction counts
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Show detailed results for stayed and exited customers
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


# Save the Day 1 results as a CSV report
day1_results = pd.DataFrame(
    [
        {
            "Model": "Logistic Regression",
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC-AUC": roc_auc
        }
    ]
)

day1_results.to_csv(
    REPORT_DIR / "day1_results.csv",
    index=False
)


# Save the complete Day 1 pipeline in a separate model file
model_path = MODEL_DIR / "day1_logistic_pipeline.joblib"

joblib.dump(
    model,
    model_path
)


# Load the model again to verify that the saved file works
loaded_model = joblib.load(model_path)


# Confirm that the loaded model gives the same predictions
same_predictions = (
    loaded_model.predict(X_test) == y_pred
).all()

print("\nModel saved at:", model_path)
print("Saved model works correctly:", same_predictions)


# Create one example customer for checking the saved model
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


# Predict whether the example customer may exit
sample_prediction = loaded_model.predict(
    sample_customer
)[0]


# Find the predicted exit probability for the example customer
sample_probability = loaded_model.predict_proba(
    sample_customer
)[0][1]


# Convert the numerical prediction into simple text
prediction_text = (
    "Customer may exit"
    if sample_prediction == 1
    else "Customer may stay"
)


# Display the example prediction
print("\nSample prediction:", prediction_text)

print(
    "Exit probability:",
    round(sample_probability * 100, 2),
    "%"
)

# Import the three new models and clone so every model gets its own preprocessing pipeline
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# Create the three models required for Day 2
day2_model_choices = {
    "Decision Tree": DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1
    ),

    "SVM": SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )
}


# Create empty containers for the model results and trained pipelines
day2_results = []
day2_trained_models = {}


# Train and test all three models using the same training and testing data
for model_name, classifier in day2_model_choices.items():

    # Create a separate preprocessing and model pipeline for the current algorithm
    current_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                clone(preprocessor)
            ),
            (
                "classifier",
                classifier
            )
        ]
    )

    # Train the complete pipeline using the Day 1 training data
    current_pipeline.fit(
        X_train,
        y_train
    )

    # Predict whether every test customer stayed or exited
    current_predictions = current_pipeline.predict(
        X_test
    )

    # Calculate the probability that every test customer exited
    current_probabilities = current_pipeline.predict_proba(
        X_test
    )[:, 1]

    # Calculate the accuracy of the current model
    current_accuracy = accuracy_score(
        y_test,
        current_predictions
    )

    # Calculate the precision for customers predicted to exit
    current_precision = precision_score(
        y_test,
        current_predictions,
        zero_division=0
    )

    # Calculate how many actual exiting customers the model found
    current_recall = recall_score(
        y_test,
        current_predictions,
        zero_division=0
    )

    # Calculate a combined score using precision and recall
    current_f1 = f1_score(
        y_test,
        current_predictions,
        zero_division=0
    )

    # Calculate how well the model separates stayed and exited customers
    current_roc_auc = roc_auc_score(
        y_test,
        current_probabilities
    )

    # Keep the trained pipeline so the winning model can be saved
    day2_trained_models[
        model_name
    ] = current_pipeline

    # Store the current model's scores in a list
    day2_results.append(
        {
            "Model": model_name,
            "Accuracy": current_accuracy,
            "Precision": current_precision,
            "Recall": current_recall,
            "F1 Score": current_f1,
            "ROC-AUC": current_roc_auc
        }
    )

    # Display the current model's results
    print("\n================================")
    print("DAY 2:", model_name.upper())
    print("================================")

    print(f"Accuracy : {current_accuracy:.4f}")
    print(f"Precision: {current_precision:.4f}")
    print(f"Recall   : {current_recall:.4f}")
    print(f"F1 Score : {current_f1:.4f}")
    print(f"ROC-AUC  : {current_roc_auc:.4f}")

    # Show correct and incorrect predictions for the current model
    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            current_predictions
        )
    )

    # Show detailed results for stayed and exited customers
    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            current_predictions,
            target_names=[
                "Stayed",
                "Exited"
            ],
            zero_division=0
        )
    )


# Convert the Day 2 results into a comparison table
day2_comparison = pd.DataFrame(
    day2_results
)


# Rank models mainly by recall because finding exiting customers is the priority
day2_comparison = day2_comparison.sort_values(
    by=[
        "Recall",
        "ROC-AUC",
        "F1 Score"
    ],
    ascending=False
).reset_index(drop=True)


# Display the complete Day 2 comparison table
print("\n================================")
print("DAY 2 MODEL COMPARISON")
print("================================")

print(
    day2_comparison.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)


# Select the first model after sorting by recall
day2_champion_name = day2_comparison.loc[
    0,
    "Model"
]


# Get the trained pipeline belonging to the Day 2 winner
day2_champion_model = day2_trained_models[
    day2_champion_name
]


# Choose a separate filename so the Day 1 model is not overwritten
day2_champion_path = (
    MODEL_DIR
    / "day2_champion_pipeline.joblib"
)


# Save the complete Day 2 champion pipeline
joblib.dump(
    day2_champion_model,
    day2_champion_path
)


# Choose a location for the Day 2 comparison report
day2_report_path = (
    REPORT_DIR
    / "day2_model_comparison.csv"
)


# Save the comparison table as a CSV file
day2_comparison.to_csv(
    day2_report_path,
    index=False
)


# Display the selected Day 2 winner and saved files
print("\n================================")
print("DAY 2 CHAMPION")
print("================================")

print("Selected model:", day2_champion_name)

print(
    "Reason: The model had the highest recall, "
    "with ROC-AUC and F1 Score used to break ties."
)

print("Model saved at:", day2_champion_path)
print("Report saved at:", day2_report_path)