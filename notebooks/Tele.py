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


# Import GridSearchCV, cross-validation, and learning-curve tools for Day 3
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    learning_curve
)


# Create five balanced folds so every model uses the same cross-validation process
day3_cross_validation = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)


# Create fresh model objects that GridSearchCV will tune
day3_candidate_models = {
    "Decision Tree": DecisionTreeClassifier(
        random_state=RANDOM_STATE
    ),

    "Random Forest": RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "SVM": SVC(
        probability=True,
        random_state=RANDOM_STATE
    )
}


# Define the Decision Tree settings that GridSearchCV will test
day3_decision_tree_grid = {
    "classifier__max_depth": [
        3,
        5,
        8,
        None
    ],

    "classifier__min_samples_split": [
        2,
        5,
        10
    ],

    "classifier__min_samples_leaf": [
        1,
        5,
        10
    ],

    "classifier__class_weight": [
        None,
        "balanced"
    ]
}


# Define the Random Forest settings that GridSearchCV will test
day3_random_forest_grid = {
    "classifier__n_estimators": [
        100,
        200
    ],

    "classifier__max_depth": [
        None,
        6,
        10
    ],

    "classifier__min_samples_split": [
        2,
        5
    ],

    "classifier__min_samples_leaf": [
        1,
        3
    ],

    "classifier__class_weight": [
        None,
        "balanced"
    ]
}


# Define the SVM settings that GridSearchCV will test
day3_svm_grid = {
    "classifier__C": [
        0.1,
        1,
        10
    ],

    "classifier__kernel": [
        "linear",
        "rbf"
    ],

    "classifier__gamma": [
        "scale",
        "auto"
    ],

    "classifier__class_weight": [
        None,
        "balanced"
    ]
}


# Connect every candidate model with its correct search settings
day3_parameter_grids = {
    "Decision Tree": day3_decision_tree_grid,
    "Random Forest": day3_random_forest_grid,
    "SVM": day3_svm_grid
}


# Create empty containers for tuned models and their results
day3_tuned_models = {}
day3_results = []


# Tune all three models using only the training data
for model_name, classifier in day3_candidate_models.items():

    # Display the model that GridSearchCV is currently tuning
    print("\n================================")
    print("DAY 3 TUNING:", model_name.upper())
    print("================================")

    # Create a fresh preprocessing and model pipeline
    search_pipeline = Pipeline(
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

    # Try every parameter combination and select the best recall score
    grid_search = GridSearchCV(
        estimator=search_pipeline,
        param_grid=day3_parameter_grids[
            model_name
        ],
        scoring="recall",
        cv=day3_cross_validation,
        n_jobs=-1,
        return_train_score=True,
        refit=True
    )

    # Run parameter tuning using only the training data
    grid_search.fit(
        X_train,
        y_train
    )

    # Get the best complete pipeline found for the current model
    best_model = grid_search.best_estimator_

    # Store the tuned pipeline for later champion selection
    day3_tuned_models[
        model_name
    ] = best_model

    # Find the result position of the best parameter combination
    best_position = grid_search.best_index_

    # Read the average training recall of the best parameter combination
    mean_train_recall = grid_search.cv_results_[
        "mean_train_score"
    ][best_position]

    # Read the average validation recall of the best parameter combination
    mean_validation_recall = grid_search.cv_results_[
        "mean_test_score"
    ][best_position]

    # Calculate the difference between training and validation recall
    train_validation_gap = (
        mean_train_recall
        - mean_validation_recall
    )

    # Give a simple diagnosis using the training and validation scores
    if (
        mean_train_recall < 0.70
        and mean_validation_recall < 0.70
    ):
        diagnosis = (
            "Possible underfitting or high bias"
        )

    elif train_validation_gap > 0.10:
        diagnosis = (
            "Possible overfitting or high variance"
        )

    else:
        diagnosis = (
            "Training and validation scores are balanced"
        )

    # Predict the classes of the final test customers
    tuned_predictions = best_model.predict(
        X_test
    )

    # Calculate exit probabilities for the final test customers
    tuned_probabilities = best_model.predict_proba(
        X_test
    )[:, 1]

    # Calculate the final test accuracy
    tuned_accuracy = accuracy_score(
        y_test,
        tuned_predictions
    )

    # Calculate the final test precision
    tuned_precision = precision_score(
        y_test,
        tuned_predictions,
        zero_division=0
    )

    # Calculate the final test recall
    tuned_recall = recall_score(
        y_test,
        tuned_predictions,
        zero_division=0
    )

    # Calculate the final test F1 score
    tuned_f1 = f1_score(
        y_test,
        tuned_predictions,
        zero_division=0
    )

    # Calculate the final test ROC-AUC
    tuned_roc_auc = roc_auc_score(
        y_test,
        tuned_probabilities
    )

    # Store the tuning and final test information
    day3_results.append(
        {
            "Model": model_name,
            "Train Recall": mean_train_recall,
            "Validation Recall": (
                mean_validation_recall
            ),
            "Recall Gap": train_validation_gap,
            "Test Accuracy": tuned_accuracy,
            "Test Precision": tuned_precision,
            "Test Recall": tuned_recall,
            "Test F1": tuned_f1,
            "Test ROC-AUC": tuned_roc_auc,
            "Diagnosis": diagnosis,
            "Best Parameters": str(
                grid_search.best_params_
            )
        }
    )

    # Display the settings and scores found by GridSearchCV
    print("\nBest parameters:")
    print(grid_search.best_params_)

    print(
        f"\nTraining recall  : "
        f"{mean_train_recall:.4f}"
    )

    print(
        f"Validation recall: "
        f"{mean_validation_recall:.4f}"
    )

    print(
        f"Recall gap       : "
        f"{train_validation_gap:.4f}"
    )

    print("Diagnosis        :", diagnosis)

    print(
        f"\nTest accuracy : "
        f"{tuned_accuracy:.4f}"
    )

    print(
        f"Test precision: "
        f"{tuned_precision:.4f}"
    )

    print(
        f"Test recall   : "
        f"{tuned_recall:.4f}"
    )

    print(
        f"Test F1       : "
        f"{tuned_f1:.4f}"
    )

    print(
        f"Test ROC-AUC  : "
        f"{tuned_roc_auc:.4f}"
    )

    # Display the correct and incorrect final test predictions
    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            tuned_predictions
        )
    )


# Convert all Day 3 tuning results into a comparison table
day3_comparison = pd.DataFrame(
    day3_results
)


# Select the champion using validation recall instead of test recall
day3_comparison = day3_comparison.sort_values(
    by=[
        "Validation Recall",
        "Test ROC-AUC",
        "Test F1"
    ],
    ascending=False
).reset_index(drop=True)


# Display the complete Day 3 comparison
print("\n================================")
print("DAY 3 TUNED MODEL COMPARISON")
print("================================")

day3_display_columns = [
    "Model",
    "Train Recall",
    "Validation Recall",
    "Recall Gap",
    "Test Accuracy",
    "Test Precision",
    "Test Recall",
    "Test F1",
    "Test ROC-AUC",
    "Diagnosis"
]

print(
    day3_comparison[
        day3_display_columns
    ].to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)


# Get the name of the model with the strongest validation recall
day3_champion_name = day3_comparison.loc[
    0,
    "Model"
]


# Get the complete tuned pipeline belonging to the champion
day3_champion_model = day3_tuned_models[
    day3_champion_name
]


# Create learning-curve scores using increasing amounts of training data
training_sizes, training_scores, validation_scores = learning_curve(
    estimator=day3_champion_model,
    X=X_train,
    y=y_train,
    train_sizes=[
        0.20,
        0.40,
        0.60,
        0.80,
        1.00
    ],
    cv=day3_cross_validation,
    scoring="recall",
    n_jobs=-1,
    shuffle=True,
    random_state=RANDOM_STATE
)


# Calculate the average training recall for every data size
mean_learning_train_scores = (
    training_scores.mean(axis=1)
)


# Calculate the average validation recall for every data size
mean_learning_validation_scores = (
    validation_scores.mean(axis=1)
)


# Store the learning-curve values in a table
day3_learning_curve = pd.DataFrame(
    {
        "Training Examples": training_sizes,
        "Mean Training Recall": (
            mean_learning_train_scores
        ),
        "Mean Validation Recall": (
            mean_learning_validation_scores
        )
    }
)


# Display how performance changes when more training data is used
print("\n================================")
print("DAY 3 CHAMPION LEARNING CURVE")
print("================================")

print(
    day3_learning_curve.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)


# Choose a separate path so earlier model versions remain unchanged
day3_champion_path = (
    MODEL_DIR
    / "day3_tuned_champion.joblib"
)


# Save the complete tuned champion pipeline
joblib.dump(
    day3_champion_model,
    day3_champion_path
)


# Choose a location for the Day 3 comparison report
day3_report_path = (
    REPORT_DIR
    / "day3_tuning_comparison.csv"
)


# Save the Day 3 tuning comparison
day3_comparison.to_csv(
    day3_report_path,
    index=False
)


# Choose a location for the learning-curve results
day3_learning_curve_path = (
    REPORT_DIR
    / "day3_learning_curve.csv"
)


# Save the learning-curve values
day3_learning_curve.to_csv(
    day3_learning_curve_path,
    index=False
)


# Display the final Day 3 champion and its important information
print("\n================================")
print("DAY 3 CHAMPION")
print("================================")

print("Selected model:", day3_champion_name)

print(
    "Validation recall:",
    round(
        day3_comparison.loc[
            0,
            "Validation Recall"
        ],
        4
    )
)

print(
    "Best parameters:",
    day3_comparison.loc[
        0,
        "Best Parameters"
    ]
)

print(
    "Bias-variance diagnosis:",
    day3_comparison.loc[
        0,
        "Diagnosis"
    ]
)

print(
    "Model saved at:",
    day3_champion_path
)

print(
    "Comparison saved at:",
    day3_report_path
)

print(
    "Learning curve saved at:",
    day3_learning_curve_path
)


# Import the tools needed for feature engineering, PCA, model comparison, and XGBoost
import numpy as np

from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import FunctionTransformer

try:
    from xgboost import XGBClassifier

except ImportError:
    raise ImportError(
        "XGBoost is not installed. "
        "Run: python -m pip install xgboost"
    )


# Create useful new columns from the existing customer information
def create_day4_features(customer_data):

    # Make a copy so the original customer data is not changed
    engineered_data = customer_data.copy()

    # Use a very small number to prevent division by zero
    small_number = 0.000001

    # Compare the customer's balance with estimated salary
    engineered_data["BalanceSalaryRatio"] = (
        engineered_data["Balance"]
        / (
            engineered_data["EstimatedSalary"]
            + small_number
        )
    )

    # Compare the number of products with account tenure
    engineered_data["ProductsPerTenure"] = (
        engineered_data["NumOfProducts"]
        / (
            engineered_data["Tenure"]
            + 1
        )
    )

    # Compare the customer's age with account tenure
    engineered_data["AgeTenureRatio"] = (
        engineered_data["Age"]
        / (
            engineered_data["Tenure"]
            + 1
        )
    )

    # Compare the customer's credit score with age
    engineered_data["CreditScoreAgeRatio"] = (
        engineered_data["CreditScore"]
        / (
            engineered_data["Age"]
            + small_number
        )
    )

    # Mark customers who are active and have a positive balance
    engineered_data["ActiveWithBalance"] = (
        (
            engineered_data["IsActiveMember"] == 1
        )
        & (
            engineered_data["Balance"] > 0
        )
    ).astype(int)

    # Mark customers whose account balance is zero
    engineered_data["HasZeroBalance"] = (
        engineered_data["Balance"] == 0
    ).astype(int)

    # Mark customers who use more than one bank product
    engineered_data["HasMultipleProducts"] = (
        engineered_data["NumOfProducts"] > 1
    ).astype(int)

    # Combine age and number of products into one interaction feature
    engineered_data["AgeProductInteraction"] = (
        engineered_data["Age"]
        * engineered_data["NumOfProducts"]
    )

    # Combine balance and active membership into one feature
    engineered_data["ActiveBalanceInteraction"] = (
        engineered_data["Balance"]
        * engineered_data["IsActiveMember"]
    )

    # Divide customer ages into simple age groups
    engineered_data["AgeGroup"] = pd.cut(
        engineered_data["Age"],
        bins=[
            0,
            30,
            40,
            50,
            60,
            np.inf
        ],
        labels=[
            "Under 30",
            "30 to 39",
            "40 to 49",
            "50 to 59",
            "60 and above"
        ],
        include_lowest=True
    ).astype(str)

    # Return the data containing both original and new features
    return engineered_data


# Convert the feature-engineering function into a pipeline step
day4_feature_engineer = FunctionTransformer(
    create_day4_features,
    validate=False,
    feature_names_out="one-to-one"
)


# Apply feature engineering once only to find the new column types
day4_example_data = create_day4_features(
    X_train
)


# Find all numerical columns after feature engineering
day4_numerical_columns = (
    day4_example_data
    .select_dtypes(include=["number"])
    .columns
    .tolist()
)


# Find all categorical columns after feature engineering
day4_categorical_columns = (
    day4_example_data
    .select_dtypes(exclude=["number"])
    .columns
    .tolist()
)


# Display the number of original and engineered features
print("\n================================")
print("DAY 4 FEATURE ENGINEERING")
print("================================")

print(
    "Original number of columns:",
    X_train.shape[1]
)

print(
    "Number of columns after feature engineering:",
    day4_example_data.shape[1]
)

print("\nNumerical columns:")
print(day4_numerical_columns)

print("\nCategorical columns:")
print(day4_categorical_columns)


# Fill missing numerical values and standardize numerical columns
day4_numerical_pipeline = Pipeline(
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
day4_categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


# Process numerical and categorical engineered features separately
day4_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            day4_numerical_pipeline,
            day4_numerical_columns
        ),
        (
            "categorical",
            day4_categorical_pipeline,
            day4_categorical_columns
        )
    ],
    sparse_threshold=0
)


# Use PCA to keep 95 percent of the useful information
day4_pca = PCA(
    n_components=0.95,
    random_state=RANDOM_STATE
)


# Count stayed and exited customers in the training data
number_of_stayed_customers = (
    y_train == 0
).sum()

number_of_exited_customers = (
    y_train == 1
).sum()


# Calculate a class imbalance value for XGBoost
xgboost_class_weight = (
    number_of_stayed_customers
    / number_of_exited_customers
)


# Display the class imbalance value used by XGBoost
print(
    "\nXGBoost scale_pos_weight:",
    round(xgboost_class_weight, 4)
)


# Create a Random Forest bagging ensemble
day4_random_forest = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)


# Create an XGBoost boosting ensemble
day4_xgboost = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=2,
    subsample=0.80,
    colsample_bytree=0.80,
    scale_pos_weight=xgboost_class_weight,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1
)


# Create an engineered Random Forest pipeline without PCA
day4_random_forest_without_pca = Pipeline(
    steps=[
        (
            "feature_engineering",
            clone(day4_feature_engineer)
        ),
        (
            "preprocessor",
            clone(day4_preprocessor)
        ),
        (
            "classifier",
            clone(day4_random_forest)
        )
    ]
)


# Create an engineered Random Forest pipeline with PCA
day4_random_forest_with_pca = Pipeline(
    steps=[
        (
            "feature_engineering",
            clone(day4_feature_engineer)
        ),
        (
            "preprocessor",
            clone(day4_preprocessor)
        ),
        (
            "pca",
            clone(day4_pca)
        ),
        (
            "classifier",
            clone(day4_random_forest)
        )
    ]
)


# Create an engineered XGBoost pipeline without PCA
day4_xgboost_without_pca = Pipeline(
    steps=[
        (
            "feature_engineering",
            clone(day4_feature_engineer)
        ),
        (
            "preprocessor",
            clone(day4_preprocessor)
        ),
        (
            "classifier",
            clone(day4_xgboost)
        )
    ]
)


# Create an engineered XGBoost pipeline with PCA
day4_xgboost_with_pca = Pipeline(
    steps=[
        (
            "feature_engineering",
            clone(day4_feature_engineer)
        ),
        (
            "preprocessor",
            clone(day4_preprocessor)
        ),
        (
            "pca",
            clone(day4_pca)
        ),
        (
            "classifier",
            clone(day4_xgboost)
        )
    ]
)


# Compare Day 3 with the new Day 4 ensemble pipelines
day4_candidates = {
    "Day 3 Champion": clone(
        day3_champion_model
    ),

    "Engineered Random Forest": (
        day4_random_forest_without_pca
    ),

    "Engineered Random Forest with PCA": (
        day4_random_forest_with_pca
    ),

    "Engineered XGBoost": (
        day4_xgboost_without_pca
    ),

    "Engineered XGBoost with PCA": (
        day4_xgboost_with_pca
    )
}


# Create empty containers for fitted models and scores
day4_trained_models = {}
day4_results = []


# Compare every candidate using the same cross-validation folds
for model_name, candidate_pipeline in day4_candidates.items():

    # Display the candidate currently being evaluated
    print("\n================================")
    print("DAY 4 EVALUATING")
    print(model_name.upper())
    print("================================")

    # Calculate training and validation scores using cross-validation
    day4_cv_scores = cross_validate(
        estimator=candidate_pipeline,
        X=X_train,
        y=y_train,
        scoring={
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc"
        },
        cv=day3_cross_validation,
        n_jobs=-1,
        return_train_score=True
    )

    # Calculate the average training recall
    day4_train_recall = (
        day4_cv_scores[
            "train_recall"
        ].mean()
    )

    # Calculate the average validation accuracy
    day4_validation_accuracy = (
        day4_cv_scores[
            "test_accuracy"
        ].mean()
    )

    # Calculate the average validation precision
    day4_validation_precision = (
        day4_cv_scores[
            "test_precision"
        ].mean()
    )

    # Calculate the average validation recall
    day4_validation_recall = (
        day4_cv_scores[
            "test_recall"
        ].mean()
    )

    # Calculate the average validation F1 score
    day4_validation_f1 = (
        day4_cv_scores[
            "test_f1"
        ].mean()
    )

    # Calculate the average validation ROC-AUC
    day4_validation_roc_auc = (
        day4_cv_scores[
            "test_roc_auc"
        ].mean()
    )

    # Calculate the difference between training and validation recall
    day4_recall_gap = (
        day4_train_recall
        - day4_validation_recall
    )

    # Give a simple bias and variance diagnosis
    if (
        day4_train_recall < 0.70
        and day4_validation_recall < 0.70
    ):
        day4_diagnosis = (
            "Possible underfitting or high bias"
        )

    elif day4_recall_gap > 0.10:
        day4_diagnosis = (
            "Possible overfitting or high variance"
        )

    else:
        day4_diagnosis = (
            "Training and validation scores are balanced"
        )

    # Fit the current candidate using all training rows
    candidate_pipeline.fit(
        X_train,
        y_train
    )

    # Keep the fitted model so the winner can be saved later
    day4_trained_models[
        model_name
    ] = candidate_pipeline

    # Store all important validation results
    day4_results.append(
        {
            "Model": model_name,
            "Train Recall": day4_train_recall,
            "Validation Accuracy": (
                day4_validation_accuracy
            ),
            "Validation Precision": (
                day4_validation_precision
            ),
            "Validation Recall": (
                day4_validation_recall
            ),
            "Validation F1": (
                day4_validation_f1
            ),
            "Validation ROC-AUC": (
                day4_validation_roc_auc
            ),
            "Recall Gap": day4_recall_gap,
            "Diagnosis": day4_diagnosis
        }
    )

    # Display the current model's validation results
    print(
        f"Training recall     : "
        f"{day4_train_recall:.4f}"
    )

    print(
        f"Validation accuracy : "
        f"{day4_validation_accuracy:.4f}"
    )

    print(
        f"Validation precision: "
        f"{day4_validation_precision:.4f}"
    )

    print(
        f"Validation recall   : "
        f"{day4_validation_recall:.4f}"
    )

    print(
        f"Validation F1       : "
        f"{day4_validation_f1:.4f}"
    )

    print(
        f"Validation ROC-AUC  : "
        f"{day4_validation_roc_auc:.4f}"
    )

    print(
        f"Recall gap          : "
        f"{day4_recall_gap:.4f}"
    )

    print(
        "Diagnosis           :",
        day4_diagnosis
    )


# Convert all Day 4 cross-validation results into a table
day4_comparison = pd.DataFrame(
    day4_results
)


# Rank candidates mainly using cross-validation recall
day4_comparison = day4_comparison.sort_values(
    by=[
        "Validation Recall",
        "Validation ROC-AUC",
        "Validation F1"
    ],
    ascending=False
).reset_index(drop=True)


# Display the full Day 4 model comparison
print("\n================================")
print("DAY 4 MODEL COMPARISON")
print("================================")

print(
    day4_comparison.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)


# Select the candidate with the strongest validation performance
day4_champion_name = day4_comparison.loc[
    0,
    "Model"
]


# Get the fitted pipeline belonging to the selected champion
day4_champion_model = day4_trained_models[
    day4_champion_name
]


# Use the selected champion to predict the untouched test customers
day4_test_predictions = day4_champion_model.predict(
    X_test
)


# Get the selected champion's probability predictions
day4_test_probabilities = (
    day4_champion_model.predict_proba(
        X_test
    )[:, 1]
)


# Calculate the final champion's test accuracy
day4_test_accuracy = accuracy_score(
    y_test,
    day4_test_predictions
)


# Calculate the final champion's test precision
day4_test_precision = precision_score(
    y_test,
    day4_test_predictions,
    zero_division=0
)


# Calculate the final champion's test recall
day4_test_recall = recall_score(
    y_test,
    day4_test_predictions,
    zero_division=0
)


# Calculate the final champion's test F1 score
day4_test_f1 = f1_score(
    y_test,
    day4_test_predictions,
    zero_division=0
)


# Calculate the final champion's test ROC-AUC
day4_test_roc_auc = roc_auc_score(
    y_test,
    day4_test_probabilities
)


# Display the final test results only after selecting the champion
print("\n================================")
print("DAY 4 FINAL CHAMPION")
print("================================")

print(
    "Selected model:",
    day4_champion_name
)

print(
    "Selection basis: Highest cross-validation recall, "
    "then ROC-AUC and F1."
)

print(
    f"Test Accuracy : "
    f"{day4_test_accuracy:.4f}"
)

print(
    f"Test Precision: "
    f"{day4_test_precision:.4f}"
)

print(
    f"Test Recall   : "
    f"{day4_test_recall:.4f}"
)

print(
    f"Test F1       : "
    f"{day4_test_f1:.4f}"
)

print(
    f"Test ROC-AUC  : "
    f"{day4_test_roc_auc:.4f}"
)


# Show correct and incorrect predictions for the final model
print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        day4_test_predictions
    )
)


# Show detailed results for stayed and exited customers
print("\nClassification Report:")

print(
    classification_report(
        y_test,
        day4_test_predictions,
        target_names=[
            "Stayed",
            "Exited"
        ],
        zero_division=0
    )
)


# Create columns for final test results in the comparison table
day4_comparison[
    "Final Test Accuracy"
] = np.nan

day4_comparison[
    "Final Test Precision"
] = np.nan

day4_comparison[
    "Final Test Recall"
] = np.nan

day4_comparison[
    "Final Test F1"
] = np.nan

day4_comparison[
    "Final Test ROC-AUC"
] = np.nan


# Find the row belonging to the selected model
day4_champion_row = (
    day4_comparison["Model"]
    == day4_champion_name
)


# Add final test results only to the selected champion's row
day4_comparison.loc[
    day4_champion_row,
    "Final Test Accuracy"
] = day4_test_accuracy

day4_comparison.loc[
    day4_champion_row,
    "Final Test Precision"
] = day4_test_precision

day4_comparison.loc[
    day4_champion_row,
    "Final Test Recall"
] = day4_test_recall

day4_comparison.loc[
    day4_champion_row,
    "Final Test F1"
] = day4_test_f1

day4_comparison.loc[
    day4_champion_row,
    "Final Test ROC-AUC"
] = day4_test_roc_auc


# Choose a separate model filename so earlier models are not overwritten
day4_model_path = (
    MODEL_DIR
    / "day4_production_churn_pipeline.joblib"
)


# Save the complete production pipeline
joblib.dump(
    day4_champion_model,
    day4_model_path
)


# Choose the path for the Day 4 comparison report
day4_report_path = (
    REPORT_DIR
    / "day4_model_comparison.csv"
)


# Save the full Day 4 comparison table
day4_comparison.to_csv(
    day4_report_path,
    index=False
)


# Create a report explaining every engineered feature
day4_feature_report = pd.DataFrame(
    {
        "Engineered Feature": [
            "BalanceSalaryRatio",
            "ProductsPerTenure",
            "AgeTenureRatio",
            "CreditScoreAgeRatio",
            "ActiveWithBalance",
            "HasZeroBalance",
            "HasMultipleProducts",
            "AgeProductInteraction",
            "ActiveBalanceInteraction",
            "AgeGroup"
        ],

        "Meaning": [
            "Balance divided by estimated salary",
            "Products divided by tenure plus one",
            "Age divided by tenure plus one",
            "Credit score divided by age",
            "Customer is active and has a positive balance",
            "Customer has a zero balance",
            "Customer uses more than one product",
            "Age multiplied by product count",
            "Balance multiplied by active membership",
            "Customer age category"
        ]
    }
)


# Choose the path for the engineered-feature report
day4_feature_report_path = (
    REPORT_DIR
    / "day4_engineered_features.csv"
)


# Save the engineered-feature explanations
day4_feature_report.to_csv(
    day4_feature_report_path,
    index=False
)


# Load the saved production model to verify the saved file
day4_loaded_model = joblib.load(
    day4_model_path
)


# Make predictions using the loaded model
day4_loaded_predictions = (
    day4_loaded_model.predict(
        X_test
    )
)


# Confirm that saving and loading did not change predictions
day4_predictions_match = np.array_equal(
    day4_test_predictions,
    day4_loaded_predictions
)


# Display all saved Day 4 files
print("\n================================")
print("DAY 4 SAVED FILES")
print("================================")

print(
    "Production model:",
    day4_model_path
)

print(
    "Comparison report:",
    day4_report_path
)

print(
    "Engineered-feature report:",
    day4_feature_report_path
)

print(
    "Predictions identical after reload:",
    day4_predictions_match
)


# Import TensorFlow and Keras tools needed to build and train the neural network
try:
    import tensorflow as tf

    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Dense, Dropout, Input
    from tensorflow.keras.models import Sequential

except ImportError:
    raise ImportError(
        "TensorFlow is not installed. "
        "Run: python -m pip install tensorflow"
    )


# Set random values so the ANN results are more reproducible
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# Display the installed TensorFlow version
print("\n================================")
print("DAY 5: KERAS ANN")
print("================================")

print("TensorFlow version:", tf.__version__)


# Split the existing training data into ANN training and validation parts
X_ann_train, X_ann_validation, y_ann_train, y_ann_validation = (
    train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_train
    )
)


# Create a fresh copy of the Day 1 preprocessing pipeline for the ANN
day5_preprocessor = clone(
    preprocessor
)


# Learn preprocessing values using only the ANN training data
X_ann_train_processed = (
    day5_preprocessor.fit_transform(
        X_ann_train
    )
)


# Apply the learned preprocessing to the ANN validation data
X_ann_validation_processed = (
    day5_preprocessor.transform(
        X_ann_validation
    )
)


# Apply the same learned preprocessing to the untouched test data
X_ann_test_processed = (
    day5_preprocessor.transform(
        X_test
    )
)


# Convert sparse matrices into normal arrays when necessary
if hasattr(
    X_ann_train_processed,
    "toarray"
):
    X_ann_train_processed = (
        X_ann_train_processed.toarray()
    )


# Convert the validation data into a normal array when necessary
if hasattr(
    X_ann_validation_processed,
    "toarray"
):
    X_ann_validation_processed = (
        X_ann_validation_processed.toarray()
    )


# Convert the test data into a normal array when necessary
if hasattr(
    X_ann_test_processed,
    "toarray"
):
    X_ann_test_processed = (
        X_ann_test_processed.toarray()
    )


# Convert all ANN inputs to float32 because TensorFlow works efficiently with this type
X_ann_train_processed = (
    X_ann_train_processed.astype(
        "float32"
    )
)

X_ann_validation_processed = (
    X_ann_validation_processed.astype(
        "float32"
    )
)

X_ann_test_processed = (
    X_ann_test_processed.astype(
        "float32"
    )
)


# Convert the answers into float32 values for TensorFlow
y_ann_train_array = (
    y_ann_train.to_numpy().astype(
        "float32"
    )
)

y_ann_validation_array = (
    y_ann_validation.to_numpy().astype(
        "float32"
    )
)

y_ann_test_array = (
    y_test.to_numpy().astype(
        "float32"
    )
)


# Count how many processed input features enter the neural network
number_of_ann_features = (
    X_ann_train_processed.shape[1]
)


# Display the ANN data shapes
print(
    "\nANN training shape:",
    X_ann_train_processed.shape
)

print(
    "ANN validation shape:",
    X_ann_validation_processed.shape
)

print(
    "ANN testing shape:",
    X_ann_test_processed.shape
)

print(
    "Number of ANN input features:",
    number_of_ann_features
)


# Build a small feed-forward neural network for customer churn
day5_ann_model = Sequential(
    [
        # Accept one processed value for every input feature
        Input(
            shape=(
                number_of_ann_features,
            )
        ),

        # Learn the first set of customer patterns using 64 neurons
        Dense(
            units=64,
            activation="relu"
        ),

        # Randomly disable some neurons during training to reduce overfitting
        Dropout(
            rate=0.30
        ),

        # Learn deeper patterns using 32 neurons
        Dense(
            units=32,
            activation="relu"
        ),

        # Apply another smaller dropout step
        Dropout(
            rate=0.20
        ),

        # Learn a smaller final hidden representation
        Dense(
            units=16,
            activation="relu"
        ),

        # Return one probability between 0 and 1 for customer exit
        Dense(
            units=1,
            activation="sigmoid"
        )
    ]
)


# Configure how the ANN learns and which metrics it records
day5_ann_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(
            name="precision"
        ),
        tf.keras.metrics.Recall(
            name="recall"
        ),
        tf.keras.metrics.AUC(
            name="auc"
        )
    ]
)


# Display the neural network layers and parameter counts
print("\nANN model structure:")

day5_ann_model.summary()


# Calculate a larger weight for exited customers because the target is imbalanced
day5_stayed_count = int(
    (
        y_ann_train == 0
    ).sum()
)

day5_exited_count = int(
    (
        y_ann_train == 1
    ).sum()
)


# Create balanced class weights for ANN training
day5_total_count = (
    day5_stayed_count
    + day5_exited_count
)

day5_class_weights = {
    0: (
        day5_total_count
        / (
            2
            * day5_stayed_count
        )
    ),

    1: (
        day5_total_count
        / (
            2
            * day5_exited_count
        )
    )
}


# Display the importance given to both target classes
print(
    "\nANN class weights:",
    day5_class_weights
)


# Stop training when validation loss stops improving
day5_early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    min_delta=0.001,
    restore_best_weights=True,
    verbose=1
)


# Train the ANN while checking validation loss after every epoch
day5_history = day5_ann_model.fit(
    X_ann_train_processed,
    y_ann_train_array,
    validation_data=(
        X_ann_validation_processed,
        y_ann_validation_array
    ),
    epochs=100,
    batch_size=32,
    class_weight=day5_class_weights,
    callbacks=[
        day5_early_stopping
    ],
    verbose=1
)


# Count how many epochs were completed before early stopping
day5_epochs_completed = len(
    day5_history.history[
        "loss"
    ]
)


# Find the epoch with the smallest validation loss
day5_best_epoch = (
    int(
        np.argmin(
            day5_history.history[
                "val_loss"
            ]
        )
    )
    + 1
)


# Display the early-stopping results
print("\n================================")
print("DAY 5 TRAINING SUMMARY")
print("================================")

print(
    "Epochs completed:",
    day5_epochs_completed
)

print(
    "Best epoch:",
    day5_best_epoch
)

print(
    "Best validation loss:",
    round(
        min(
            day5_history.history[
                "val_loss"
            ]
        ),
        4
    )
)


# Predict the exit probability for every untouched test customer
day5_ann_probabilities = (
    day5_ann_model.predict(
        X_ann_test_processed,
        verbose=0
    )
    .reshape(-1)
)


# Convert probabilities of at least 0.5 into class 1 predictions
day5_ann_predictions = (
    day5_ann_probabilities >= 0.50
).astype(int)


# Calculate the ANN test accuracy
day5_ann_accuracy = accuracy_score(
    y_test,
    day5_ann_predictions
)


# Calculate the ANN test precision
day5_ann_precision = precision_score(
    y_test,
    day5_ann_predictions,
    zero_division=0
)


# Calculate the ANN test recall
day5_ann_recall = recall_score(
    y_test,
    day5_ann_predictions,
    zero_division=0
)


# Calculate the ANN test F1 score
day5_ann_f1 = f1_score(
    y_test,
    day5_ann_predictions,
    zero_division=0
)


# Calculate the ANN test ROC-AUC using probability predictions
day5_ann_roc_auc = roc_auc_score(
    y_test,
    day5_ann_probabilities
)


# Display the ANN test results
print("\n================================")
print("DAY 5 ANN TEST RESULTS")
print("================================")

print(
    f"Accuracy : "
    f"{day5_ann_accuracy:.4f}"
)

print(
    f"Precision: "
    f"{day5_ann_precision:.4f}"
)

print(
    f"Recall   : "
    f"{day5_ann_recall:.4f}"
)

print(
    f"F1 Score : "
    f"{day5_ann_f1:.4f}"
)

print(
    f"ROC-AUC  : "
    f"{day5_ann_roc_auc:.4f}"
)


# Show the ANN correct and incorrect prediction counts
print("\nANN Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        day5_ann_predictions
    )
)


# Show detailed ANN results for stayed and exited customers
print("\nANN Classification Report:")

print(
    classification_report(
        y_test,
        day5_ann_predictions,
        target_names=[
            "Stayed",
            "Exited"
        ],
        zero_division=0
    )
)


# Use the Day 4 classical champion to predict the same test customers
day5_classical_predictions = (
    day4_champion_model.predict(
        X_test
    )
)


# Get probability predictions from the Day 4 classical champion
day5_classical_probabilities = (
    day4_champion_model.predict_proba(
        X_test
    )[:, 1]
)


# Calculate the classical champion's accuracy
day5_classical_accuracy = accuracy_score(
    y_test,
    day5_classical_predictions
)


# Calculate the classical champion's precision
day5_classical_precision = precision_score(
    y_test,
    day5_classical_predictions,
    zero_division=0
)


# Calculate the classical champion's recall
day5_classical_recall = recall_score(
    y_test,
    day5_classical_predictions,
    zero_division=0
)


# Calculate the classical champion's F1 score
day5_classical_f1 = f1_score(
    y_test,
    day5_classical_predictions,
    zero_division=0
)


# Calculate the classical champion's ROC-AUC
day5_classical_roc_auc = roc_auc_score(
    y_test,
    day5_classical_probabilities
)


# Put the ANN and classical champion results into one comparison table
day5_comparison = pd.DataFrame(
    [
        {
            "Model": (
                "Phase 1 Classical Champion: "
                + day4_champion_name
            ),
            "Accuracy": (
                day5_classical_accuracy
            ),
            "Precision": (
                day5_classical_precision
            ),
            "Recall": (
                day5_classical_recall
            ),
            "F1 Score": (
                day5_classical_f1
            ),
            "ROC-AUC": (
                day5_classical_roc_auc
            )
        },

        {
            "Model": "Day 5 Keras ANN",
            "Accuracy": (
                day5_ann_accuracy
            ),
            "Precision": (
                day5_ann_precision
            ),
            "Recall": (
                day5_ann_recall
            ),
            "F1 Score": (
                day5_ann_f1
            ),
            "ROC-AUC": (
                day5_ann_roc_auc
            )
        }
    ]
)


# Sort the benchmark mainly by recall and then by ROC-AUC
day5_comparison = day5_comparison.sort_values(
    by=[
        "Recall",
        "ROC-AUC",
        "F1 Score"
    ],
    ascending=False
).reset_index(drop=True)


# Display the head-to-head benchmark
print("\n================================")
print("DAY 5 ANN VS CLASSICAL CHAMPION")
print("================================")

print(
    day5_comparison.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)


# Check whether the ANN improved recall compared with the classical champion
day5_recall_difference = (
    day5_ann_recall
    - day5_classical_recall
)


# Give an honest conclusion about whether the ANN helped
if day5_recall_difference > 0.01:

    day5_conclusion = (
        "The ANN improved recall by "
        f"{day5_recall_difference:.4f}. "
        "The ANN found more customers who exited."
    )

elif day5_recall_difference < -0.01:

    day5_conclusion = (
        "The classical champion had better recall by "
        f"{abs(day5_recall_difference):.4f}. "
        "The ANN did not improve churn detection."
    )

else:

    day5_conclusion = (
        "The ANN and classical champion had similar recall. "
        "The simpler classical model may be preferable for tabular data."
    )


# Display the final Day 5 conclusion
print("\nDay 5 conclusion:")
print(day5_conclusion)


# Create a table containing the loss and metrics from every training epoch
day5_training_history = pd.DataFrame(
    day5_history.history
)


# Add an easy-to-read epoch number
day5_training_history.insert(
    0,
    "Epoch",
    range(
        1,
        len(day5_training_history) + 1
    )
)


# Choose paths for the ANN, preprocessor, reports, and metadata
day5_ann_model_path = (
    MODEL_DIR
    / "day5_ann_churn_model.keras"
)

day5_preprocessor_path = (
    MODEL_DIR
    / "day5_ann_preprocessor.joblib"
)

day5_comparison_path = (
    REPORT_DIR
    / "day5_ann_vs_classical.csv"
)

day5_history_path = (
    REPORT_DIR
    / "day5_ann_training_history.csv"
)

day5_summary_path = (
    REPORT_DIR
    / "day5_ann_summary.csv"
)


# Save the trained Keras ANN using the Keras model format
day5_ann_model.save(
    day5_ann_model_path
)


# Save the fitted preprocessing pipeline used by the ANN
joblib.dump(
    day5_preprocessor,
    day5_preprocessor_path
)


# Save the ANN versus classical comparison
day5_comparison.to_csv(
    day5_comparison_path,
    index=False
)


# Save the training and validation scores from every epoch
day5_training_history.to_csv(
    day5_history_path,
    index=False
)


# Save important ANN information in a small summary file
day5_summary = pd.DataFrame(
    [
        {
            "ANN Input Features": (
                number_of_ann_features
            ),
            "Epochs Completed": (
                day5_epochs_completed
            ),
            "Best Epoch": (
                day5_best_epoch
            ),
            "Decision Threshold": 0.50,
            "ANN Recall": (
                day5_ann_recall
            ),
            "Classical Recall": (
                day5_classical_recall
            ),
            "Recall Difference": (
                day5_recall_difference
            ),
            "Conclusion": (
                day5_conclusion
            )
        }
    ]
)

day5_summary.to_csv(
    day5_summary_path,
    index=False
)


# Reload the saved ANN to make sure the saved model works
day5_loaded_ann = tf.keras.models.load_model(
    day5_ann_model_path
)


# Reload the saved preprocessing pipeline
day5_loaded_preprocessor = joblib.load(
    day5_preprocessor_path
)


# Select five customers for checking the reloaded ANN
day5_sample_customers = X_test.head(
    5
)


# Transform the sample customers using the reloaded preprocessing
day5_sample_processed = (
    day5_loaded_preprocessor.transform(
        day5_sample_customers
    )
)


# Convert sparse sample data into a normal array when necessary
if hasattr(
    day5_sample_processed,
    "toarray"
):
    day5_sample_processed = (
        day5_sample_processed.toarray()
    )


# Convert sample input values into TensorFlow's preferred type
day5_sample_processed = (
    day5_sample_processed.astype(
        "float32"
    )
)


# Predict exit probabilities using the reloaded ANN
day5_sample_probabilities = (
    day5_loaded_ann.predict(
        day5_sample_processed,
        verbose=0
    )
    .reshape(-1)
)


# Convert sample probabilities into 0 or 1 predictions
day5_sample_predictions = (
    day5_sample_probabilities >= 0.50
).astype(int)


# Create an easy-to-read sample prediction table
day5_sample_output = (
    day5_sample_customers.copy()
)

day5_sample_output[
    "Predicted Exited"
] = day5_sample_predictions

day5_sample_output[
    "Exit Probability"
] = day5_sample_probabilities


# Display the reloaded ANN predictions
print("\n================================")
print("RELOADED ANN SAMPLE PREDICTIONS")
print("================================")

print(
    day5_sample_output.to_string(
        index=False
    )
)


# Display all files created during Day 5
print("\n================================")
print("DAY 5 SAVED FILES")
print("================================")

print(
    "Keras ANN model:",
    day5_ann_model_path
)

print(
    "ANN preprocessor:",
    day5_preprocessor_path
)

print(
    "Benchmark report:",
    day5_comparison_path
)

print(
    "Training history:",
    day5_history_path
)

print(
    "ANN summary:",
    day5_summary_path
)