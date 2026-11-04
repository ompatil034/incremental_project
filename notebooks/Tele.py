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