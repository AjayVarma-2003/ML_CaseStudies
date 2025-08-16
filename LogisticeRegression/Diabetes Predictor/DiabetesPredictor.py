##################################################################################
# Required Python packages
#
##################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path

##################################################################################
# File Paths and artifacts
#
##################################################################################

INPUTPATH = "diabetes.data"
FILENAME = "diabetes.csv"
ARTIFACT = Path("Artifact-samples")
ARTIFACT.mkdir(exist_ok = True)
MODELPATH = ARTIFACT / "DiabetesPredictorLogistic.joblib"
RANDOM_STATE = 42
TEST_SIZE = 0.2

##################################################################################
# Headers of dataset
#
##################################################################################

HEADERS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedegreeFunction", "Age", "Outcome"
]

##################################################################################
# Function Name :- read_data()
# Description :-   Read the data into pandas dataframe
# Input :-         path of CSV file
# Output :-        Gives the data
# Author :-        Ajay Yogesh Varma
# Data :-          10-08-2025
##################################################################################

def read_data(path):
    """
        Read the data into pandas dataframe
    """
    data = pd.read_csv(path, header = None)
    return data

##################################################################################
# Function Name :- get_headers()
# Description :-   Dataset headers
# Input :-         Dataset
# Output :-        Returns the header
# Author :-        Ajay Yogesh Varma
# Date :-          10-08-2025
##################################################################################

def get_headers(dataset):
    """
        It returns the dataset headers.
    """
    return dataset.columns.values

##################################################################################
# Functin Name :- add_headers()
# Description :-  Add the headers to the dataset
# Input :-        Dataset and headers
# Output :-       Updated Dataset
# Author :-       A-jay Yogesh Varma
# Date :-         10-08-2025
##################################################################################

def add_headers(dataset, headers):
    """
        Add headers to dataset.
    """
    dataset.columns = headers
    return dataset

##################################################################################
# Function Name :- data_file_to_csv()
# Input :-         Nothing
# Output :-        Write the data to csv
# Author :-        Ajay Yogesh Varma
# Date :-          10-08-2025
##################################################################################

def data_file_to_csv():
    """
        Convert raw .data file into CSV with headers.
        This function and read_data function, get_headers, add_headers function only 
        used when you have to convert .data file into .csv file.
        Otherwise these not be used.
        Call of this function is called in main. Use if needed.
    """
    dataset = read_data(INPUTPATH)
    dataset = add_headers(dataset, HEADERS)
    dataset.to_csv(FILENAME, index = False)
    print("File Saved ...")

##################################################################################
# Function Name :- handel_missing_values
# Description :-   Filter missing values from datset
# Input :-         Datset with missing values
# Output :-        Dataset with removed missing values rows
# Author :-        Ajay Yogesh Varma
# Data :-          10-08-2025
##################################################################################

def handle_missing_values(df, feature_headers):
    """
        Handle missing values if any.
        In this case study there is no missing value.
        Hence when needed then use this function.
    """

    # Replace missing value in whole dataframe
    df = df.replace("", np.nan)    # fill blank space with type of missing values either may be ? or null or nill etc.
    
    # Cast features to numeric
    df[feature_headers] = df[feature_headers].apply(pd.to_numeric, errors = 'coerce')

    return df

##################################################################################
# Function Name :- dataset_statistics()
# Description :-   Displays the statistics of dataset
# Input :-         Dataset
# Output :-        None
# Author :-        Ajay Yogesh Varma
# Date :-          10-08-2025
##################################################################################

def dataset_statistics(df):
    """
        Print basic statistical summary of datset
    """
    print("Statistical summary of dataset is : ")
    print(df.describe())

##################################################################################
# Function Name :- split_dataset()
# Description :- splits dataset using testing percentage
# Input :-       Dataset features, target, test percentage, random state
# Output :-      Dataset after splitting
# Author :-      Ajay Yogesh Varma
# Date :-        10-08-2025
##################################################################################

def split_dataset(x, y, test_percent, ran_state):
    """
        Splits the dataset into training and testing parts.
    """
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = test_percent, random_state = ran_state)

    return X_train, X_test, Y_train, Y_test

##################################################################################
# Function Name :- build_pipeline()
# Description :-   Build a pipeline
# Input :-         None
# Output :-        pipe
# Author :-        Ajay Yogesh Varma
# Date :-          10-08-2025
##################################################################################

def build_pipline(x):
    """
        Build pipline to assign tasks.
        We are not using Simpleimputer as there are no missing values in this dataset.
        Use Simpleimputer to handle missing values in datset whenever required.
    """
    
    pipe = Pipeline(steps = [
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter = 1000)),
    ])

    return pipe

##################################################################################
# Function Name :- train_pipeline()
# Description :-   Train a pipeline
# Input :-         pipeline, X_train, Y_train
# Output :-        trained pipeline/model
# Author :-        Ajay Yogesh Varma
# Date :-          10-08-2025
##################################################################################

def train_pipeline(pipeline, X_train, Y_train):
    """
        Trains the pipeline or model
    """

    pipeline.fit(X_train, Y_train)

    return pipeline

##################################################################################
# Function Name :- predict()
# Descriptions :-  Gives the predictions of testing data
# Input :-         trained model and test data
# Output :-        predictions
# Author :-        Ajay Yogesh Varma
# Data :-          10-08-2025
##################################################################################

def prediction(model, X_test):
    """
        It predicts the output for the test data.
    """

    predictions = model.predict(X_test)

    return predictions

##################################################################################
# Function Name :- train_accuracy()
# Descriptions :-  Returns the accuracy of model while training
# Input :-         trained model, X_train , Y_train
# Output :-        training accuracy
# Author :-        Ajay Yogesh Varma
# Date :-          10-08-2025
##################################################################################

def train_accuracy(model, X_train, Y_train):
    """
        It returns the training accuracy of model.
    """

    # predictions of X_train
    prediction = model.predict(X_train)

    acc = (accuracy_score(Y_train, prediction)) * 100

    return acc

##################################################################################
# Function Name :- test_accuracy()
# Description :    Returns the accuracy of model while testing
# Input :-         Y-predicted, Y-test
# Output :-        testing accuracy score
# Author :-        Ajay Yogesh Varma
# Date :-          10-08-2025
##################################################################################

def test_accuracy(Y_test, Y_predicted):
    """
        Returns the testing accuracy of model.
    """

    acc = (accuracy_score(Y_test, Y_predicted)) * 100

    return acc

##################################################################################
# Fucntion Name :- classification_data()
# Descripiton :-   Returns the classification report of model
# Input :-         Y_test, Y_predicted
# Output :-        classification report
# Author :-        Ajay Yogesh Varma
# Date :-          10-2-08-2025
##################################################################################

def classification_data(Y_test, Y_predicted):
    """
        This returns the classification report of the model.
    """

    report = classification_report(Y_test, Y_predicted)

    return report

##################################################################################
# Function Name :- GetConfusionMatrix()
# Description :-   Gives the confusion matrix of model
# Input :-         Y_test, Y_predicted
# Output :-        confusion matrix
# Author :-        Ajay Yogesh Varma
# Date :-          10-08-2025
##################################################################################

def GetConfusionMatrix(Y_test, Y_predicted):
    """
        Returns the confusion matrix of the model.
    """

    conf_mat = confusion_matrix(Y_test, Y_predicted)

    return conf_mat

##################################################################################
# Function Name :- plot_confusion_matirx
# Description :-   Display confusion matrix
# Input :-         Y_test, Y_predicted, title
# Output :-        None
# Author :-        Ajay Yogesh Varma
# Date :-          10-08-2025
##################################################################################

def plot_confusion_matrix(Y_test, Y_predicted, title = "Confusion Matrix"):
    """
        Plot the confusion matrix
    """

    cm = confusion_matrix(Y_test, Y_predicted)
    fig, ax = plt.subplots()
    cax = ax.matshow(cm)
    fig.colorbar(cax)

    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha = 'center', va = 'center')

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

##################################################################################
# Function Name :- plot_feature_importance
# Description :-   Display feature importance
# Input :-         trained model, feature names, title
# Output :-        None
# Author :-        Ajay Yogesh Varma
# Date :-          10-08-2025
##################################################################################

def plot_feature_importance(model, feature_name, title = "Feature Importance(Logistic Regression)"):
    """
        Ploting feature importance graph.
    """
    importances = []

    if hasattr(model, "named_steps") and "rf" in model.named_steps:
        rf = model.named_steps['rf']
        importances = rf.feature_importances_
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        print("Feature importances not availabel for this model.")

    idx = np.argsort(importances)[:: -1]
    plt.figure(figsize = (8, 4))
    plt.bar(range(len(importances)), importances[idx])
    plt.xticks(range(len(importances)), [feature_name[i] for i in idx], rotation = 45, ha = 'right')
    plt.ylabel("Importance")
    plt.title(title)
    plt.tight_layout()
    plt.show()

##################################################################################
# Function Name :- save_model()
# Description :-   save the model
# Input :-         model and path
# Output :-        None
# Author :-        Ajay Yogesh Varma
# Date :-          10-08-2025
##################################################################################

def save_model(model, path = MODELPATH):
    """
        Saves the model which is trained and tested
    """
    joblib.dump(model, path)
    
    print(f"Model saved at {path}")

##################################################################################
# Function Name :- load_model()
# Description :-   load the model which is trained and saved
# Input :-         path
# Output :-        saved model
# Author :-        Ajay Yogesh Varma
# Date :-          10-08-2025
##################################################################################

def load_model(path = MODELPATH):
    """
        It will load the model which is saved
    """

    model = joblib.load(path)
    
    print(f"Model loaded from {path}")

    return model

##################################################################################
# Function Name :- main()
# Description :-   Main function from where execution starts
# Author :-        Ajay Yogesh Varma
# Date :-          10-08-2025
##################################################################################

def main():
    # 1. Ensure CSV exists (run if needed)
    # data_file_to_csv()

    # 2. Load CSV
    dataset = pd.read_csv(FILENAME)

    # 3. Basic stats
    dataset_statistics(dataset)

    # 4. Prepare features and target
    dataset.drop(columns = 'Age', axis = 1)    # removing unnecessary data column
    features = dataset.drop(columns = 'Outcome', axis = 1)
    target = dataset['Outcome']

    # 5. Handel missing values
    # dataset = handle_missing_values(dataset, features)

    # 6. Split dataset into training and testing parts
    X_train, X_test, Y_train, Y_test = split_dataset(features, target, TEST_SIZE, RANDOM_STATE)

    print("X_train shape : ", X_train.shape)
    print("X_test shape : ", X_test.shape)
    print("Y_train shape : ", Y_train.shape)
    print("Y_test shape : ", Y_test.shape)

    # 7. Build and train pipeline
    pipeline = build_pipline(features)
    trained_model = train_pipeline(pipeline, X_train, Y_train)
    print("Trained pipeline : : ", trained_model)

    # 8. Predictions
    Y_predicted = prediction(trained_model, X_test)

    # 9. Metrics
    training_acc = train_accuracy(trained_model, X_train, Y_train)
    testing_acc = test_accuracy(Y_test, Y_predicted)
    classification = classification_data(Y_test, Y_predicted)
    conf_mat = GetConfusionMatrix(Y_test, Y_predicted)
    
    print("Training accuracy of model is : ", training_acc)
    print("Testing accuracy of model is : ", testing_acc)
    print("Classification report of model is : ")
    print(classification)
    print("Confusion matrix of model is : ")
    print(conf_mat)

    # Plot confusion matrix
    plot_confusion_matrix(Y_test, Y_predicted, title = "Confusion matrix")

    # Check Feature importance
    # plot_feature_importance(trained_model, features, title = "Feature Importance (Logistic Regression)")

    # 10. Save Model(pipline) using joblib
    save_model(trained_model, MODELPATH)

    # 11. Load model and test a sample
    loaded = load_model(MODELPATH)
    sample = X_train.iloc[[0]]
    pred_loaded = loaded.predict(sample)
    print(f"Loaded model prediction for the first test sample : {pred_loaded[0]}")

##################################################################################
# Application starter
#
##################################################################################

if __name__ == "__main__":
    main()