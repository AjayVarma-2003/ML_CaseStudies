################################################################
# Required pacakages
################################################################
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

################################################################
# Artifacts
################################################################
FILENAME = "breast-cancer-wisconsin.csv"
ARTIFACT = Path("artifact-sample")
ARTIFACT.mkdir(exist_ok = True)
MODELPATH = ARTIFACT / "BreastCancerPredictorLogistic.joblib"
RANDOM_STATE = 42
TEST_SIZE = 0.2

################################################################
# Function Name :- read_data()
# Description :-   Read the data into pandas dataframe
# Input :-         path to csv file
# Output :-        dataframe
# Author :-        Ajay Yogesh Varma
# Date :-          25-08-2025
################################################################
def read_data(path):
    """
        Read the data into pandas dataframe
    """
    df = pd.read_csv(path)
    return df

################################################################
# Function Name :- clean_data()
# Description :-   Used to clean or handle missing values if any
#                  and removes unnecessary columns.
# Input :-         dataset
# Output :-        Cleaned dataset
# Author :-        Ajay Yogesh Varma
# Date :-          25-08-2025
################################################################
def clean_data(df):
    """
        Checks and Cleans or handles missing values if there are any
    """
    print("Number of missing values in dataset are : ")
    print(df.isnull().sum())
    # No missing values in dataset
    
    df = df.drop(columns = ["CodeNumber"], axis = 1)

    # handling ? data
    df['BareNuclei'] = df['BareNuclei'].replace('?', np.nan)
    df = df.replace(np.nan, 0)

    return df

################################################################
# Function Name :- split_dataset()
# Description :-   splits dataset into 4 parts of training and testing
# Input :-         dataset, test size, random state
# Output :-        4 splitted parts
# Author :-        Ajay Yogesh Varma
# Date :-          25-08-2025
################################################################
def split_dataset(features, label, TestSize, RandomState):
    """
        Splits dataset into 4 parts.
    """
    X_train, X_test, Y_train, Y_test = train_test_split(features, label, test_size = TestSize, random_state = RandomState)

    return X_train, X_test, Y_train, Y_test

################################################################
# Function Name :- build_pipeline()
# Description :-   Creates pipeline which performs operations given in it.
# Input :-         X_train
# Output :-        pipeline
# Author :-        Ajay Yogesh Varma
# Date :-          25-08-2025
################################################################
def build_pipeline():
    """
        Creates pipeline with two tasks.
    """
    pipe = Pipeline(steps = [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter = 1000),)
        ]
    )

    return pipe

################################################################
# Function Name :- train_pipeline()
# Description :-   Trains a pipeline
# Input :-         pipeline, X_train, Y_train
# Output :-        trained pipeline / model
# Author :-        Ajay Yogesh Varma
# Date :-          25-08-2025
################################################################
def train_pipeline(pipe, x_train, y_train):
    """
        Trains the pipeline / model
    """
    pipe.fit(x_train, y_train)

    return pipe

################################################################
# Function Name :- prediction()
# Description :-   gives predictions of test data
# Input :-         pipeline, X_test
# Output :-        prediction
# Author :-        Ajay Yogesh Varma
# Date :-          25-08-2025
################################################################
def prediction(pipeline, X_test):
    """
        Predicts on the testing data
    """
    predictions = pipeline.predict(X_test)

    return predictions

################################################################
# Function Name :- calculate_accuracy()
# Description :-   Calculates the accuracy of model.
# Input :-         Y_test, Y_predicted
# Output :-        accuracy score of model
# Author :-        Ajay Yogesh Varma
# Date :-          25-08-2025
################################################################
def calculate_accuracy(Y_test, Y_predicted):
    """
        Calculated the accuracy of model using Y_test and Y_predicted
    """
    accuracy = (accuracy_score(Y_test, Y_predicted)) * 100

    return accuracy

################################################################
# Function Name :- save_model()
# Description :-   saves the current ml model.
# Input :-         model and path
# Output :-        None
# Author :-        Ajay Yogesh Varma
# Date :-          25-08-2025
################################################################
def save_model(model, path = MODELPATH):
    """
        Saves the current model.
        Due to this we can get this trained model and test directly whenever needed
    """
    joblib.dump(model, path)
    
    print(f"Model saved at path : {path}")

################################################################
# Functio Name :- load_model()
# Description :-  loads the saved ml model
# Input :-        path
# Output :-       saved model
# Author :-       Ajay Yogesh Varma
# Date :-         25-08-2025
################################################################
def load_model(path = MODELPATH):
    """
        Loads the saved ml model
    """

    model = joblib.load(path)

    print(f"Model loaded from path : {path}")

    return model

################################################################
# Fuction Name :- main()
# Description :-  Main function from where execution of program starts.
# Author :-       Ajay Yogesh Varma
# Date :-         25-08-2025
################################################################
def main():
    # read data
    dataset = pd.read_csv(FILENAME)
    print("First 5 entries of dataset is : ")
    print(dataset.head())

    # Clean data
    dataset = clean_data(dataset)
    print("Dataset after cleaning : ")
    print(dataset.head())

    # Split dataset into train and test part
    x = dataset.drop(columns = ["CancerType"], axis = 1)
    y = dataset["CancerType"]
    X_train, X_test, Y_train, Y_test = split_dataset(x, y, TEST_SIZE, RANDOM_STATE)

    print("Dimension of X_train is : ", X_train.shape)
    print("Dimension of X_test is : ", X_test.shape)
    print("Dimension of Y_train is : ", Y_train.shape)
    print("Dimension of Y_test is : ", Y_test.shape)

    # Build and train pipeline
    pipe = build_pipeline()
    model = train_pipeline(pipe, X_train, Y_train)
    print("Trained model is : ", model)

    Y_predicted = prediction(model, X_test)

    # Calculate accuracy score
    accuracy = calculate_accuracy(Y_test, Y_predicted)
    print("The accuracy of model is : ", accuracy)

    # Saving model
    save_model(model, MODELPATH)

    # Load Model
    # model = load_model(MODELPATH)
    # Y_predicted = model.predict(Y_test)

################################################################
# Application Starter
################################################################
if __name__ == "__main__":
    main()