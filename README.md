# Diabetic Prediction Using SVM

## Overview

This project is a machine learning based diabetic prediction system that uses the Support Vector Machine SVM algorithm to predict whether a person is likely to have diabetes based on medical attributes.

The model is trained on patient health data such as glucose level BMI age insulin blood pressure skin thickness and number of pregnancies.

## Features

* Data preprocessing and cleaning
* Exploratory data analysis
* Feature scaling using StandardScaler
* Training using Support Vector Machine SVM
* Model evaluation using accuracy score
* Prediction system for new patient data
* Simple and user friendly interface

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit learn
* Matplotlib
* Streamlit

## Dataset Features

The dataset contains the following input features:

* Pregnancies
* Glucose
* Blood Pressure
* Skin Thickness
* Insulin
* BMI
* Diabetes Pedigree Function
* Age

Output:

* 0 = Non Diabetic
* 1 = Diabetic

## Project Workflow

1. Load the dataset
2. Perform data preprocessing
3. Split the dataset into training and testing data
4. Scale the feature values
5. Train the SVM model
6. Evaluate the model performance
7. Save the trained model
8. Build a prediction system using Streamlit

## Installation

```bash
pip install pandas numpy scikit-learn matplotlib streamlit
```

## Run the Project

```bash
streamlit run app.py
```

## Future Improvements

* Add more advanced ML models
* Improve frontend design
* Deploy the project online
* Add patient report generation
* Use larger healthcare datasets

## Author

Niladri Mondal
