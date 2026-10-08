# Student Performance Prediction

## Project Overview

This project uses Machine Learning to predict a student's final Mathematics grade.

The project uses the UCI Student Performance dataset and a Linear Regression model.

## Features Used

The model uses the following information:

- Age
- Study Time
- Number of Past Failures
- Number of Absences
- First Period Grade (G1)
- Second Period Grade (G2)

The target variable is the Final Grade (G3).

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Machine Learning Model

Linear Regression is used to predict the final grade.

The dataset is divided into:

- 80% Training Data
- 20% Testing Data

## Model Performance

The model achieved the following results:

MAE: 1.33

MSE: 4.46

RMSE: 2.11

R² Score: 0.78

The R² score of 0.78 indicates that the model explains approximately 78% of the variation in the test data.

## Visualizations

The project includes the following visualizations:

- Final Grade Distribution
- Study Time vs Final Grade
- Absences vs Final Grade
- Actual vs Predicted Final Grades

## Project Structure

student-performance-prediction/

    dataset/
        student-mat.csv

    visualizations/
        final_grade_distribution.png
        studytime_vs_grade.png
        absences_vs_grade.png
        actual_vs_predicted.png

    main.py
    README.md
    requirements.txt
    .gitignore

## How to Run

1. Install Python.

2. Clone this repository.

3. Open the project folder in VS Code.

4. Create and activate a virtual environment.

5. Install the required libraries using:

pip install -r requirements.txt

6. Run the project using:

python main.py

## Project Goal

The goal of this project is to demonstrate how Machine Learning can be used to analyze student data and predict academic performance.

## Author

Suman Rautela

B.Tech CSE (AI & ML)

