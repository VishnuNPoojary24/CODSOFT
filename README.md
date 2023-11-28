# CodSoft-ML
## Spam SMS Detection
### Overview
This project uses NLP techniques to build a spam SMS classifier. The model, based on TF-IDF and Naive Bayes, distinguishes between spam and legitimate messages.

### Files
spam_detection.ipynb: Jupyter Notebook for code and analysis.  
spam.csv: Dataset with labeled SMS messages.  
### Setup Libraries:
pip install pandas scikit-learn nltk numpy  
### Dataset:
Download and place spam.csv in the project directory.  
### Usage
Open spam_detection.ipynb.  
Run the notebook for training and evaluation.  
Input your SMS to check classification.  
### Notes
TF-IDF for feature extraction, Naive Bayes for prediction.  
Data preprocessing: stop word removal and stemming.


## Movie Genre Classification
### Overview
This project focuses on creating a machine learning model for predicting the genre of a movie based on its plot summary or textual information. The model utilizes TF-IDF (Term Frequency-Inverse Document Frequency) for feature extraction and a Support Vector Machine (SVM) classifier for genre prediction.  

### Files
movie_genre_classification.ipynb: Jupyter Notebook containing the code and analysis.  
train_data.csv: Dataset with labeled movie data for training.  
test_data.csv: Dataset for testing the trained model.  
### Setup Libraries  
Ensure you have the necessary libraries installed:  
pip install pandas scikit-learn numpy  
### Usage
Open movie_genre_classification.ipynb.  
Load and preprocess the training dataset (train_data.csv).  
Train the model using TF-IDF features and an SVM classifier.  
Load the test dataset (test_data.csv).  
Input a movie description for genre prediction.  
Evaluate the model's performance using metrics like confusion matrix, accuracy, and classification report.  
### Notes
TF-IDF is used to convert plot summaries into numerical features.  
The SVM classifier with a linear kernel is employed for genre prediction.  
The model is trained on the provided training dataset and tested on a separate test dataset.  
User interaction is implemented to input a movie description and obtain the predicted genre.  


## Churn Prediction with Random Forest
### Overview
This project is dedicated to predicting customer churn using machine learning techniques. The model is trained on a dataset containing customer information, and the Random Forest algorithm is employed for classification.
### Files
churn_prediction.ipynb: Jupyter Notebook containing the code and analysis.  
Churn_Modelling.csv: Dataset with customer information.  
### Setup Libraries
Ensure you have the necessary libraries installed:  
pip install pandas scikit-learn imbalanced-learn  
### Data Exploration and Preprocessing
Irrelevant columns (RowNumber, CustomerId, Surname) are dropped.  
Categorical variables (Geography, Gender) are converted to numerical using one-hot encoding.  
The dataset is split into features (X) and the target variable (y).  
Imbalanced data is handled using SMOTE (Synthetic Minority Over-sampling Technique).  
The data is split into training and testing sets.  
Model Training and Evaluation  
### Random Forest
A Random Forest classifier is trained on the resampled data.  
Grid search is used for hyperparameter tuning.  
Model performance is evaluated using accuracy, confusion matrix, and classification report.  

### User Input for Churn Prediction
User input is accepted for relevant features.  
The input is converted to a DataFrame and standardized.  
Predictions are made on the user input, and the churn status is displayed.  
