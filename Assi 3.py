# 25-11-23
# CSC461- Assignment3 -Machine Learning
# Ayesha Zahid
# Fa21-BSE-003


#-Question No 2-#


#import libraries#
from sklearn import preprocessing
import pandas as pd

#importing  different Machine Learning  classifiers#

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

#import Machine Learning evaluation metrics#

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn import metrics, model_selection

#read CSV file (dataset)

gender_file = pd.read_csv('D:\Ayesha\SEM 5\Data Science\gender-prediction.csv')
#gender_file

#convert categorical data (beard, hair_length, scarf, eye_color, gender) into numerical data

labels = preprocessing.LabelEncoder()
gender_file['beard_encoded'] = labels.fit_transform(gender_file['beard'])
gender_file['hair_length_encoded'] = labels.fit_transform(gender_file['hair_length'])
gender_file['scarf_encoded'] = labels.fit_transform(gender_file['scarf'])
gender_file['eye_color_encoded'] = labels.fit_transform(gender_file['eye_color'])
gender_file['gender_encoded'] = labels.fit_transform(gender_file['gender'])

#Assign input data to  variable "features"and output data to variable"targets" 

features = list(zip(gender_file['height'], gender_file['weight'], gender_file['beard_encoded'], gender_file['hair_length_encoded'], gender_file['shoe_size'], gender_file['scarf_encoded'], gender_file['eye_color_encoded']))
targets = gender_file['gender_encoded']

#Assigning input values to 'x' and output values to 'y'
x = features
y = targets

#making train/test split
X_train, x_test, Y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 42)

#-Question No. 2--
#Part 1: How many instances are incorrectly classified?
#Executing the Required Models (Logistic Regression Mode, Support Vector Machines, and Multilayer Perceptron classification algorithms Model

model_lr = LogisticRegression()
model_svc = SVC()
model_mlp = MLPClassifier()

#training the models using the training data (2/3 of total data)

model_lr.fit(X_train, Y_train)
model_svc.fit(X_train, Y_train)
model_mlp.fit(X_train, Y_train)

#make predictions using the trained models

prediction_lr = model_lr.predict(x_test)
prediction_svc = model_svc.predict(x_test)
prediction_mlp = model_mlp.predict(x_test)

#Calculating incorrects by different models

lr_incorrect = (y_test != prediction_lr).sum()
svc_incorrect = (y_test != prediction_svc).sum()
mlp_incorrect = (y_test != prediction_mlp).sum()

print("Part 1")
print("Answers")

print("Incorrects by Logistic Regression: ",lr_incorrect)
print("Incorrects by SVC: ",svc_incorrect)
print("Incorrects by MLP: ",mlp_incorrect)

#Part 1
#Answers
#Incorrects by Logistic Regression:  2
#Incorrects by SVC:  4
#Incorrects by MLP:  12

#--Question No. 2--#
#Part 2: Rerun the experiment using train/test split ratio of 80/20. Do you see any change in the results? 
#make train/test split for 80/20 train/test data, so test data size would be 0.20

X_train, x_test, Y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 42)
model_lr = LogisticRegression()
model_svc = SVC()
model_mlp = MLPClassifier()

#train the model using the data (training data)

model_lr.fit(X_train, Y_train)
model_svc.fit(X_train, Y_train)
model_mlp.fit(X_train, Y_train)

#make prediction using the trained model

prediction_lr = model_lr.predict(x_test)
prediction_svc = model_svc.predict(x_test)
prediction_mlp = model_mlp.predict(x_test)

#Calculate incorrect

lr_correct = (y_test == prediction_lr).sum()
lr_incorrect = (y_test != prediction_lr).sum()

svc_correct = (y_test == prediction_svc).sum()
svc_incorrect = (y_test != prediction_svc).sum()

mlp_correct = (y_test == prediction_mlp).sum()
mlp_incorrect = (y_test != prediction_mlp).sum()

print("Corrects by Logistic Regression: ",lr_correct)
print("Incorrects by Logistic Regression: ",lr_incorrect)

print("Corrects by SVC: ",svc_correct)
print("Incorrects by SVC: ",svc_incorrect)

print("Corrects by MLP: ",mlp_correct)
print("Incorrects by MLP: ",mlp_incorrect)

#Corrects by Logistic Regression:  22
#Incorrects by Logistic Regression:  0
#Corrects by SVC:  18
#Incorrects by SVC:  4
#Corrects by MLP:  17
#Incorrects by MLP:  5


#-Question No. 2#

#Part 4: Try to exclude these 2 attribute(s) from the dataset. Rerun the experiment (using 80/20 train/test split), 
#did you find any change in the results? Explain. 
#excluding the "Powerful" attributs and assigning values to 'x' and 'y'
 
x = list(zip(gender_file['weight'], gender_file['beard_encoded'], gender_file['hair_length_encoded'], gender_file['scarf_encoded'], gender_file['eye_color_encoded']))
y = gender_file['gender_encoded']

#make train/test split

X_train, x_test, Y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 42)
model_lr = LogisticRegression()
model_svc = SVC()
model_mlp = MLPClassifier()

#train the model using the data (training data)

model_lr.fit(X_train, Y_train)
model_svc.fit(X_train, Y_train)
model_mlp.fit(X_train, Y_train)

#make prediction using the trained model

prediction_lr = model_lr.predict(x_test)
prediction_svc = model_svc.predict(x_test)
prediction_mlp = model_mlp.predict(x_test)

#Calculate incorrect

lr_correct = (y_test == prediction_lr).sum()
lr_incorrect = (y_test != prediction_lr).sum()

svc_correct = (y_test == prediction_svc).sum()
svc_incorrect = (y_test != prediction_svc).sum()

mlp_correct = (y_test == prediction_mlp).sum()
mlp_incorrect = (y_test != prediction_mlp).sum()

print("Corrects by Logistic Regression: ",lr_correct)
print("Incorrects by Logistic Regression: ",lr_incorrect)

print("Corrects by SVC: ",svc_correct)
print("Incorrects by SVC: ",svc_incorrect)

print("Corrects by MLP: ",mlp_correct)
print("Incorrects by MLP: ",mlp_incorrect)

#print("Instances incorrectly classified according to MLP are: ",mlp_incorrect)

#Corrects by Logistic Regression:  20
#Incorrects by Logistic Regression:  2
#Corrects by SVC:  18
#Incorrects by SVC:  4
#Corrects by MLP:  10
#Incorrects by MLP:  12


#-Question No. 3-#

#Again combining all the data icluding all the attributes for Question 3

features = list(zip(gender_file['height'], gender_file['weight'], gender_file['beard_encoded'], gender_file['hair_length_encoded'], gender_file['shoe_size'], gender_file['scarf_encoded'], gender_file['eye_color_encoded']))
targets = gender_file['gender_encoded']

x = features
y = targets

#importing RandomForestClassifier, LeavePOut, cross_val_score, numpy for question 3

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

from sklearn.model_selection import LeavePOut

# Creating a Random Forest Classifier instance
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

X_train, x_test, Y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 42)
rf_classifier.fit(X_train, Y_train)
prediction_rf = rf_classifier.predict(x_test)

# Perform Monte Carlo Cross-Validation with 10 iterations
iterations = 10
    
f1_score = rf_classifier.score(x_test, y_test)

f1_scores_monte_carlo = cross_val_score(rf_classifier, x, y, cv=10, scoring='f1')

n_samples = len(x)
p = 1
leave_p_out = LeavePOut(p)

f1_scores_leave_p_out = cross_val_score(rf_classifier, x, y, cv=leave_p_out, scoring='f1')

# Print F1 scores obtained from Monte Carlo Cross-Validation
#print("F1 scores for Monte Carlo Cross-Validation:", f1_scores_monte_carlo)
print("Mean F1 score for Monte Carlo Cross-Validation:", np.mean(f1_scores_monte_carlo))

# Print F1 scores obtained from Leave P-Out Cross-Validation
#print("F1 scores for Leave P-Out Cross-Validation:", f1_scores_leave_p_out)
print("Mean F1 score for Leave P-Out Cross-Validation:", np.mean(f1_scores_leave_p_out))

#Mean F1 score for Monte Carlo Cross-Validation: 0.9755244755244755
#Mean F1 score for Leave P-Out Cross-Validation: 0.5545454545454546
