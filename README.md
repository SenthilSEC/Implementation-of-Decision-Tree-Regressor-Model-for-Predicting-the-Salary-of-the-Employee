# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.

2.Calculate the null values present in the dataset and apply label encoder.

3.Determine test and training data set and apply decison tree regression in dataset.

4.Calculate Mean square error,data prediction and r2.
 

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Senthil Arunachalam P
RegisterNumber:  212224240147
*/
```
```py
import pandas as pd 
data=pd.read_csv("Salary.csv") 
data.head() 
data.info() 
data.isnull().sum() 
from sklearn.preprocessing import LabelEncoder 
le=LabelEncoder() 
data["Position"]=le.fit_transform(data["Position"]) 
x=data[["Position","Level"]] 
x.head() 
y=data["Salary"] 
y.head() 
from sklearn.model_selection import train_test_split 
from sklearn import metrics
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2) 
from sklearn.tree import DecisionTreeRegressor 
dt=DecisionTreeRegressor() 
dt.fit(x_train,y_train) 
y_pred=dt.predict(x_test) 
print(y_pred )

mse=metrics.mean_squared_error(y_test, y_pred)
print(mse)
r2= metrics.r2_score(y_test,y_pred)
print(r2)
print (dt.predict([[5,6]]))
```

## Output:
Data Head:

<img width="328" height="238" alt="image" src="https://github.com/user-attachments/assets/ca65f055-0b3b-4ae9-b325-c543fd04dbe6" />

isnull().sum():

<img width="316" height="170" alt="image" src="https://github.com/user-attachments/assets/849a8bde-93c1-4abb-81e2-cde1da5b4420" />

Data Head for salary:

<img width="179" height="105" alt="image" src="https://github.com/user-attachments/assets/36c908ad-ee34-40f0-a46c-02df885db1d5" /> 

Data info:

<img width="428" height="249" alt="image" src="https://github.com/user-attachments/assets/68c2c1fc-9e14-48f5-b3e5-a1e1a68b7682" />

Mean Squared Error and R2 Value:

<img width="229" height="89" alt="image" src="https://github.com/user-attachments/assets/b166964f-c81d-4e6f-83bd-c8d02d42f0e0" />

Data Prediction:

<img width="125" height="38" alt="image" src="https://github.com/user-attachments/assets/3d607e89-9cb6-4e63-ba80-2655d4ff2ea5" />

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
