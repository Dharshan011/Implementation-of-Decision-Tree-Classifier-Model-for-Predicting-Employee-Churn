# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Prepare your data

2.Define your model

3.Define your cost function

4.Define your learning rate

5.Train your model

6.Evaluate your model

7.Tune hyperparameters

8.Deploy your model 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by :DHARSHAN V
Register No: 212222230031 
*/
```
```
import pandas as pd
data=pd.read_csv("Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:

![Screenshot 2023-10-16 214652](https://github.com/Dharshan011/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113497491/8668b9e2-bfb8-4c91-b327-640e5040c9b9)


![Screenshot 2023-10-16 214700](https://github.com/Dharshan011/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113497491/6795ee76-74c7-4f9d-baef-497dfa965684)


![Screenshot 2023-10-16 214707](https://github.com/Dharshan011/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113497491/74d269b0-6ab3-4a5b-a6f4-a87da18f29c2)

![Screenshot 2023-10-16 214720](https://github.com/Dharshan011/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113497491/fb178812-c7ac-418e-aec4-fc5190856ab5)


![Screenshot 2023-10-16 214729](https://github.com/Dharshan011/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113497491/8a725804-4e1d-4a1b-bd31-ded42390e8b8)



![Screenshot 2023-10-16 214737](https://github.com/Dharshan011/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113497491/d04edf5e-7604-45a3-925e-b799ecb91023)























## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
