# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1. Start the program

Step 2. Import the necessary python libraries

Step 3. Read the dataset of Mall_Customers csv file

Step 4. From sklearn libraary select the cluster and import KMeans Clustering

Step 5. Find the sum of squared distance between each points and the centroid in a cluster using Elbow Method

Step 6. Plot the graph x and y as Number of Clusters and wcss respectively

Step 7. Using the matplotlib library draw the scatter plot for the given number of clusters (ie. here n_clusters = 5)

Step 8. Stop the program

## Program:

Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: KISHAN SHREE B
RegisterNumber: 212223100022

```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
data = pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse = metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
plt.figure(figsize=(20, 8))
plot_tree(dt, feature_names=x.columns, filled=True)
plt.show()

```
## Output:
X Value:

![image](https://github.com/user-attachments/assets/a385bb8e-641c-42f7-8c59-b0212586e350)

Y Value:

![image](https://github.com/user-attachments/assets/fe1dcf5a-f275-43ca-ad7a-c367fda1a18e)

DecisionTree:

![image](https://github.com/user-attachments/assets/739aa259-1e17-4814-a1db-c4e51b03bc51)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
