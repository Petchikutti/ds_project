#Import the necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

#Download the dataset
df=pd.read_csv("Mall_Customers.csv")
print(df.head())

#Check the data info
print(df.info())

print(df.isnull().sum())
df=df.dropna()
print(df.isnull().sum())

#Statistical analysis
print(df.describe())

#Outliers
#Scatter plots
fig= px.scatter(df,x="Age",y="Annual Income (k$)",size='Age')
fig.show()

#Correlation
corr=df.corr()

correlations=df.corr(method='pearson')
plt.figure(figsize=(15,12))
sns.heatmap(correlations,cmap="coolwarm",annot=True)
plt.show()

#Testing and Training the datas
X=df[["Annual Income (k$)","Spending Score (1-100)"]]
y=df["Age"]

#Split the data into training and testing sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#Create a decision tree regressor object
regressor=DecisionTreeRegressor(random_state=42)

#Fit the model using the training data
regressor.fit(X_train,y_train)

#Make predictions using the testing set
y_pred=regressor.predict(X_test)

#Evaluate the performance of the regressor
mse=mean_squared_error(y_test,y_pred)
print(f'Mean Squared Error: {mse}')


