#Feature Engineering
import pandas as pd

# Load the dataset
data = pd.read_csv('Mall_Customers.csv')

# Example feature engineering steps
# Create a new feature 'total_income' by summing 'annual_income' and 'spending_score'
data['total_income'] = data['Annual Income'] + data['Spending Score']

# Create age groups based on specific ranges
age_bins = [0, 30, 40, 50, 60, 100]
age_labels = ['<30', '30-40', '40-50', '50-60', '60+']
data['Age_group'] = pd.cut(data['Age'], bins=age_bins, labels=age_labels)

# Encode categorical variables using one-hot encoding
data_encoded = pd.get_dummies(data, columns=['Genre'])

# Normalize numerical features using Min-Max scaling
data_encoded['total_income'] = (data_encoded['total_income'] - data_encoded['total_income'].min()) / (data_encoded['total_income'].max() - data_encoded['total_income'].min())

# Drop unnecessary columns
data_final = data_encoded.drop(['CustomerID', 'Age', 'Annual Income(k$)', 'Spending Score(1-100)'], axis=1)

# Print the updated dataset
print(data_final.head())




#Applying Clustering Algorithm
import pandas as pd
from sklearn.cluster import KMeans

# Load the dataset
data = pd.read_csv('Mall_Customers.csv')

# Select the features you want to use for clustering
features = ['Annual Income(k$)', 'Spending Score(1-100)']
X = data[features]

# Instantiate the K-means clustering algorithm
kmeans = KMeans(n_clusters=3, random_state=42)

# Fit the algorithm to the data
kmeans.fit(X)

# Get the cluster labels for each data point
labels = kmeans.labels_

# Add the cluster labels to the original dataset
data['Cluster'] = labels

# Print the resulting clusters
print(data['Cluster'].value_counts())





#Interpretation
import pandas as pd
import matplotlib.pyplot as plt

# Assuming you have already performed clustering and have the cluster labels in your dataset

# Load the dataset
data = pd.read_csv('Mall_Customers.csv')

# Assuming you have a 'cluster' column in your dataset representing the cluster labels
# Calculate the average spending score for each cluster
cluster_avg_spending = data.groupby('cluster')['Spending Score(1-100)'].mean()

# Plot the average spending score for each cluster
plt.bar(cluster_avg_spending.index, cluster_avg_spending.values)
plt.xlabel('Cluster')
plt.ylabel('Average Spending Score')
plt.title('Average Spending Score by Cluster')
plt.show()






#Visualization
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("Mall_Customers.csv")

# Scatter plot
plt.scatter(data['Age'], data['Annual Income'])
plt.xlabel('Age')
plt.ylabel('Annual Income')
plt.title('Age vs Annual Income')
plt.show()

# Histogram
plt.hist(data['Age'], bins=10)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution')
plt.show()

# Bar chart
income_counts = data['Annual Income'].value_counts()
plt.bar(income_counts.index, income_counts.values)
plt.xlabel('Annual Income')
plt.ylabel('Count')
plt.title('Income Distribution')
plt.show()




