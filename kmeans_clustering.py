import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data = {
    'CustomerID': range(1, 21),
    'AnnualIncome': [15, 16, 17, 18, 19, 25, 26, 27, 28, 29, 60, 61, 62, 63, 64, 85, 86, 87, 88, 89],
    'SpendingScore': [39, 81, 6, 77, 40, 6, 77, 40, 76, 6, 94, 3, 72, 14, 99, 15, 98, 24, 35, 65]
}

df = pd.DataFrame(data)

X = df[['AnnualIncome', 'SpendingScore']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(8, 5))
colors = ['red', 'green', 'blue']
for i in range(3):
    plt.scatter(
        df[df['Cluster'] == i]['AnnualIncome'],
        df[df['Cluster'] == i]['SpendingScore'],
        label=f'Cluster {i}',
        color=colors[i]
    )

centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1], s=300, c='yellow', marker='X', label='Centroids')

plt.title("Customer Segmentation using K-Means")
plt.xlabel("Annual Income (k₹)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.grid(True)
plt.show()
