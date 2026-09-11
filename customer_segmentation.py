import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# 1. Load Customer Dataset
data = pd.read_csv("data/customers.csv")

print("Customer Data:")
print(data.head())

# 2. Select Features
features = [
    "Age",
    "Annual_Income",
    "Spending_Score",
    "Purchase_Frequency"
]

X = data[features]

# 3. Data Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Find Optimal Number of Clusters
inertia = []

for k in range(2, 8):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_scaled)
    inertia.append(model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(2, 8), inertia, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()

# 5. Apply K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
data["Cluster"] = kmeans.fit_predict(X_scaled)

# 6. Display Cluster Results
print("\nCustomer Segments:")
print(data[
    [
        "Customer_ID",
        "Age",
        "Annual_Income",
        "Spending_Score",
        "Purchase_Frequency",
        "Cluster"
    ]
])

# 7. Analyze Each Segment
print("\nSegment Analysis:")
segment_analysis = data.groupby("Cluster")[features].mean()
print(segment_analysis)

# 8. PCA for Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

data["PCA1"] = X_pca[:, 0]
data["PCA2"] = X_pca[:, 1]

# 9. Visualize Customer Segments
plt.figure(figsize=(9, 6))

for cluster in sorted(data["Cluster"].unique()):
    cluster_data = data[data["Cluster"] == cluster]
    plt.scatter(
        cluster_data["PCA1"],
        cluster_data["PCA2"],
        label=f"Cluster {cluster}"
    )

plt.title("Customer Segmentation")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.show()

# 10. Spending Score Visualization
plt.figure(figsize=(8, 5))
data.groupby("Cluster")["Spending_Score"].mean().plot(kind="bar")
plt.title("Average Spending Score by Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Average Spending Score")
plt.show()

# 11. Purchase Frequency Visualization
plt.figure(figsize=(8, 5))
data.groupby("Cluster")["Purchase_Frequency"].mean().plot(kind="bar")
plt.title("Average Purchase Frequency by Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Average Purchase Frequency")
plt.show()

# 12. Save Results
data.to_csv("data/customer_segments.csv", index=False)

print("\nCustomer segmentation completed successfully!")
print("Results saved to: data/customer_segments.csv")
