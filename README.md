# Customer Segmentation Project

## Project Overview

Customer Segmentation is a data analytics project that groups customers based on their demographic and purchasing behavior.

The project uses Machine Learning, specifically the K-Means clustering algorithm, to identify different customer groups.

This helps businesses understand customer preferences and create targeted marketing strategies.

## Objectives

- Analyze customer demographic information
- Study customer purchasing behavior
- Group similar customers into segments
- Identify high-value and low-value customers
- Visualize customer segments
- Generate useful business insights

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- K-Means Clustering
- PCA

## Project Structure

Customer-Segmentation/

    data/
        customers.csv
        customer_segments.csv

    customer_segmentation.py
    requirements.txt
    README.md
    .gitignore

## Dataset Features

| Column | Description |
|---|---|
| Customer_ID | Unique customer ID |
| Age | Customer age |
| Annual_Income | Annual income |
| Spending_Score | Customer spending score |
| Purchase_Frequency | Number of purchases |
| Cluster | Assigned customer segment |

## Methodology

### Step 1: Load Dataset
The customer dataset is loaded using Pandas.

### Step 2: Feature Selection
The following features are selected:

- Age
- Annual Income
- Spending Score
- Purchase Frequency

### Step 3: Data Preprocessing
StandardScaler is used to scale the numerical features.

### Step 4: K-Means Clustering
K-Means clustering is used to divide customers into four groups.

### Step 5: Elbow Method
The Elbow Method is used to analyze the suitable number of clusters.

### Step 6: Visualization
PCA is used to reduce the features into two dimensions for visualization.

## Customer Segments

The algorithm creates four customer clusters.

The characteristics of each cluster can be analyzed using the average:

- Age
- Annual Income
- Spending Score
- Purchase Frequency

## Business Insights

Customer segmentation can help businesses:

- Create personalized offers
- Target high-value customers
- Improve marketing campaigns
- Recommend suitable products
- Increase customer engagement
- Improve customer retention

## How to Run

### 1. Open the project folder

```bash
cd Customer-Segmentation
```

### 2. Install required libraries

```bash
pip install -r requirements.txt
```

### 3. Run the Python program

```bash
python customer_segmentation.py
```

## Output

The program generates:

1. Elbow Method graph
2. Customer Segmentation graph
3. Average Spending Score graph
4. Average Purchase Frequency graph
5. Customer segment analysis
6. customer_segments.csv output file

## Expected Outcome

The project provides practical experience in:

- Customer analytics
- Data preprocessing
- Machine learning
- K-Means clustering
- Data visualization
- Business intelligence
- Customer segmentation

## Author

Customer Segmentation Project

Built using Python and Machine Learning.
