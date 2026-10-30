# ONLINE RETAIL SALES ANALYSIS AND CUSTOMER SEGMENTATION USING PYTHON

## 1. ABSTRACT

The Online Retail Sales Analysis and Customer Segmentation project is a Data Science project developed using Python. The main objective of this project is to analyze online retail transaction data, identify sales patterns, understand customer purchasing behaviour, and group customers based on their purchasing activities.

The project uses an online retail dataset containing information about invoice numbers, product details, quantities, transaction dates, unit prices, customer IDs, and countries.

The project begins with data collection and data preprocessing. Missing values, duplicate records, invalid transactions, and incorrect data formats are handled during the data cleaning process.

Exploratory Data Analysis (EDA) is performed to understand sales trends, identify top-selling products, analyze country-wise revenue, and study customer purchasing patterns.

RFM (Recency, Frequency, and Monetary) analysis is used to calculate customer purchasing behaviour. K-Means clustering, an unsupervised machine learning algorithm, is then applied to group customers into different segments.

Finally, data visualization techniques are used to present the results through monthly sales charts, product sales charts, customer segmentation graphs, and country-wise sales analysis.

The project helps businesses understand customer behaviour, identify valuable customers, improve marketing strategies, and make better business decisions using data-driven insights.

## 2. INTRODUCTION

In today's business environment, online retail companies generate large amounts of transaction data every day.

This data contains valuable information about customer purchases, product performance, revenue generation, and shopping behaviour.

However, raw transaction data cannot directly provide useful business insights. It must be processed, cleaned, analyzed, and visualized.

Data Science provides different techniques to convert raw data into meaningful information.

This project focuses on analyzing online retail transactions using Python and applying machine learning techniques to identify different customer groups.

The project combines data preprocessing, exploratory data analysis, feature engineering, data visualization, and machine learning.

## 3. PROBLEM STATEMENT

Online retail businesses face several challenges:

1. Difficulty identifying their best-selling products.
2. Difficulty understanding monthly sales performance.
3. Difficulty identifying high-value customers.
4. Lack of knowledge about customer purchasing frequency.
5. Difficulty identifying customers who have stopped purchasing.
6. Challenges in developing effective marketing strategies.

The main problem is that businesses have large amounts of transaction data but may not have sufficient information about customer behaviour and sales patterns.

Therefore, this project analyzes retail transaction data and applies customer segmentation techniques to generate meaningful business insights.

## 4. PROJECT OBJECTIVES

The major objectives are:

1. To understand online retail transaction data.
2. To clean and preprocess the dataset.
3. To identify and handle missing values and duplicate records.
4. To calculate total revenue and average order value.
5. To analyze monthly sales trends.
6. To identify the top-selling products.
7. To analyze country-wise revenue.
8. To understand customer purchasing behaviour.
9. To calculate Recency, Frequency, and Monetary values.
10. To apply K-Means clustering for customer segmentation.
11. To visualize the results using Python libraries.
12. To provide useful business recommendations.

## 5. SCOPE OF THE PROJECT

This project focuses on retail transaction analysis and customer segmentation.

The scope includes:

- Transaction-level sales analysis.
- Product performance analysis.
- Country-wise sales analysis.
- Monthly revenue analysis.
- Customer purchasing behaviour.
- Customer value analysis.
- Customer grouping using machine learning.
- Data visualization and business reporting.

The project can be extended in the future by developing a customer churn prediction model, product recommendation system, or interactive business dashboard.

## 6. DATASET DESCRIPTION

The project uses the Online Retail dataset.

The dataset contains 541,909 original records and 8 columns.

### Dataset attributes

| Column | Description |
|---|---|
| InvoiceNo | Unique invoice or transaction number |
| StockCode | Product identification code |
| Description | Product description |
| Quantity | Number of products purchased |
| InvoiceDate | Date and time of transaction |
| UnitPrice | Price of one product |
| CustomerID | Unique customer identification number |
| Country | Country of the customer |

### Example dataset

| InvoiceNo | StockCode | Quantity | UnitPrice | Country |
|---|---|---:|---:|---|
| 536365 | 85123A | 6 | 2.55 | United Kingdom |
| 536365 | 71053 | 6 | 3.39 | United Kingdom |
| 536365 | 84406B | 8 | 2.75 | United Kingdom |

### Dataset features

- Total original records: 541,909
- Total columns: 8
- Unique invoice numbers: 25,900
- Unique countries: 38
- Missing Description values: 1,454
- Missing CustomerID values: 135,080
- Duplicate records: 5,268

These values describe the original uploaded dataset before cleaning.

## 7. TECHNOLOGIES AND TOOLS USED

### 7.1 Python

Python is the primary programming language used for implementing the project.

It provides libraries for data manipulation, visualization, statistical analysis, and machine learning.

### 7.2 Pandas

Pandas is used for:

- Reading CSV files.
- Handling missing values.
- Removing duplicate records.
- Filtering data.
- Grouping transaction records.
- Calculating revenue.
- Creating new columns.

### 7.3 NumPy

NumPy is used for numerical operations and mathematical calculations.

### 7.4 Matplotlib

Matplotlib is used to create:

- Line charts.
- Bar charts.
- Scatter plots.
- Monthly revenue graphs.

### 7.5 Seaborn

Seaborn is used for statistical data visualization and customer segmentation plots.

### 7.6 Scikit-learn

Scikit-learn is used for:

- Feature scaling.
- K-Means clustering.
- Model evaluation.
- Silhouette score calculation.

### 7.7 Jupyter Notebook

Jupyter Notebook provides an interactive environment to execute Python code step by step and visualize the results.

## 8. SYSTEM REQUIREMENTS

### Hardware requirements

- Processor: Intel i3 or above.
- RAM: Minimum 4 GB; 8 GB recommended.
- Storage: Sufficient space for the dataset and output files.
- System: Desktop or laptop.

### Software requirements

- Operating System: Windows, Linux, or macOS.
- Python: Version 3.10 or above.
- Jupyter Notebook or VS Code.
- Required Python libraries.

### Library installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## 9. PROJECT METHODOLOGY

The project follows these stages:

1. Data collection.
2. Data understanding.
3. Data preprocessing.
4. Exploratory Data Analysis.
5. Data visualization.
6. Feature engineering.
7. RFM analysis.
8. Machine learning using K-Means.
9. Model evaluation.
10. Business insights and conclusions.

## 10. DATA COLLECTION

The dataset is loaded into Python using the Pandas library.

```python
import pandas as pd

df = pd.read_csv("online_retail.csv")

print(df.head())
print(df.shape)
```

Explanation:

- `read_csv()` loads the CSV dataset.
- `head()` displays the first five rows.
- `shape` returns the number of rows and columns.

The dataset is then inspected to understand its structure and contents.

## 11. DATA UNDERSTANDING

Before performing analysis, the dataset must be understood.

The following commands are used:

```python
print(df.info())

print(df.describe())

print(df.isnull().sum())

print(df.duplicated().sum())
```

Purpose:

- `info()` displays column names, data types, and non-null counts.
- `describe()` provides statistical information.
- `isnull().sum()` identifies missing values.
- `duplicated().sum()` identifies duplicate records.

This step helps identify data quality problems.

## 12. DATA PREPROCESSING AND CLEANING

Data preprocessing is the process of converting raw data into a suitable format for analysis.

### 12.1 Removing duplicate records

Duplicate records can affect the accuracy of revenue calculations.

```python
df = df.drop_duplicates()
```

### 12.2 Handling missing values

The Description column contains missing values.

```python
df["Description"] = df["Description"].fillna(
    "Unknown Product"
)
```

CustomerID contains missing values.

These records can be retained for general sales analysis, but customers without an ID cannot be included in customer-level segmentation.

```python
customer_df = df.dropna(
    subset=["CustomerID"]
)
```

### 12.3 Converting date column

InvoiceDate is converted into datetime format.

```python
df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"],
    errors="coerce"
)
```

This allows monthly, yearly, and time-based analysis.

### 12.4 Removing cancelled transactions

Some invoice numbers start with the letter C, indicating cancelled transactions.

```python
df = df[
    ~df["InvoiceNo"].astype(str).str.startswith("C")
]
```

### 12.5 Removing invalid purchases

Negative quantities may represent returned products, while non-positive prices are not treated as normal sales.

```python
df = df[
    (df["Quantity"] > 0) &
    (df["UnitPrice"] > 0)
].copy()
```

For this project, cancelled and invalid purchase records are excluded from the main positive-sales analysis.

## 13. FEATURE ENGINEERING

Feature engineering means creating new useful columns from existing dataset columns.

### 13.1 Calculating total sales

Total sales are calculated by multiplying Quantity by UnitPrice.

Formula:

Total Sales = Quantity × UnitPrice

```python
df["TotalSales"] = (
    df["Quantity"] * df["UnitPrice"]
)
```

### 13.2 Extracting month

```python
df["Month"] = df["InvoiceDate"].dt.month
```

### 13.3 Extracting year

```python
df["Year"] = df["InvoiceDate"].dt.year
```

### 13.4 Extracting hour

```python
df["Hour"] = df["InvoiceDate"].dt.hour
```

These new features help analyze time-based purchasing behaviour.

## 14. EXPLORATORY DATA ANALYSIS (EDA)

Exploratory Data Analysis is used to understand the dataset and identify patterns, relationships, and trends.

### 14.1 Total revenue analysis

Total revenue is calculated by adding all valid transaction values.

```python
total_revenue = df["TotalSales"].sum()

print("Total Revenue:", total_revenue)
```

The cleaned purchase records in the supplied dataset produced approximately 10,642,110.80 monetary units.

### 14.2 Total order analysis

```python
total_orders = df["InvoiceNo"].nunique()

print("Total Orders:", total_orders)
```

This calculates the number of unique invoices.

### 14.3 Customer analysis

```python
total_customers = df["CustomerID"].nunique()

print("Total Customers:", total_customers)
```

This identifies the number of unique customers with available customer IDs.

### 14.4 Average order value

Formula:

Average Order Value = Total Revenue / Number of Unique Orders

```python
average_order_value = (
    total_revenue / total_orders
)

print("Average Order Value:", average_order_value)
```

Average Order Value helps understand the average revenue generated per invoice.

## 15. MONTHLY SALES ANALYSIS

Monthly sales analysis helps identify changes in business revenue over time.

The objective is to identify high-performing and low-performing months.

```python
df["MonthYear"] = (
    df["InvoiceDate"].dt.to_period("M")
)

monthly_sales = (
    df.groupby("MonthYear")["TotalSales"]
    .sum()
)
```

Visualization:

```python
monthly_sales.plot(
    kind="line",
    marker="o",
    figsize=(12, 5)
)

plt.title("Monthly Sales Analysis")
plt.xlabel("Month")
plt.ylabel("Revenue")

plt.xticks(rotation=45)
plt.show()
```

Expected interpretation:

- Identify months with high revenue.
- Identify months with low revenue.
- Understand seasonal sales patterns.
- Support future sales planning.

## 16. TOP-SELLING PRODUCT ANALYSIS

This analysis identifies products generating the highest revenue.

```python
top_products = (
    df.groupby("Description")["TotalSales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print(top_products)
```

Visualization:

```python
top_products.plot(
    kind="barh",
    figsize=(10, 6)
)

plt.title("Top 10 Selling Products")
plt.xlabel("Total Sales")
plt.ylabel("Products")

plt.show()
```

Business importance:

- Identify popular products.
- Improve inventory planning.
- Understand product performance.
- Support promotional strategies.

## 17. COUNTRY-WISE SALES ANALYSIS

This analysis identifies which countries contribute the highest revenue.

```python
country_sales = (
    df.groupby("Country")["TotalSales"]
    .sum()
    .sort_values(ascending=False)
)

print(country_sales.head(10))
```

Visualization:

```python
country_sales.head(10).plot(
    kind="bar",
    figsize=(12, 5)
)

plt.title("Country-wise Sales Analysis")
plt.xlabel("Country")
plt.ylabel("Revenue")

plt.show()
```

Business importance:

- Identify important markets.
- Understand geographical sales performance.
- Support regional marketing decisions.
- Identify opportunities for expansion.

## 18. CUSTOMER SEGMENTATION USING RFM ANALYSIS

RFM analysis is a customer segmentation technique based on three important purchasing characteristics.

### 18.1 Recency

Recency measures how recently a customer purchased a product.

A lower Recency value indicates a more recent purchase.

Formula:

Recency = Reference Date − Last Purchase Date

### 18.2 Frequency

Frequency measures how many unique invoices a customer has generated.

A higher Frequency value indicates more frequent purchases.

### 18.3 Monetary

Monetary measures the total amount spent by a customer.

A higher Monetary value indicates higher customer spending.

### RFM calculation

```python
customer_df = df.dropna(
    subset=["CustomerID"]
).copy()

snapshot_date = (
    customer_df["InvoiceDate"].max()
    + pd.Timedelta(days=1)
)

rfm = customer_df.groupby("CustomerID").agg({

    "InvoiceDate": lambda x:
        (snapshot_date - x.max()).days,

    "InvoiceNo": "nunique",

    "TotalSales": "sum"

})

rfm.columns = [
    "Recency",
    "Frequency",
    "Monetary"
]

print(rfm.head())
```

The resulting RFM table provides customer-level purchasing information.

## 19. MACHINE LEARNING USING K-MEANS CLUSTERING

K-Means is an unsupervised machine learning algorithm used to group similar data points into clusters.

In this project, K-Means is used to group customers based on their Recency, Frequency, and Monetary values.

### 19.1 Why K-Means?

K-Means is selected because:

- It is suitable for customer segmentation.
- It groups customers based on similarities.
- It is relatively simple to implement.
- It helps identify different purchasing patterns.

### 19.2 Feature selection

```python
X = rfm[
    ["Recency", "Frequency", "Monetary"]
]
```

### 19.3 Feature scaling

Different features have different numerical ranges.

StandardScaler is used to standardize them.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

### 19.4 Elbow Method

The Elbow Method helps identify a suitable number of clusters.

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inertia = []

for k in range(1, 11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    inertia.append(model.inertia_)

plt.plot(
    range(1, 11),
    inertia,
    marker="o"
)

plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")

plt.show()
```

The optimal number of clusters is selected by examining the point where the reduction in inertia begins to slow down.

### 19.5 Applying K-Means

Five clusters can be used as an initial experiment.

The final cluster count should be selected based on the Elbow Method and model evaluation.

```python
kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

rfm["Cluster"] = kmeans.fit_predict(
    X_scaled
)

print(rfm.head())
```

## 20. MODEL EVALUATION

The Silhouette Score is used to evaluate the quality of customer clusters.

It measures how well each data point fits within its assigned cluster compared with other clusters.

The score generally ranges from -1 to +1.

- Close to +1: Better-separated clusters.
- Around 0: Overlapping clusters.
- Negative values: Some points may be assigned to inappropriate clusters.

```python
from sklearn.metrics import silhouette_score

score = silhouette_score(
    X_scaled,
    rfm["Cluster"]
)

print("Silhouette Score:", score)
```

The score should be interpreted along with the business usefulness of the resulting segments.

## 21. CUSTOMER CLUSTER VISUALIZATION

Customer clusters are visualized using a scatter plot.

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(
    data=rfm,
    x="Frequency",
    y="Monetary",
    hue="Cluster",
    palette="bright",
    s=60
)

plt.xscale("log")
plt.yscale("log")

plt.title("Customer Segmentation")
plt.xlabel("Purchase Frequency")
plt.ylabel("Monetary Value")

plt.show()
```

This visualization helps understand the distribution of customers across clusters.

## 22. CUSTOMER SEGMENT INTERPRETATION

After clustering, average RFM values are calculated for each cluster.

```python
cluster_analysis = rfm.groupby(
    "Cluster"
).agg({

    "Recency": "mean",
    "Frequency": "mean",
    "Monetary": "mean"

})

print(cluster_analysis)
```

Possible interpretations:

### High-value customers

- Low Recency.
- High Frequency.
- High Monetary value.

Business strategy: Provide loyalty rewards and exclusive offers.

### Regular customers

- Moderate Recency.
- Moderate Frequency.
- Consistent purchasing activity.

Business strategy: Encourage repeat purchases.

### At-risk customers

- High Recency.
- Previously active purchasing behaviour.

Business strategy: Introduce customer re-engagement campaigns.

### Low-engagement customers

- Low Frequency.
- Low Monetary value.

Business strategy: Offer targeted discounts and product recommendations.

Note: Cluster numbers themselves do not have fixed meanings. The labels must be assigned after inspecting each cluster's average RFM values.

## 23. BUSINESS INTELLIGENCE DASHBOARD

A combined dashboard is developed using Matplotlib and Seaborn.

The dashboard contains four main visualizations and business KPIs.

### Dashboard components

1. Total Revenue.
2. Total Orders.
3. Total Customers.
4. Average Order Value.
5. Monthly Sales Analysis.
6. Top-Selling Products.
7. Customer Segmentation.
8. Country-wise Sales.

The dashboard provides a summarized view of retail business performance.

It helps users understand important business information without examining individual transaction records.

## 24. EXPECTED OUTPUTS

The project generates the following outputs:

| Output | Purpose |
|---|---|
| Total Revenue | Overall sales value |
| Total Orders | Unique invoice count |
| Total Customers | Identified customer count |
| Average Order Value | Average revenue per invoice |
| Monthly Sales Chart | Monthly revenue trends |
| Product Sales Chart | Top-performing products |
| Country Sales Chart | Geographical sales performance |
| Customer Clusters | Customer purchasing groups |
| Silhouette Score | Cluster quality evaluation |
The actual numerical results and final cluster interpretations should be recorded after running the complete notebook.

## 25. ADVANTAGES OF THE PROJECT

1. Provides better understanding of customer purchasing behaviour.
2. Identifies products generating higher revenue.
3. Helps analyze sales performance over time.
4. Identifies high-value customer groups.
5. Supports customer retention strategies.
6. Helps businesses make data-driven decisions.
7. Demonstrates practical use of Python and machine learning.
8. Combines data analysis and machine learning in one project.

## 26. LIMITATIONS

1. CustomerID is missing for many transactions.
2. The dataset contains historical transaction information.
3. K-Means results depend on the selected features and cluster count.
4. Customer segmentation does not automatically predict future purchases.
5. The project does not establish why a customer purchases or stops purchasing.
6. The current dashboard is based on static visualizations.
7. Returns and cancellations are excluded from the main positive-sales analysis.

## 27. FUTURE ENHANCEMENTS

The project can be improved by implementing:

### 27.1 Customer Churn Prediction

Develop a supervised machine learning model to identify customers who may stop purchasing.

### 27.2 Product Recommendation System

Recommend products based on customer purchasing history.

### 27.3 Interactive Dashboard

Develop a Streamlit or Power BI dashboard with interactive filters.

### 27.4 Sales Forecasting

Use time-series forecasting techniques to estimate future revenue.

### 27.5 Market Basket Analysis

Identify products frequently purchased together using association rule mining.

## 28. CONCLUSION

The Online Retail Sales Analysis and Customer Segmentation project demonstrates how Data Science techniques can be applied to real-world retail transaction data.

The project uses Python libraries for data preprocessing, exploratory data analysis, data visualization, feature engineering, and machine learning.

RFM analysis is used to understand customer purchasing behaviour, while K-Means clustering groups customers based on their purchasing characteristics.

The visualizations provide information about monthly revenue, top-selling products, geographical sales distribution, and customer segments.

Overall, the project demonstrates the importance of data-driven decision-making in retail businesses and provides a foundation for developing advanced customer analytics and business intelligence solutions.

## 29. REFERENCES

1. Python Documentation – https://docs.python.org/3/
2. Pandas Documentation – https://pandas.pydata.org/docs/
3. NumPy Documentation – https://numpy.org/doc/
4. Matplotlib Documentation – https://matplotlib.org/stable/
5. Seaborn Documentation – https://seaborn.pydata.org/
6. Scikit-learn Documentation – https://scikit-learn.org/stable/
7. Jupyter Documentation – https://docs.jupyter.org/

---

# END OF PROJECT DOCUMENTATION
