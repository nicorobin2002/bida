Sure! Here's your complete content formatted as a **README.md** document, clean and clear, with **no text missed**:

---

# 📘 Data Warehousing and Data Mining Practical Guide

---

## ✅ Practical 1

### **1A. Import Data from a Data Warehouse into Microsoft Excel and Create Pivot Table & Pivot Chart**

#### **Step 1: Import Data from the Data Warehouse**
- Open Excel and go to the **Data** tab.
- Click **Get Data → From Database** → choose the data source.
- Select the appropriate **table or view** from the database.
- Click **Load** to import the data into Excel.

#### **Step 2: Create a Pivot Table**
- Click on any cell in the imported dataset.
- Go to the **Insert** tab → Click **PivotTable**.
- Choose **New Worksheet** or **Existing Worksheet**.
- Drag and drop fields into **Rows, Columns, Values**, and **Filters** as needed.

#### **Step 3: Create a Pivot Chart**
- Click anywhere inside the PivotTable.
- Go to the **Insert** tab → Select **PivotChart**.
- Choose the chart type (e.g., Column, Line, Pie, etc.).
- Click **OK** to generate the chart.
- Customize using the **Chart Tools** options.

---

### **1B. Import OLAP Cube into Excel and Create Pivot Table & Chart**

#### **Step 1: Connect to OLAP Cube in Excel**
- Open Excel → Go to **Data** tab.
- Click **Get Data → From Database → From Analysis Services**.
- Enter Server Name and click **Next**.
- Select the database and cube → Click **Next** → **Finish**.
- Choose **PivotTable Report** → Place on new worksheet.

#### **Step 2: Create a Pivot Table**
- Drag and drop fields:
  - **Rows**: e.g., Product Category, Region
  - **Columns**: e.g., Year, Quarter
  - **Values**: e.g., Sales Amount, Profit
  - **Filters** (optional): Country, Time Period

#### **Step 3: Create a Pivot Chart**
- Select PivotTable → Go to **Insert Tab → PivotChart**
- Choose chart type (Column, Line, Pie, etc.) → Click **OK**
- Customize chart (titles, labels, slicers, formatting)

#### **Step 4: Refresh Data**
- Right-click PivotTable → Click **Refresh**

---

## ✅ Practical 2

### **Apply What-If Analysis in Excel & Generate Reports**

#### **Step 1: Import Data Warehouse Data**
- Open Excel → Data Tab → Get Data → From Other Sources → SQL Server
- Enter server & database name → Click OK → Select tables/views → Click **Load**

#### **Step 2: Create PivotTables and Charts**
- Insert PivotTable → Drag fields into Rows, Columns, Values, Filters
- Create PivotChart → Choose chart type → Format for visualization

#### **Step 3: Apply What-If Analysis**
- **Scenario Manager**: Add scenarios (Best, Worst, Expected) → Compare results
- **Goal Seek**: Set target value → Adjust input → Get results
- **Data Tables**: Use multiple input combinations → Analyze impacts

#### **Step 4: Generate Reports**
- **Summary Report** via Scenario Manager
- **Charts & Dashboards**: Combine PivotTables, Charts, Slicers

---

## ✅ Practical 3

### **Data Classification using R/Python**

```python
pip install pandas numpy scikit-learn matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

print(df.head())

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```

---

## ✅ Practical 4

### **Data Clustering using R/Python (K-Means)**

```python
pip install pandas numpy scikit-learn matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

X, y = make_blobs(n_samples=300, centers=3, random_state=42, cluster_std=1.0)
df = pd.DataFrame(X, columns=['Feature1', 'Feature2'])
print(df.head())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

centers = kmeans.cluster_centers_

plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['Feature1'], y=df['Feature2'], hue=df['Cluster'], palette='viridis', s=50)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering Visualization')
plt.legend()
plt.show()

# Elbow Method
inertia = []
K_range = range(1, 10)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(K_range, inertia, marker='o', linestyle='--')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()
```

---

## ✅ Practical 5

### **Linear Regression using R**

```r
x <- c(151, 174, 138, 186, 128, 136, 179, 163, 152, 131)
y <- c(63, 81, 56, 91, 47, 57, 76, 72, 62, 48)

relation <- lm(y ~ x)
a <- data.frame(x=170)
result <- predict(relation, a)
print(result)

plot(x, y,
     col = "blue",
     main = "Height and Weight Regression",
     abline(relation),
     cex = 1.3,
     pch = 16,
     xlab = "Height in cm",
     ylab = "Weight in kg")
```

[🔗 R PCH Symbols](https://r-charts.com/en/tags/base-r/pch-symbols_files/figure-html/pch-symbols.png)

---

## ✅ Practical 6

### **Logistic Regression using Python**

```python
pip install pandas numpy scikit-learn matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = {
    "Age": [25, 45, 35, 50, 23, 40, 30, 60, 27, 55],
    "Income": [30000, 80000, 50000, 90000, 25000, 70000, 45000, 100000, 32000, 85000],
    "Purchased": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)
print(df.head())

X = df[['Age', 'Income']]
y = df['Purchased']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Decision Boundary
from matplotlib.colors import ListedColormap

X_set, y_set = X_train, y

_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min()-1, stop=X_set[:, 0].max()+1, step=0.1),
                     np.arange(start=X_set[:, 1].min()-1, stop=X_set[:, 1].max()+1, step=1000))
plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.3, cmap=ListedColormap(('red', 'green')))
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.xlabel('Age')
plt.ylabel('Income')
plt.title('Logistic Regression Decision Boundary')
plt.legend()
plt.show()
```

---

## ✅ Practical 7

### **Read CSV and Analyze Data (Python + Pandas)**

```python
pip install pandas numpy matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "data.csv"
df = pd.read_csv(file_path)

print("\n First 5 Rows of Dataset:")
print(df.head())

print("\n Summary Statistics:")
print(df.describe())

print("\n Data Types and Missing Values:")
print(df.info())

missing_values = df.isnull().sum()
print("\n Missing Values Count:")
print(missing_values[missing_values > 0])

df.fillna(df.mean(), inplace=True)

print("\n Column-Wise Unique Values Count:")
print(df.nunique())

print("\n Correlation Matrix:")
print(df.corr())

plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

df.hist(figsize=(10, 8), bins=20, color='skyblue', edgecolor='black')
plt.suptitle("Histogram of Numeric Variables")
plt.show()

df.to_csv("cleaned_data.csv", index=False)
print("\n Data Cleaning Completed. Saved as 'cleaned_data.csv'")
```

---

## ✅ Practical 8

### **8A. Data Visualization in Python**

```python
pip install pandas numpy matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = {
    "Date": pd.date_range(start="2023-01-01", periods=12, freq='M'),
    "Sales": [5000, 7000, 8000, 6500, 7200, 9000, 11000, 10500, 9500, 9800, 12000, 13000],
    "Profit": [800, 1200, 1500, 1000, 1300, 1700, 2200, 2100, 1900, 2000, 2500, 2700],
    "Category": ["Electronics", "Clothing", "Electronics", "Furniture", "Clothing", "Electronics",
                 "Furniture", "Clothing", "Electronics", "Furniture", "Clothing", "Electronics"]
}
df = pd.DataFrame(data)
print(df.head())

# 1. Line Chart
plt.figure(figsize=(10, 5))
plt.plot(df["Date"], df["Sales"], marker='o', color='blue', label="Sales")
plt.xlabel("Month")
plt.ylabel("Sales ($)")
plt.title("Monthly Sales Trend")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.show()

# 2. Bar Chart
plt.figure(figsize=(8, 5))
sns.barplot(x=df["Category"], y=df["Sales"], palette="viridis")
plt.title("Sales by Product Category")
plt.show()

# 3. Scatter Plot
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df["Sales"], y=df["Profit"], hue=df["Category"], palette="deep", s=100)
plt.title("Sales vs. Profit")
plt.show()

# 4. Pie Chart
plt.figure(figsize=(7, 7))
df.groupby("Category")["Sales"].sum().plot.pie(autopct='%1.1f%%',
                                               colors=["skyblue", "lightcoral", "gold"],
                                               startangle=90)
plt.title("Sales Contribution by Category")
plt.ylabel("")
plt.show()
```

---

### **8B. Data Visualization in Power BI**

#### **Step-by-Step Guide**
1. Prepare your `SalesData.xlsx` or `SalesData.csv` with columns:
   - Date, Sales, Profit, Category, Region, Customer_Type

2. Open Power BI Desktop → Click **Get Data → Excel/CSV** → Load your file.

3. Create these visuals:
   - **Line Chart**: Monthly Sales Trend
   - **Bar Chart**: Sales by Category
   - **Scatter Chart**: Sales vs. Profit
   - **Pie Chart**: Sales by Region
   - **KPI Card**: Total Sales

4. Add:
   - **Slicers** for Region, Category, Customer Type
   - **Tooltips**, Conditional Formatting
   - **Publish** via File → Publish → Power BI Service

---

Let me know if you want this as a downloadable `.md` or `.pdf` file too!
