Sure! Here's the **entire content** formatted cleanly in **README.md** style using **Markdown**, as requested. This version preserves **all text exactly as given**, adds clear formatting, and organizes the practicals for better readability.

---

```markdown
# Data Analysis & Visualization Practical Manual

---

## **Practical 1**

### Task:
- Import the data warehouse data in Microsoft Excel and create the Pivot table and Pivot Chart.
- Import the cube in Microsoft Excel and create the Pivot table and Pivot Chart to perform data analysis.

---

### Output:

*Add your Excel Pivot Table and Pivot Chart screenshots here*

---

## **Practical 2**

### Task:
Apply the what–if Analysis for data visualization. Design and generate necessary reports based on the data warehouse data. Use Excel.

1. On the **Data** tab, in the **Forecast** group, click **What-If Analysis**.
2. Click on **Scenario Manager**
3. The Scenario Manager dialog box appears.
4. Add a scenario by clicking on **Add**.
5. Type a name (60% highest), select cell C4 (% sold for the highest price) for the Changing cells and click on OK.
6. Enter the corresponding value 0.6 and click on OK again.
7. Next, add 4 other scenarios (70%, 80%, 90%, and 100%).

Finally, your Scenario Manager should be consistent with the picture below:

*Insert Screenshot or Image Here*

---

## **Practical 3**

### Task:
Perform the data classification using classification algorithm using R/Python.

### Code (R):
```r
> rainfall <- c(799,1174.8,865.1,1334.6,635.4,918.5,685.5,784.2,985,882.8,1071)
> rainfall.timeseries <- ts(rainfall,start = c(2021,1),frequency = 12) 
> print(rainfall.timeseries)
        Jan    Feb    Mar    Apr    May    Jun    Jul    Aug    Sep    Oct    Nov
2021  799.0 1174.8  865.1 1334.6  635.4  918.5  685.5  784.2  985.0  882.8 1071.0
> png(file = "rainfall.png")
> plot(rainfall.timeseries) 
> dev.off()
> plot(rainfall.timeseries) 
```

### Output:
*Insert plot here*

---

## **Practical 4**

### Task:
Perform the data clustering using clustering algorithm using R/Python.

### Code (R):
```r
> newiris <- iris 
> newiris$Species <- NULL 
> (kc <- kmeans(newiris,3))
> table(iris$Species,kc$cluster)
> plot(newiris[c("Sepal.Length","Sepal.Width")],col=kc$cluster) 
> points(kc$centers[,c("Sepal.Length","Sepal.Width")],col=1:3,pch=8,cex=2)
```

### Output:
*Insert plot here*

---

## **Practical 5**

### Task:
Perform the Linear regression on the given data warehouse data using R/Python.

### Code (R):
```r
> x <- c(151,174,138,186,128,136,179,163,152,131) 
> y <- c(63,81,56,91,47,57,76,72,62,48)
> relation <- lm(y~x)
> png(file = "linearregression.png")
> plot(y,x,col = "blue",main = "Height & Weight Regression",abline(lm(x~y)),cex=1.3, pch=16, xlab="Weight in Kg", ylab="Height in cm") 
> dev.off()
```

### Output:
*Insert regression plot here*

---

## **Practical 6**

### Task:
Perform the logistic regression on the given data warehouse data using R/Python.

### Code (R):
```r
> quality <- read.csv("C:/quality.csv") 
> str(quality)
> table(quality$PoorCare) 
> install.packages("caTools") 
> library(caTools) 
> qualityTrain = subset(quality,split==TRUE) 
> qualityTest = subset(quality,split==FALSE) 
> nrow(qualityTrain)
> nrow(qualityTest)
> QualityLog = glm(PoorCare ~ OfficeVisits + Narcotics, data=qualityTrain, family=binomial) 
> summary(QualityLog)
> predictTrain = predict(QualityLog, type="response") 
> summary(predictTrain)
> tapply(predictTrain, qualityTrain$PoorCare, mean) 
> table(qualityTrain$PoorCare, predictTrain > 0.5) 
> table(qualityTrain$PoorCare, predictTrain > 0.7) 
> table(qualityTrain$PoorCare, predictTrain < 0.2) 
> install.packages("ROCR") 
> library(ROCR)
> ROCRpred = prediction(predictTrain, qualityTrain$PoorCare) 
> ROCRperf = performance(ROCRpred, "tpr", "fpr") 
> plot(ROCRperf)
> plot(ROCRperf, colorize=TRUE) 
> plot(ROCRperf, colorize=TRUE, print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.2,10.7))
```

---

## **Practical 7**

### Task:
Write a Python program to read data from a CSV file, perform simple data analysis, and generate basic insights using Pandas.

### Code (Python):
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'C:\\WineQT.csv'
df = pd.read_csv(file_path)

# Summary statistics
print("Summary Statistics")
print(df.describe())

# Correlation matrix
print("\nCorrelation Matrix: ")
print(df.corr())

# Quality distribution
print("\nQuality Distribution: ")
print(df['quality'].value_counts().sort_index())

# Quality correlation
print("\nCorrelation with Quality:")
print(df.corr()['quality'].sort_values(ascending=False))

# Plotting
plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1)
sns.countplot(x='quality', data=df, palette='viridis')
plt.title('Quality Distribution')

plt.subplot(2, 2, 2)
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')

plt.subplot(2, 2, 3)
sns.boxplot(x='quality', y='alcohol', data=df, palette='viridis')
plt.title('Alcohol vs Quality')

plt.subplot(2, 2, 4)
sns.boxplot(x='quality', y='density', data=df, palette='viridis')
plt.title('Density vs Quality')

plt.tight_layout()
plt.show()
```

---

## **Practical 8**

### Task:
Perform data visualization using Python on any sales data.

### Code (Python):
```python
# Complete code for sales data visualization provided above...
```

*Insert charts: sales trends, sales by region, sales vs quantity, top products*

---

### Also:
**Perform data visualization using Power BI**

#### Key Steps:
- Import data from Excel/CSV
- Transform data using Power Query Editor
- Create visuals: Cards, Line Chart, Map, Pie Chart
- Add slicers & filters
- Create DAX measures:
```DAX
Total Sales = SUM(Sales[Revenue])
Sales Growth = ([Total Sales] - PREVIOUSMONTH([Total Sales])) / PREVIOUSMONTH([Total Sales])
```

---

## **Practical 9**

### Task:
Create the Data staging area for the selected database using SQL & Power BI.

#### Steps:
1. Load Excel data to Power BI using "Get Data".
2. Transform data: rename columns, handle nulls, set data types.
3. Create **FactSales** and **Dimension tables** (Customer, Product, Dates).
4. Define relationships in Model View.
5. Use DAX for validation:
```DAX
Total_Sales = SUM(FactSales[Total_Amount])
Missing_Customers = COUNTROWS(FILTER(FactSales, ISBLANK(FactSales[Customer_ID])))
Duplicate_Orders = COUNTROWS(FactSales) - DISTINCTCOUNT(FactSales[Order_ID])
```

---

## **Practical 10**

### Task:
Create the cube with suitable dimension and fact tables based on **ROLAP**, **MOLAP**, and **HOLAP** model.

#### Tools: SQL Server Analysis Services (SSAS)

#### Key Steps:
- Create a new Analysis Services Project
- Create a Data Source and Data Source View
- Create and define Cube using Fact and Dimension tables
- Process the Cube
- Validate cube deployment in SQL Server Management Studio

---

> *End of Practical Manual*
```

---

Let me know if you want a downloadable `.md` file or would like this split into separate files per practical!
