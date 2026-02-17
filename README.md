# 🛒 Online Retail Data Analysis

> End-to-end exploratory data analysis and business intelligence on a UK-based online retailer's transaction data spanning December 2010 to December 2011.

---

## 📌 Problem Statement

Online retailers accumulate large volumes of transactional data that, without proper analysis, remain underutilised. This project addresses the challenge of transforming raw retail transaction records into actionable business intelligence by answering key questions:

- **How much revenue was generated** over the analysis period, and how did it grow?
- **Which countries, products, and customers** drive the most value?
- **What are the monthly and weekly revenue trends**, and when did the business peak or dip?
- **How recently and frequently** do customers purchase, and what is their monetary contribution?
- **Which stock items** are top sellers by revenue, customer reach, and order volume?

The goal is to support data-driven decision-making across sales, marketing, and inventory management.

---

## 📂 Dataset

| Property | Detail |
|---|---|
| **File** | `online_retail.csv` |
| **Rows** | 541,909 raw transactions |
| **Columns** | 8 |
| **Period** | December 2010 – December 2011 |
| **Source** | UK-based non-store online retail |

### Columns

| Column | Type | Description |
|---|---|---|
| `InvoiceNo` | object | Unique invoice identifier |
| `StockCode` | object | Product/stock identifier |
| `Description` | object | Product name |
| `Quantity` | int64 | Number of units per transaction |
| `InvoiceDate` | object → datetime | Date and time of the invoice |
| `UnitPrice` | float64 | Price per unit (GBP) |
| `CustomerID` | float64 | Unique customer identifier |
| `Country` | object | Country of the customer |

---

## ⚙️ Requirements

```bash
pip install pandas numpy matplotlib seaborn
```

| Library | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, and aggregation |
| `numpy` | Numerical operations |
| `matplotlib` | Plotting charts |
| `seaborn` | Statistical visualisations |

---

## 🔍 Code Explanation

### 1. Data Loading

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("online_retail.csv", encoding='latin1')
df = pd.DataFrame(data)
```

The dataset uses `latin1` encoding to handle special characters in product descriptions. The raw dataset contains **541,909 rows** across 8 columns.

---

### 2. Exploratory Data Analysis (EDA)

```python
df.info()
df.isnull().sum()
df['Country'].value_counts()
```

Initial EDA revealed:
- `Description` had **1,454 null values**
- `CustomerID` had **135,080 null values** (~25% missing)
- All other columns were complete
- The United Kingdom dominates with **495,478** transactions, followed distantly by Germany (9,495) and France (8,557)

---

### 3. Data Cleaning

#### Removing Invalid Quantities
```python
negative_quantity = (df['Quantity'] <= 0).sum()  # Found 10,624 invalid rows
df.drop(df[df['Quantity'] <= 0].index, inplace=True)
```
Rows where Quantity ≤ 0 represent returns or data errors and are removed.

#### Removing Invalid Unit Prices
```python
negative_unit_price = (df['UnitPrice'] <= 0).sum()  # Found 1,181 invalid rows
df.drop(df[df['UnitPrice'] <= 0].index, inplace=True)
```

#### Fixing Missing Descriptions
```python
desc_map = (
    df.dropna(subset=['Description'])
      .drop_duplicates('StockCode')
      .set_index('StockCode')['Description']
)
df['Description'] = df['StockCode'].map(desc_map)
df['Description'] = df['Description'].fillna('Unknown')
```
Missing descriptions are back-filled using the same StockCode from other records. Any remaining unknowns are filled with `'Unknown'`.

#### Removing Rows with Missing CustomerID
```python
df.drop(df[df['CustomerID'].isna()].index, inplace=True)
```
After all cleaning steps, the dataset is reduced to **397,884 clean rows** with zero nulls in any column.

---

### 4. Feature Engineering

```python
# Parse date and time
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
df['date_only'] = df['InvoiceDate'].dt.date
df['time_only'] = df['InvoiceDate'].dt.time

# Create numeric InvoiceNo
df['Invoice_No'] = pd.to_numeric(df['InvoiceNo'], errors='coerce')

# Revenue KPI
df['revenue'] = df['Quantity'] * df['UnitPrice']
```

New columns derived from existing data:
- **`date_only`** / **`time_only`** — split datetime for temporal analysis
- **`Invoice_No`** — numeric version of the invoice identifier
- **`revenue`** — row-level revenue = Quantity × UnitPrice

---

### 5. Revenue Analysis

#### Yearly Revenue
```python
yearly_revenue = df.groupby(df['date_only'].apply(lambda x: x.year))['revenue'].sum()
```
| Year | Revenue (£) |
|---|---|
| 2010 | 572,713.89 |
| 2011 | 8,338,694.01 |

#### Monthly Revenue
```python
monthly_revenue = df.groupby(df['InvoiceDate'].dt.to_period('M'))['revenue'].sum()
```
Revenue peaks in **November 2011 (£1,161,817)** and shows a steep drop in December 2011 (£518,192), likely due to incomplete month data.

#### Month-over-Month (MoM) Growth
```python
mom_growth = monthly_revenue.pct_change() * 100
```

#### Week-over-Week (WoW) Growth
```python
df['year'] = df['InvoiceDate'].dt.isocalendar().year
df['week'] = df['InvoiceDate'].dt.isocalendar().week
weekly_revenue = df.groupby(['year', 'week'])['revenue'].sum()
wow_growth = weekly_revenue.pct_change() * 100
```

Both MoM and WoW growth rates are visualised as line graphs and exported to CSV.

---

### 6. Country-Level Business Analysis

```python
country_summary = (
    df.groupby('Country')
    .agg(
        Total_Revenue=('revenue', 'sum'),
        Unique_Customers=('CustomerID', 'nunique'),
        Unique_Invoices=('InvoiceNo', 'nunique')
    )
    .reset_index()
)
```

Aggregates revenue, unique customers, and unique invoices per country. Results are exported to `Country_Level_Business_Metrics.csv` and visualised as bar charts.

---

### 7. Customer Recency, Frequency & Monetary (RFM) Analysis

```python
customer_metrics = (
    df.groupby(['CustomerID', 'Country'])
    .agg(
        Recency=('InvoiceDate', 'max'),
        Frequency=('InvoiceNo', 'nunique'),
        Monetary=('revenue', 'sum')
    )
    .reset_index()
)
```

RFM metrics per customer allow segmentation of high-value, frequent, and recently active customers. Exported to `Customer_Recency_Frequency_Monetary_By_Country.csv`.

---

### 8. Stock-Level Business Analysis

```python
stock_summary = (
    df.groupby(['StockCode', 'Description'])
    .agg(
        Unique_Customers=('CustomerID', 'nunique'),
        Unique_Invoices=('InvoiceNo', 'nunique'),
        total_quantity=('Quantity', 'sum'),
        Total_Revenue=('revenue', 'sum')
    )
    .reset_index()
)
```

Top 5 products are identified by Revenue, Unique Customers, and Unique Invoices. Visualised as horizontal bar charts and exported to `Stock_Level_Business_Metrics.csv`.

---

### 9. Sales Summary

```python
sales_summary = pd.DataFrame([{
    'Total_Revenue': total_revenue,
    'Total_Customers': df['CustomerID'].nunique(),
    'Total_Invoices': df['InvoiceNo'].nunique(),
    'Total_Quantity': df['Quantity'].sum(),
    'Avg_MOM_Growth': mom_growth.mean(),
    'Avg_WOW_Growth': wow_growth.mean()
}])
```

A single-row summary DataFrame capturing the headline KPIs of the entire analysis period. Exported to `Sales_Summary.csv`.

---

## 📊 Data Insights

| Insight | Value |
|---|---|
| **Total Revenue** (Dec 2010 – Dec 2011) | £8,911,407.90 |
| **Unique Customers** | 4,338 |
| **Unique Invoices** | 18,532 |
| **Total Quantity Sold** | 5,167,812 units |
| **Avg. Month-over-Month Revenue Growth** | ~3.34% |
| **Avg. Week-over-Week Revenue Growth** | ~7.80% |

### 📅 Temporal Trends
- **Peak Revenue Month:** November 2011 — £1,161,817
- **Steepest Decline:** December 2011 (likely due to partial month data)
- Revenue grew dramatically from 2010 (£572K) to 2011 (£8.3M), reflecting strong business scaling

### 🌍 Country-Level
- **United Kingdom** ranks #1 in total revenue, number of customers, and number of invoices — by a very large margin
- Netherlands, EIRE (Ireland), Germany, and France follow as the top international markets

### 📦 Stock-Level
- **Paper Craft, Little Birdie** is the top revenue-generating stock item (£168,469), but it was purchased by only **1 customer** in very large quantities — a bulk/wholesale outlier
- **Regency Cakestand 3 Tier** has the highest number of unique customers (881), making it the most broadly popular product
- **White Hanging Heart T-Light Holder** has the highest number of unique invoices (1,978), indicating it appears most frequently across orders

### 👤 Customer Behaviour
- The dataset spans customers from 38+ countries
- RFM analysis reveals a strong long-tail distribution — a small number of customers contribute disproportionately high revenue (Pareto principle applies)

---

## 📁 Output Files

| File | Description |
|---|---|
| `Retail.csv` | Cleaned and feature-enriched transaction dataset |
| `Sales_Summary.csv` | Headline KPIs for the full period |
| `Country_Level_Business_Metrics.csv` | Revenue, customers, and invoices by country |
| `Customer_Recency_Frequency_Monetary_By_Country.csv` | RFM metrics per customer |
| `Stock_Level_Business_Metrics.csv` | Performance metrics per stock item |

---

## 🚀 How to Run

1. Clone or download the repository
2. Place `online_retail.csv` in the same directory as the notebook
3. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn
   ```
4. Open `Retail_Data.ipynb` in Jupyter Notebook or Google Colab
5. Run all cells sequentially

---

## 📝 Notes

- Transactions with `Quantity ≤ 0` (returns/cancellations) and `UnitPrice ≤ 0` (free or erroneous items) are excluded from all analysis
- Rows with missing `CustomerID` are dropped since customer-level analysis requires valid identifiers
- The December 2011 revenue dip is expected — the data ends mid-month (9th December), so it does not represent a full month

---

*Analysis performed using Python (Pandas, NumPy, Matplotlib, Seaborn) on Google Colab.*
