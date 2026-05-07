# Retail Insight: Customer Purchase Analysis

Customer purchase analysis project based on transactional retail data. The goal is to clean and process raw sales records, explore purchasing patterns, segment customers with RFM analysis and KMeans, and build a small baseline classifier for high-value customers.

## Tech Stack

- Python
- NumPy
- pandas
- scikit-learn
- matplotlib
- seaborn
- Jupyter Notebook

## Project Structure

```text
customer-purchase-analysis/
├── data/
│   ├── README.md
│   └── Online Retail.xlsx
├── notebooks/
│   └── 01_customer_purchase_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── analysis.py
│   ├── data_processing.py
│   ├── modeling.py
│   └── segmentation.py
├── outputs/
│   ├── figures/
│   └── reports/
├── .gitignore
├── README.md
└── requirements.txt
```

## Dataset

This project uses the Online Retail Dataset from the UCI Machine Learning Repository. The dataset contains transactions for a UK-based online retail store from 2010-12-01 to 2011-12-09.

Place the raw file here:

```text
data/Online Retail.xlsx
```

## Analysis Plan

1. Load the Excel dataset.
2. Clean the transaction table:
   - remove cancelled invoices,
   - remove missing customer IDs,
   - remove invalid quantities and prices,
   - create `TotalPrice`,
   - convert invoice dates.
3. Explore the data:
   - monthly revenue,
   - top-selling products,
   - revenue by country,
   - average order value,
   - share of returning customers.
4. Build RFM features:
   - Recency,
   - Frequency,
   - Monetary value.
5. Segment customers with KMeans.
6. Train a simple classifier for the `High Value` customer segment.

## How To Run

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Or use conda:

```bash
conda create -n customer-purchase-analysis python=3.11
conda activate customer-purchase-analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Open the notebook:

```bash
jupyter notebook notebooks/01_customer_purchase_analysis.ipynb
```
