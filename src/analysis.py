import numpy as np
import pandas as pd


def monthly_revenue(transactions: pd.DataFrame) -> pd.DataFrame:
    return (
        transactions.groupby("InvoiceMonth", as_index=False)["TotalPrice"]
        .sum()
        .rename(columns={"TotalPrice": "Revenue"})
        .sort_values("InvoiceMonth")
    )


def top_products(transactions: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return (
        transactions.groupby("Description", as_index=False)
        .agg(QuantitySold=("Quantity", "sum"), Revenue=("TotalPrice", "sum"))
        .sort_values(["QuantitySold", "Revenue"], ascending=False)
        .head(n)
    )


def revenue_by_country(transactions: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return (
        transactions.groupby("Country", as_index=False)["TotalPrice"]
        .sum()
        .rename(columns={"TotalPrice": "Revenue"})
        .sort_values("Revenue", ascending=False)
        .head(n)
    )


def order_summary(transactions: pd.DataFrame) -> pd.DataFrame:
    orders = (
        transactions.groupby("InvoiceNo", as_index=False)
        .agg(
            CustomerID=("CustomerID", "first"),
            InvoiceDate=("InvoiceDate", "min"),
            OrderValue=("TotalPrice", "sum"),
            Items=("Quantity", "sum"),
        )
    )
    return orders


def returning_customer_share(transactions: pd.DataFrame) -> float:
    orders_per_customer = transactions.groupby("CustomerID")["InvoiceNo"].nunique()
    return np.mean(orders_per_customer.to_numpy() > 1)
