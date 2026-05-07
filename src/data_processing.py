from pathlib import Path
import pandas as pd

RAW_DATA_PATH = Path("data") / "Online Retail.xlsx"

def load_raw_data(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the Online Retail Excel dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. "
            "Download the Online Retail dataset and place it in the data folder."
        )
    return pd.read_excel(path)


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Clean transaction-level retail data for analysis and modeling."""
    cleaned = df.copy()

    cleaned["InvoiceNo"] = cleaned["InvoiceNo"].astype(str)
    cleaned = cleaned[~cleaned["InvoiceNo"].str.startswith("C")]

    cleaned = cleaned.dropna(subset=["CustomerID", "Description"])
    cleaned = cleaned[(cleaned["Quantity"] > 0) & (cleaned["UnitPrice"] > 0)]

    cleaned["InvoiceDate"] = pd.to_datetime(cleaned["InvoiceDate"])
    cleaned["CustomerID"] = cleaned["CustomerID"].astype(int).astype(str)
    cleaned["Description"] = cleaned["Description"].str.strip()
    cleaned["TotalPrice"] = cleaned["Quantity"] * cleaned["UnitPrice"]
    cleaned["InvoiceMonth"] = cleaned["InvoiceDate"].dt.to_period("M").astype(str)

    return cleaned.reset_index(drop=True)


def load_and_clean(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw data and return the cleaned transaction table."""
    return clean_transactions(load_raw_data(path))
