import pandas as pd
import numpy as np

def load_and_engineer(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["DATE"])
    df = df.rename(columns = {"DATE": "date","revenue": "revenue"})
    df = df.sort_values("date").reset_index(drop=True)

    #The spending columns in Robyn
    spend_cols = ["tv_S", "ooh_S", "print_S", "facebook_S", "search_S", "competitor_sales_B"]

    #Lag features
    for col in ["tv_S","facebook_S", "search_S"]:
        df[f"{col}_lag1"] = df[col].shift(1)
        df[f"{col}_lag2"] = df[col].shift(2)

    #Rolling spend that captures adstock and carryover effect
    df["tv_rolling4"] = df["tv_S"].rolling(4).mean()
    df["fb_rolling4"] = df["facebook_S"].rolling(4).mean()

    total_spend = df[["tv_S", "facebook_S", "search_S", "ooh_S", "print_S"]].sum(axis=1)
    df["tv_sov"] = df["tv_S"] / (total_spend + 1e-9)

    #Seasonality
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month

    df = df.dropna().reset_index(drop=True)
    return df

