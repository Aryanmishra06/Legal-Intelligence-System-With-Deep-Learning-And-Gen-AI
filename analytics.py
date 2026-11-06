import pandas as pd
import os

DATA_PATH = "C:/Users/Acer/ARYAN CODE/Clg Project/Court_dataset.csv"

def load_data():
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame(columns=["name", "case_type", "case_category"])
    
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip().str.lower()
    
    # Auto detect columns
    cols = df.columns.tolist()
    name_col = next((c for c in cols if "name" in c or "title" in c), cols[0])
    type_col = next((c for c in cols if "type" in c), cols[1] if len(cols) > 1 else cols[0])
    category_col = next((c for c in cols if "category" in c), cols[2] if len(cols) > 2 else cols[0])
    
    df = df.rename(columns={
        name_col: "name",
        type_col: "case_type",
        category_col: "case_category"
    })
    
    # Fill NaN
    df = df.fillna("")
    
    # Only normalize dataframe columns, NOT the keyword
    df["name_norm"] = df["name"].str.lower().str.replace("-", "/")
    df["case_type_norm"] = df["case_type"].str.lower()
    df["case_category_norm"] = df["case_category"].str.lower()
    
    return df

# -----------------------------
# Search function
# -----------------------------
def search_cases(df, keyword):
    if df.empty:
        return df

    if not keyword or keyword.strip() == "":
        return df.head(20)
    
    # Normalize keyword here
    kw = keyword.lower().replace("-", "/").strip()
    
    results = df[
        df["name_norm"].str.contains(kw, regex=False, na=False) |
        df["case_type_norm"].str.contains(kw, regex=False, na=False) |
        df["case_category_norm"].str.contains(kw, regex=False, na=False)
    ]
    
    return results.head(50)

# -----------------------------
# Stats
# -----------------------------
def get_stats(df):
    total = len(df)
    pending = 0
    closed = 0
    return total, pending, closed

def judge_stats(df):
    return df["case_type"].value_counts().head(5).to_dict()