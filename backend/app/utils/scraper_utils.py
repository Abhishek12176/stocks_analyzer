import re
import numpy as np
import pandas as pd
import requests
from typing import Optional

SUMMARY_LABELS = {
    "promoter": ["promoter", "promoters"],
    "fii": ["fii", "fpi", "foreign institutional"],
    "dii": ["dii", "domestic institutional"],
    "government": ["government", "govt", "president of india"],
    "public": ["public", "others", "retail"],
}

SUMMARY_WORDS = [
    "promoter", "promoters", "fii", "fpi", "dii", "government",
    "govt", "public", "others", "no. of shareholders", "total",
    "total shareholding", "grand total",
]

BAD_HOLDER_WORDS = [
    "sales", "reserves", "fixed assets", "expenses", "borrowings",
    "cash from operating", "other assets", "other liabilities", "cwip",
    "total assets", "net profit", "operating profit", "revenue", "liabilities",
    "equity capital", "cash equivalents", "trade receivables", "inventories",
    "share capital", "balance sheet", "profit loss", "cash flow",
    "promoter and promoter group", "promoters and promoter group",
    "pledged", "locked",
    "individual share capital", "foreign portfolio investors category",
    "clearing members",
]

AGGREGATE_HOLDER_NAMES = {
    "any other",
    "any other (specify)",
    "any others",
    "banks",
    "bodies corporate",
    "foreign portfolio investors category i",
    "foreign portfolio investors category ii",
    "insurance companies",
    "mutual funds",
    "non resident indians",
    "non resident indians (nris)",
    "other financial institutions",
    "trusts",
    "resident individuals",
    "hindu undivided family",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,hi;q=0.8",
    "Referer": "https://www.screener.in/",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


def clean_text(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def parse_percent(value) -> float:
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) if not pd.isna(value) else np.nan
    text = clean_text(value)
    if not text or text.lower() in {"nan", "none", "-", "n/a", "--"}:
        return np.nan
    match = re.search(r"-?\d+(?:,\d+)*(?:\.\d+)?", text.replace("%", ""))
    if not match:
        return np.nan
    return float(match.group(0).replace(",", ""))


def format_percent(value) -> str:
    num = parse_percent(value)
    return "N/A" if pd.isna(num) else f"{num:.2f}%"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            " ".join(clean_text(x) for x in col if clean_text(x))
            for col in df.columns
        ]
    else:
        df.columns = [clean_text(col) for col in df.columns]
    return df


def is_summary_row(name: str) -> bool:
    text = clean_text(name).lower()
    if not text:
        return True
    return any(word in text for word in SUMMARY_WORDS)


def is_valid_holder_name(name: str) -> bool:
    text = clean_text(name).lower()
    text = re.sub(r"\s+", " ", text).strip(" -:|")
    if len(text) < 3:
        return False
    if text in AGGREGATE_HOLDER_NAMES:
        return False
    if any(word in text for word in BAD_HOLDER_WORDS):
        return False
    if re.fullmatch(r"[\d\W_]+", text):
        return False
    if re.search(r"\b(jun|sep|dec|mar)\s+\d{4}\b", text):
        return False
    return True


def fetch_html(url: str) -> Optional[str]:
    session = requests.Session()
    session.headers.update(HEADERS)
    try:
        session.get("https://www.screener.in/", timeout=15)
    except Exception:
        pass
    try:
        response = session.get(url, timeout=25)
        response.raise_for_status()
        return response.text
    except Exception:
        return None


def find_shareholding_table(tables: list) -> Optional[pd.DataFrame]:
    for table in tables:
        if table is None or table.empty or len(table.columns) < 2:
            continue
        table = normalize_columns(table)
        first_col = table.iloc[:, 0].map(clean_text).str.lower()
        has_promoter = first_col.str.contains("promoter", na=False).any()
        has_fii = first_col.str.contains("fii|fpi", regex=True, na=False).any()
        has_dii = first_col.str.contains("dii", na=False).any()
        has_public = first_col.str.contains("public|others", regex=True, na=False).any()
        if has_promoter and has_fii and has_dii and has_public:
            return table
    return None


def build_quarterly_data(table: pd.DataFrame) -> pd.DataFrame:
    table = normalize_columns(table)
    quarters = [clean_text(col) for col in table.columns[1:]]
    data: dict = {"Quarter": quarters}
    row_map: dict = {}
    for idx, label in enumerate(table.iloc[:, 0].tolist()):
        label_text = clean_text(label).lower()
        for key, words in SUMMARY_LABELS.items():
            if key not in row_map and any(word in label_text for word in words):
                row_map[key] = idx
    columns = {
        "promoter": "Promoter (%)",
        "fii": "FII/FPI (%)",
        "dii": "DII (%)",
        "government": "Government (%)",
        "public": "Public (%)",
    }
    for key, col_name in columns.items():
        if key in row_map:
            data[col_name] = [
                format_percent(x) for x in table.iloc[row_map[key], 1:].tolist()
            ]
    return pd.DataFrame(data)


def pick_name_and_holding_columns(df: pd.DataFrame):
    columns = list(df.columns)
    lower_cols = {col: clean_text(col).lower() for col in columns}
    name_candidates = [
        col for col in columns
        if any(word in lower_cols[col] for word in ["name", "shareholder", "holder"])
    ]
    holding_candidates = [
        col for col in columns
        if any(word in lower_cols[col] for word in ["holding", "%", "percent", "share"])
    ]
    if not name_candidates or not holding_candidates:
        return None, None
    name_col = name_candidates[0]
    best_holding_col = None
    best_count = 0
    for col in holding_candidates:
        if col == name_col:
            continue
        values = df[col].map(parse_percent)
        count = values.notna().sum()
        if count > best_count:
            best_count = count
            best_holding_col = col
    return name_col, best_holding_col


def finalize_major_shareholders(rows: list, limit: int = 10):
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["Holding (%)"])
    df = df[(df["Holding (%)"] > 0) & (df["Holding (%)"] <= 100)]
    df = df[df["Shareholder"].map(is_valid_holder_name)]
    df = df[~df["Shareholder"].map(is_summary_row)]
    df = df.drop_duplicates(subset=["Shareholder"], keep="first")
    if df.empty:
        return None
    df = df.sort_values("Holding (%)", ascending=False).head(limit)
    return df.reset_index(drop=True)


def extract_major_shareholders_from_tables(tables: list, limit: int = 10):
    for table in tables:
        if table is None or table.empty or len(table.columns) < 2:
            continue
        df = normalize_columns(table)
        name_col, holding_col = pick_name_and_holding_columns(df)
        if name_col is None or holding_col is None:
            continue
        rows = []
        for _, row in df.iterrows():
            name = clean_text(row[name_col])
            if not is_valid_holder_name(name) or is_summary_row(name):
                continue
            holding = parse_percent(row[holding_col])
            if pd.isna(holding) or holding <= 0 or holding > 100:
                continue
            rows.append({
                "Shareholder": name,
                "Holding (%)": holding,
            })
        result = finalize_major_shareholders(rows, limit=limit)
        if result is not None and not result.empty:
            return result
    return None


def clean_symbol(symbol: str) -> str:
    return symbol.upper().replace(".NS", "").replace(".BO", "").strip()
