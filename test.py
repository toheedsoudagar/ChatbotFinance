# chatbot_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
from dotenv import load_dotenv
from typing import List, Optional, Tuple

# Optional fuzzy matching (rapidfuzz). Install if you want fuzzy fallback:
# pip install rapidfuzz
try:
    from rapidfuzz import process, fuzz
    _HAS_FUZZ = True
except Exception:
    _HAS_FUZZ = False

# Optional Gemini integration (only used for free-text fallback summaries)
try:
    import google.generativeai as genai
except Exception:
    genai = None

# --- Config ---
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL = None
if API_KEY and genai:
    try:
        genai.configure(api_key=API_KEY)
        MODEL = genai.GenerativeModel("gemini-2.0-flash")
    except Exception:
        MODEL = None

# --- Paths (edit if needed) ---
DATA_PATH = "data/Revenue File.xlsx"
PARTY_PATH = "/mnt/data/PartyName.txt"
LEDGER_PATH = "/mnt/data/Ledger Name.txt"

# --- Utilities: normalization / text helpers ---
def normalize_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def extract_year_from_query(q: str) -> Optional[int]:
    m = re.search(r"\b(20\d{2})\b", q)
    return int(m.group(1)) if m else None

def extract_monthname_and_year(q: str) -> Optional[Tuple[int,int]]:
    month_names = {
        "january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
        "july":7,"august":8,"september":9,"october":10,"november":11,"december":12
    }
    month_match = next((m for m in month_names if m in q), None)
    year = extract_year_from_query(q)
    if month_match and year:
        return month_names[month_match], year
    return None

# --- Load dataset ---
@st.cache_data(ttl=600)
def load_data(file_path: str = DATA_PATH, sheet_name: str = "Revenue") -> pd.DataFrame:
    """Load Excel and normalize columns/ types. Returns empty DataFrame if file missing."""
    if not os.path.exists(file_path):
        return pd.DataFrame()
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    except Exception:
        try:
            df = pd.read_excel(file_path)
        except Exception:
            return pd.DataFrame()
    # normalize column names
    df.columns = [normalize_text(c).replace(" ", "_") for c in df.columns.astype(str)]
    df = df.dropna(how="all").copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    # ensure strings
    for c in df.select_dtypes(["object"]).columns:
        df[c] = df[c].astype(str).str.strip()
    # add normalized party column if partyname exists
    if "partyname" in df.columns:
        df["_party_norm"] = df["partyname"].fillna("").map(normalize_text)
    return df

@st.cache_data(ttl=600)
def load_list_from_file(path: str, df_col: Optional[pd.Series] = None) -> List[str]:
    """Load values from text file if present, else derive from a dataframe column series."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            uniq = list(dict.fromkeys(lines))
            return uniq
        except Exception:
            pass
    if df_col is not None:
        return df_col.dropna().astype(str).unique().tolist()
    return []

# --- Business masks ---
def revenue_mask(df: pd.DataFrame) -> pd.Series:
    """
    Revenue rows: vouchertype == 'Receipt', amount_type == 'Cr', amount > 0
    If columns missing, returns all-False mask.
    """
    if df.empty:
        return pd.Series([False] * 0, index=df.index)
    if not all(c in df.columns for c in ("vouchertype", "amount_type", "amount")):
        return pd.Series([False] * len(df), index=df.index)
    vt = df["vouchertype"].fillna("").str.lower().str.strip()
    at = df["amount_type"].fillna("").str.lower().str.strip()
    amt = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    return (vt == "receipt") & (at == "cr") & (amt > 0)

def payments_mask(df: pd.DataFrame) -> pd.Series:
    if df.empty or "vouchertype" not in df.columns or "amount_type" not in df.columns or "amount" not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    vt = df["vouchertype"].fillna("").str.lower().str.strip()
    at = df["amount_type"].fillna("").str.lower().str.strip()
    amt = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    return (vt == "payment") & (at == "dr") & (amt > 0)

# --- Party & Ledger matching helpers (STRICT whole-word matching + fallback) ---
def ensure_party_norm_in_df(df: pd.DataFrame) -> pd.DataFrame:
    if "partyname" in df.columns and "_party_norm" not in df.columns:
        df["_party_norm"] = df["partyname"].fillna("").map(normalize_text)
    return df

def find_party_matches(
    df: pd.DataFrame,
    query: str,
    top_n: int = 10,
    fuzz_threshold: int = 80,
    strict: bool = True,
    require_all_tokens: bool = True
) -> List[Tuple[str, float]]:
    """
    Return list of (original_partyname, score).
    Strict mode uses whole-word token matches (recommended). Fallback to loose/fuzzy if needed.
    """
    q = normalize_text(query)
    if not q:
        return []
    df = ensure_party_norm_in_df(df)
    matches: List[Tuple[str, float]] = []
    tokens = [t for t in re.findall(r"\w+", q) if len(t) > 2]

    if strict and tokens:
        if require_all_tokens:
            pattern = r"(?=.*" + r")(?=.*".join(rf"\b{re.escape(tok)}\b" for tok in tokens) + r")"
            full_re = re.compile(pattern, flags=re.IGNORECASE)
            mask = df["_party_norm"].apply(lambda s: bool(full_re.search(s)))
        else:
            token_pattern = r"|".join(rf"\b{re.escape(tok)}\b" for tok in tokens)
            full_re = re.compile(token_pattern, flags=re.IGNORECASE)
            mask = df["_party_norm"].apply(lambda s: bool(full_re.search(s)))

        if mask.any():
            for name in df.loc[mask, "partyname"].unique():
                matches.append((name, 100.0))
            return matches[:top_n]

    # loose substring match
    mask_sub = df["_party_norm"].str.contains(re.escape(q), case=False, na=False)
    if mask_sub.any():
        for name in df.loc[mask_sub, "partyname"].unique():
            matches.append((name, 100.0))
        return matches[:top_n]

    # token presence
    if tokens:
        token_mask = df["_party_norm"].apply(lambda s: all(tok in s for tok in tokens))
        if token_mask.any():
            for name in df.loc[token_mask, "partyname"].unique():
                matches.append((name, 95.0))
            return matches[:top_n]

    # fuzzy fallback
    if _HAS_FUZZ:
        candidates = list(df["partyname"].dropna().astype(str).unique())
        results = process.extract(q, candidates, scorer=fuzz.WRatio, limit=top_n)
        filtered = [(r[0], float(r[1])) for r in results if r[1] >= fuzz_threshold]
        if filtered:
            return filtered

    return []

def find_group_from_query_using_list(
    query: str,
    value_list: List[str],
    strict: bool = True,
    require_all_tokens: bool = True
) -> List[str]:
    """
    Best-effort: return list entries from value_list that match the query.
    Strict=whole-word matching; strict=False uses substring search.
    """
    q = normalize_text(query)
    tokens = [t for t in re.findall(r"\w+", q) if len(t) > 2]
    if not tokens:
        return []
    value_norms = [(p, normalize_text(p)) for p in value_list]
    matched = set()

    if strict:
        if require_all_tokens:
            pattern = r"(?=.*" + r")(?=.*".join(rf"\b{re.escape(tok)}\b" for tok in tokens) + r")"
            combined_re = re.compile(pattern, flags=re.IGNORECASE)
            for original, norm in value_norms:
                if combined_re.search(norm):
                    matched.add(original)
        else:
            token_pattern = r"|".join(rf"\b{re.escape(tok)}\b" for tok in tokens)
            token_re = re.compile(token_pattern, flags=re.IGNORECASE)
            for original, norm in value_norms:
                if token_re.search(norm):
                    matched.add(original)
    else:
        for t in tokens:
            for original, norm in value_norms:
                if t in norm:
                    matched.add(original)

    return list(matched)

# --- Aggregations & presentation helpers ---
def compute_total_revenue(df_local: pd.DataFrame) -> float:
    if df_local.empty:
        return 0.0
    return df_local.loc[revenue_mask(df_local), "amount"].sum()

def compute_revenue_by_month(df_local: pd.DataFrame) -> pd.DataFrame:
    rev = df_local.loc[revenue_mask(df_local)].copy()
    if "date" not in rev.columns or rev.empty:
        return pd.DataFrame(columns=["month", "value"])
    grouped = rev.groupby(rev["date"].dt.to_period("M"))["amount"].sum().reset_index()
    grouped["month"] = grouped["date"].astype(str)
    grouped = grouped.rename(columns={"amount":"value"})[["month","value"]]
    grouped = grouped.sort_values("month").reset_index(drop=True)
    return grouped

def compute_revenue_by_year(df_local: pd.DataFrame) -> pd.DataFrame:
    rev = df_local.loc[revenue_mask(df_local)].copy()
    if "date" not in rev.columns or rev.empty:
        return pd.DataFrame(columns=["year", "value"])
    grouped = rev.groupby(rev["date"].dt.year)["amount"].sum().reset_index()
    grouped.columns = ["year","value"]
    grouped["year"] = grouped["year"].astype(int)
    grouped = grouped.sort_values("year").reset_index(drop=True)
    return grouped

def compute_revenue_by_month_for_year(df_local: pd.DataFrame, year: int, fill_empty_months: bool = True) -> pd.DataFrame:
    rev = df_local.loc[revenue_mask(df_local)].copy()
    if "date" not in rev.columns:
        return pd.DataFrame(columns=["month","value"])
    revy = rev[rev["date"].dt.year == year]
    month_map = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    if revy.empty:
        if fill_empty_months:
            return pd.DataFrame({"month": list(month_map.values()), "value":[0]*12})
        return pd.DataFrame(columns=["month","value"])
    month_grp = revy.groupby(revy["date"].dt.month)["amount"].sum().reset_index()
    month_grp.columns = ["month_num","value"]
    month_grp["month"] = month_grp["month_num"].map(month_map)
    if fill_empty_months:
        df_all = pd.DataFrame({"month_num": list(range(1,13))})
        df_all = df_all.merge(month_grp, on="month_num", how="left").fillna({"value":0})
        df_all["month"] = df_all["month_num"].map(month_map)
        return df_all[["month","value"]]
    return month_grp[["month","value"]]

def format_currency(val: float) -> str:
    try:
        return f"₹{val:,.0f}"
    except Exception:
        return f"₹{float(val):,.2f}"

def plot_chart(df_plot: pd.DataFrame, x_col: str, y_col: str, title: str, kind: str = "bar"):
    if df_plot.empty:
        st.write("No data to plot.")
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    plt.tight_layout(pad=3)
    x_vals = df_plot[x_col].astype(str).tolist()
    y_vals = df_plot[y_col].tolist()
    if kind == "bar":
        bars = ax.bar(x_vals, y_vals, color="#4CAF50", alpha=0.95)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(format_currency(h), xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")
    else:
        ax.plot(x_vals, y_vals, marker="o", color="#2196F3", linewidth=2)
        for i, v in enumerate(y_vals):
            ax.text(i, v, format_currency(v), ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xlabel(x_col.capitalize(), fontsize=10)
    ax.set_ylabel("Amount (₹)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    st.pyplot(fig)

def build_summary(grouped: pd.DataFrame, label_kind: str) -> str:
    if grouped.empty:
        return f"No data available for {label_kind}."
    total_rev = grouped["value"].sum()
    idx = grouped["value"].idxmax()
    max_label = str(grouped.iloc[idx, 0])
    max_val = grouped.iloc[idx]["value"]
    if label_kind == "year":
        summary = (f"📅 **Year-wise Summary:**\n"
                   f"• Total revenue across all years: {format_currency(total_rev)}\n"
                   f"• Highest in {max_label}: {format_currency(max_val)}")
    else:
        summary = (f"📆 **Month-wise Summary:**\n"
                   f"• Total revenue across selected months: {format_currency(total_rev)}\n"
                   f"• Highest in {max_label}: {format_currency(max_val)}")
    return summary

# --- NEW: direct data query handler (answers simple requests via pandas) ---
def handle_direct_data_queries(df: pd.DataFrame, q: str) -> Optional[str]:
    """
    Detect and answer questions like:
      - "how many party name total"
      - "count unique ledger_name"
      - "what are the different voucher types"
      - "show columns"
    Returns a text response if handled, else None.
    """
    if df is None or df.empty:
        return None
    ql = q.lower()
    # quick "show columns"
    if re.search(r"\b(columns|show columns|list columns)\b", ql):
        cols = df.columns.tolist()
        return "Columns: " + ", ".join(cols)
    # mapping of common aliases to actual column names (extend if your sheet uses different names)
    possible_columns = {
        "partyname": ["party name", "partyname", "party"],
        "ledger_name": ["ledger name", "ledger_name", "ledger", "ledgername"],
        "vouchertype": ["voucher type", "voucher_type", "vouchertype"],
        "amount_type": ["amount type", "amount_type"],
        "date": ["date", "month", "year", "transaction date"]
    }
    # reverse lookup: detect which column user refers to
    detected_col = None
    for col, aliases in possible_columns.items():
        for a in aliases:
            if a in ql:
                if col in df.columns:
                    detected_col = col
                    break
        if detected_col:
            break
    # handle requests
    if detected_col:
        if re.search(r"\b(how many|count|number of|how many different|total)\b", ql):
            cnt = int(df[detected_col].nunique(dropna=True))
            return f"There are **{cnt} unique values** in the `{detected_col}` column."
        if re.search(r"\b(list|show|what are|different|distinct)\b", ql):
            uniques = df[detected_col].dropna().unique().tolist()
            sample = ", ".join(map(str, uniques[:20]))
            more = f" (+{len(uniques)-20} more)" if len(uniques) > 20 else ""
            return f"The `{detected_col}` column has {len(uniques)} unique values. Examples: {sample}{more}"
    if "unique" in ql or "distinct" in ql or "different" in ql:
        for col in df.columns:
            if col in ql:
                cnt = int(df[col].nunique(dropna=True))
                return f"There are **{cnt} unique values** in the `{col}` column."
    return None

# --- NEW: top-ledgers handler (answers "top N ledgers" queries) ---
def handle_top_ledgers_query(df: pd.DataFrame, q: str) -> Optional[str]:
    """
    Detect queries like 'top N ledgers' or 'highest revenue ledgers' and compute results.
    Returns text summary; stores last_grouped_df for optional plotting.
    """
    if df.empty or "ledger_name" not in df.columns:
        return None
    ql = q.lower()
    # detect "top N" otherwise default to 3
    m = re.search(r"top\s*(\d+)", ql)
    top_n = int(m.group(1)) if m else 3
    if "ledger" not in ql and "ledgers" not in ql:
        return None
    # apply revenue mask
    rev_mask = revenue_mask(df)
    df_rev = df.loc[rev_mask & df["ledger_name"].notna()].copy()
    if df_rev.empty:
        return "No revenue data found for ledgers."
    # restrict to this year if user asks about this year/current year/financial year
    if "this year" in ql or "current year" in ql or "financial year" in ql or "this financial year" in ql:
        year_now = pd.Timestamp.now().year
        df_rev = df_rev[df_rev["date"].dt.year == year_now]
    # group and sort
    grouped = (
        df_rev.groupby("ledger_name")["amount"]
        .sum()
        .reset_index()
        .rename(columns={"amount": "revenue"})
        .sort_values("revenue", ascending=False)
        .reset_index(drop=True)
    )
    if grouped.empty:
        return "No ledger revenue found for the requested period."
    top_df = grouped.head(top_n)
    # summary lines
    lines = []
    for i, row in top_df.iterrows():
        lines.append(f"**{i+1}. {row['ledger_name']}** — {format_currency(row['revenue'])}")
    summary_text = f"📊 **Top {top_n} Ledgers by Revenue:**\n" + "\n".join(lines)
    # persist for optional plotting
    st.session_state.last_grouped_df = top_df.rename(columns={"ledger_name": "Ledger", "revenue":"value"})[["Ledger","value"]].copy()
    st.session_state.last_grouped_title = f"Top {top_n} Ledgers by Revenue"
    st.session_state.last_grouped_kind = "bar"
    return summary_text

# --- NEW: top-party contributors handler (by % threshold or top N) ---
def handle_top_party_contributors(df: pd.DataFrame, q: str) -> Optional[str]:
    """
    Detect queries asking for parties contributing over a % of total revenue
    or top N contributors by revenue.
    """
    if df.empty or "partyname" not in df.columns:
        return None
    ql = q.lower()
    if "party" not in ql and "parties" not in ql and "partyname" not in ql:
        return None
    # detect percentage threshold like 'over 5%' or 'greater than 5 percent'
    m = re.search(r"(\d+(?:\.\d+)?)\s*%|\b(\d+(?:\.\d+)?)\s*percent", ql)
    threshold = float(m.group(1) or m.group(2)) if m else None
    # detect top N
    n_match = re.search(r"top\s*(\d+)", ql)
    top_n = int(n_match.group(1)) if n_match else None

    rev_mask = revenue_mask(df)
    df_rev = df.loc[rev_mask & df["partyname"].notna()].copy()
    if df_rev.empty:
        return "No revenue data found for parties."

    grouped = (
        df_rev.groupby("partyname")["amount"]
        .sum()
        .reset_index()
        .rename(columns={"amount": "revenue"})
        .sort_values("revenue", ascending=False)
        .reset_index(drop=True)
    )
    total = grouped["revenue"].sum()
    if total == 0:
        return "Total revenue is zero; cannot compute contributions."

    grouped["pct"] = (grouped["revenue"] / total * 100).round(2)

    if threshold:
        subset = grouped[grouped["pct"] >= threshold]
        if subset.empty:
            return f"No parties contribute over {threshold}% of total revenue."
        title = f"Parties contributing ≥ {threshold}% of total revenue"
    elif top_n:
        subset = grouped.head(top_n)
        title = f"Top {top_n} Party Contributors by Revenue"
    else:
        subset = grouped.head(5)
        title = "Top 5 Party Contributors by Revenue"

    lines = [
        f"**{i+1}. {row['partyname']}** — {format_currency(row['revenue'])} ({row['pct']}%)"
        for i, row in subset.iterrows()
    ]
    summary_text = f"📊 **{title}:**\n" + "\n".join(lines)

    # persist for optional plotting
    st.session_state.last_grouped_df = subset.rename(columns={"partyname": "Party", "revenue": "value"})[["Party","value"]].copy()
    st.session_state.last_grouped_title = title
    st.session_state.last_grouped_kind = "bar"
    return summary_text

# --- Load data & lists ---
df = load_data(DATA_PATH)
party_list = load_list_from_file(PARTY_PATH, df["partyname"] if "partyname" in df.columns else None)
ledger_list = load_list_from_file(LEDGER_PATH, df["ledger_name"] if "ledger_name" in df.columns else None)
df = ensure_party_norm_in_df(df)

# --- Streamlit UI setup ---
st.set_page_config(page_title="Financial Data Chatbot", page_icon="💬", layout="centered")
st.title("Financial Data Chatbot ")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_grouped_df" not in st.session_state:
    st.session_state.last_grouped_df = None
if "last_grouped_title" not in st.session_state:
    st.session_state.last_grouped_title = None
if "last_grouped_kind" not in st.session_state:
    st.session_state.last_grouped_kind = "bar"

# show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

def push_model(content: str, display_as_markdown: bool = True):
    with st.chat_message("model"):
        if display_as_markdown:
            st.markdown(content)
        else:
            st.write(content)
    st.session_state.messages.append({"role":"model","content":content})

# ---------- Chat input ----------
user_input = st.chat_input("Ask about your financial data...")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role":"user","content":user_input})
    q = user_input.lower().strip()

    viz_terms = {"plot", "chart", "barchart", "bar chart", "line", "graph", "visual", "visualize", "show chart", "show graph"}
    wants_chart = any(t in q for t in viz_terms)

    # quick follow-up: "plot" to render last grouped result
    if q in {"plot", "show chart", "show", "visualize", "chart", "plot chart"}:
        last_df = st.session_state.get("last_grouped_df")
        last_title = st.session_state.get("last_grouped_title")
        last_kind = st.session_state.get("last_grouped_kind", "bar")
        if last_df is not None:
            push_model(f"📈 Showing chart for: **{last_title}**")
            plot_chart(last_df, last_df.columns[0], "value", last_title, kind=last_kind)
        else:
            push_model("No recent result available to plot. Ask a question like 'Year wise revenue' or 'Top 3 ledgers' first.")
        st.session_state.messages.append({"role":"model","content":"(plotted or failed)"})
        handled = True
    else:
        handled = False

    # ---------- PRIORITY: top-ledgers (run before direct data queries) ----------
    if not handled:
        top_ledger_resp = handle_top_ledgers_query(df, q)
        if top_ledger_resp:
            push_model(top_ledger_resp)
            if wants_chart and st.session_state.last_grouped_df is not None:
                plot_chart(st.session_state.last_grouped_df, st.session_state.last_grouped_df.columns[0], "value", st.session_state.last_grouped_title)
            handled = True

    # ---------- PRIORITY: top-party contributors (run before direct data queries) ----------
    if not handled:
        top_party_resp = handle_top_party_contributors(df, q)
        if top_party_resp:
            push_model(top_party_resp)
            if wants_chart and st.session_state.last_grouped_df is not None:
                plot_chart(st.session_state.last_grouped_df, st.session_state.last_grouped_df.columns[0], "value", st.session_state.last_grouped_title)
            handled = True

    # ---------- DIRECT DATA QUERIES (unique/list of columns etc.) ----------
    if not handled:
        direct_resp = handle_direct_data_queries(df, q)
        if direct_resp:
            push_model(direct_resp)
            handled = True

    # ---------- Specific month + year (e.g., April 2020) ----------
    if not handled:
        my = extract_monthname_and_year(q)
        if my:
            month_num, year_val = my
            if "date" in df.columns:
                df_f = df[(df["date"].dt.year == year_val) & (df["date"].dt.month == month_num)]
                total = df_f.loc[revenue_mask(df_f), "amount"].sum()
            else:
                total = 0.0
            month_label = [k.capitalize() for k,v in {
                "january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
                "july":7,"august":8,"september":9,"october":10,"november":11,"december":12
            }.items() if v == month_num][0]
            resp_text = f"📆 **Revenue for {month_label} {year_val}:** {format_currency(total)}"
            push_model(resp_text)
            chart_df = pd.DataFrame({"month":[f"{month_label} {year_val}"], "value":[total]})
            st.session_state.last_grouped_df = chart_df
            st.session_state.last_grouped_title = f"Revenue for {month_label} {year_val}"
            st.session_state.last_grouped_kind = "bar"
            handled = True

    # ---------- Yearwise/monthwise overall or with plot intent ----------
    if not handled:
        wants_year = bool(re.search(r"\b(yearwise|year\s*wise|year|annual|annually)\b", q))
        wants_month = bool(re.search(r"\b(monthwise|month\s*wise|month|monthly)\b", q))
        if wants_chart and wants_year:
            ydf = compute_revenue_by_year(df)
            resp_summary = build_summary(ydf, "year")
            push_model(resp_summary)
            st.session_state.last_grouped_df = ydf.copy()
            st.session_state.last_grouped_title = "Year-wise Revenue (All Data)"
            st.session_state.last_grouped_kind = "bar"
            if wants_chart:
                plot_chart(ydf, "year", "value", "Year-wise Revenue (All Data)", kind="bar")
            handled = True
        elif wants_chart and wants_month:
            mdf = compute_revenue_by_month(df)
            resp_summary = build_summary(mdf, "month")
            push_model(resp_summary)
            st.session_state.last_grouped_df = mdf.copy()
            st.session_state.last_grouped_title = "Monthly Revenue (All Data)"
            st.session_state.last_grouped_kind = "bar"
            if wants_chart:
                plot_chart(mdf, "month", "value", "Monthly Revenue (All Data)", kind="bar")
            handled = True

    # ---------- Generic total revenue queries (no category) ----------
    if not handled:
        if re.search(r"\b(total\s+revenue|what\s+is\s+the\s+total\s+revenue|total\s+revenue\s+this\s+month|total\s+revenue\s+in\s+\d{4})\b", q) \
           or ("revenue" in q and ("total" in q or "this month" in q or re.search(r"\b\d{4}\b", q)) and not re.search(r"\b(hostel|mess|canteen|coffee|cantein|coffee lab|party|ledger)\b", q)):
            if "this month" in q and "date" in df.columns:
                now = pd.Timestamp.now()
                dfm = df[(df["date"].dt.year == now.year) & (df["date"].dt.month == now.month)]
                val = dfm.loc[revenue_mask(dfm), "amount"].sum()
                resp_text = f"Total revenue for this month ({now.strftime('%Y-%m')}): {format_currency(val)}"
                push_model(resp_text)
                st.session_state.last_grouped_df = pd.DataFrame({"month":[now.strftime('%Y-%m')],"value":[val]})
                st.session_state.last_grouped_title = f"Revenue for {now.strftime('%Y-%m')}"
                st.session_state.last_grouped_kind = "bar"
                handled = True
            elif re.search(r"\b(20\d{2})\b", q):
                match = re.search(r"\b(20\d{2})\b", q)
                year = int(match.group(1))
                if "date" in df.columns:
                    dfy = df[df["date"].dt.year == year]
                    val = dfy.loc[revenue_mask(dfy), "amount"].sum()
                else:
                    val = 0.0
                resp_text = f"Total revenue in {year}: {format_currency(val)}"
                push_model(resp_text)
                monthly = compute_revenue_by_month_for_year(df, year, fill_empty_months=True)
                st.session_state.last_grouped_df = monthly.copy()
                st.session_state.last_grouped_title = f"Monthly Revenue – {year}"
                st.session_state.last_grouped_kind = "bar"
                handled = True
            else:
                total = compute_total_revenue(df)
                resp_text = f"Total revenue (all data): {format_currency(total)}"
                push_model(resp_text)
                st.session_state.last_grouped_df = None
                handled = True

    # ---------- Category / party / ledger-based revenue (STRICT matching used) ----------
    if not handled and "revenue" in q:
        # First check ledger matches if the question mentions ledger
        matched_ledgers = []
        if "ledger" in q or "ledger_name" in q or "ledgername" in q:
            matched_ledgers = find_group_from_query_using_list(q, ledger_list, strict=True, require_all_tokens=True)
            # fallback to strict fuzzy on df if none
            if not matched_ledgers:
                if "ledger_name" in df.columns:
                    # use party-matching helper by temporarily renaming df column to reuse function
                    temp_df = df.copy()
                    temp_df["partyname"] = temp_df["ledger_name"]
                    temp_df["_party_norm"] = temp_df["partyname"].map(normalize_text)
                    fuzzy_matches = find_party_matches(temp_df, q, top_n=8, fuzz_threshold=80, strict=True, require_all_tokens=True)
                    if fuzzy_matches:
                        matched_ledgers = [m[0] for m in fuzzy_matches]
            # relax if still nothing
            if not matched_ledgers:
                matched_ledgers = find_group_from_query_using_list(q, ledger_list, strict=True, require_all_tokens=False)

        if matched_ledgers:
            # compute revenue by ledger_name
            mask = df["ledger_name"].isin(matched_ledgers) & revenue_mask(df)
            year = extract_year_from_query(q)
            if year and "date" in df.columns:
                mask &= df["date"].dt.year == year
            subset = df.loc[mask].copy()
            total = subset["amount"].sum() if not subset.empty else 0.0
            ledger_label = ", ".join(matched_ledgers[:5]) + ("" if len(matched_ledgers) <= 5 else f" (+{len(matched_ledgers)-5} more)")
            if year:
                resp_text = f"📊 **Total revenue for ledger(s) {ledger_label} in {year}:** {format_currency(total)}"
            else:
                resp_text = f"📊 **Total revenue for ledger(s) {ledger_label} (all data):** {format_currency(total)}"
            push_model(resp_text)
            by_year = subset.groupby(subset["date"].dt.year)["amount"].sum().reset_index().rename(columns={"date":"year","amount":"value"}) if (not subset.empty and "date" in subset.columns) else pd.DataFrame(columns=["year","value"])
            by_month = subset.groupby(subset["date"].dt.to_period("M"))["amount"].sum().reset_index().rename(columns={"date":"month","amount":"value"}) if (not subset.empty and "date" in subset.columns) else pd.DataFrame(columns=["month","value"])
            if not by_month.empty:
                by_month["month"] = by_month["month"].astype(str)
            if not by_month.empty:
                st.session_state.last_grouped_df = by_month[["month","value"]].copy()
                st.session_state.last_grouped_title = f"Monthly Revenue – {ledger_label}"
                st.session_state.last_grouped_kind = "bar"
                if wants_chart:
                    push_model(build_summary(by_month, "month"))
                    plot_chart(by_month[["month","value"]], "month", "value", st.session_state.last_grouped_title, kind="bar")
            elif not by_year.empty:
                st.session_state.last_grouped_df = by_year.copy()
                st.session_state.last_grouped_title = f"Year-wise Revenue – {ledger_label}"
                st.session_state.last_grouped_kind = "bar"
                if wants_chart:
                    push_model(build_summary(by_year, "year"))
                    plot_chart(by_year[["year","value"]], "year", "value", st.session_state.last_grouped_title, kind="bar")
            else:
                st.session_state.last_grouped_df = pd.DataFrame({"value":[total]})
                st.session_state.last_grouped_title = f"Revenue – {ledger_label}"
                st.session_state.last_grouped_kind = "bar"
            handled = True

        else:
            # fallback to party matching (strict)
            matched_parties = find_group_from_query_using_list(q, party_list, strict=True, require_all_tokens=True)
            if not matched_parties:
                fuzzy_matches = find_party_matches(df, q, top_n=8, fuzz_threshold=80, strict=True, require_all_tokens=True)
                if fuzzy_matches:
                    matched_parties = [m[0] for m in fuzzy_matches]
            if not matched_parties:
                matched_parties = find_group_from_query_using_list(q, party_list, strict=True, require_all_tokens=False)
            if not matched_parties:
                fuzzy_matches = find_party_matches(df, q, top_n=8, fuzz_threshold=75, strict=False, require_all_tokens=True)
                if fuzzy_matches:
                    matched_parties = [m[0] for m in fuzzy_matches]

            if not matched_parties:
                push_model("⚠️ No party, ledger, or category match found in your query. Try phrasing like 'canteen revenue 2023' or provide the party/ledger name.")
                handled = True
            else:
                mask = df["partyname"].isin(matched_parties) & revenue_mask(df)
                year = extract_year_from_query(q)
                if year and "date" in df.columns:
                    mask &= df["date"].dt.year == year
                subset = df.loc[mask].copy()
                total = subset["amount"].sum() if not subset.empty else 0.0
                party_label = ", ".join(matched_parties[:5]) + ("" if len(matched_parties) <= 5 else f" (+{len(matched_parties)-5} more)")
                if year:
                    resp_text = f"📊 **Total revenue for {party_label} in {year}:** {format_currency(total)}"
                else:
                    resp_text = f"📊 **Total revenue for {party_label} (all data):** {format_currency(total)}"
                push_model(resp_text)
                by_year = subset.groupby(subset["date"].dt.year)["amount"].sum().reset_index().rename(columns={"date":"year","amount":"value"}) if (not subset.empty and "date" in subset.columns) else pd.DataFrame(columns=["year","value"])
                by_month = subset.groupby(subset["date"].dt.to_period("M"))["amount"].sum().reset_index().rename(columns={"date":"month","amount":"value"}) if (not subset.empty and "date" in subset.columns) else pd.DataFrame(columns=["month","value"])
                if not by_month.empty:
                    by_month["month"] = by_month["month"].astype(str)
                if not by_month.empty:
                    st.session_state.last_grouped_df = by_month[["month","value"]].copy()
                    st.session_state.last_grouped_title = f"Monthly Revenue – {party_label}"
                    st.session_state.last_grouped_kind = "bar"
                    if wants_chart:
                        push_model(build_summary(by_month, "month"))
                        plot_chart(by_month[["month","value"]], "month", "value", st.session_state.last_grouped_title, kind="bar")
                elif not by_year.empty:
                    st.session_state.last_grouped_df = by_year.copy()
                    st.session_state.last_grouped_title = f"Year-wise Revenue – {party_label}"
                    st.session_state.last_grouped_kind = "bar"
                    if wants_chart:
                        push_model(build_summary(by_year, "year"))
                        plot_chart(by_year[["year","value"]], "year", "value", st.session_state.last_grouped_title, kind="bar")
                else:
                    st.session_state.last_grouped_df = pd.DataFrame({"value":[total]})
                    st.session_state.last_grouped_title = f"Revenue – {party_label}"
                    st.session_state.last_grouped_kind = "bar"
                handled = True

    # ---------- Fallback: model-based summary or preview ----------
    if not handled:
        if MODEL:
            df_preview = df.head(5).to_string(index=False) if not df.empty else "Dataset is empty or missing."
            context = f"Dataset columns: {', '.join(df.columns) if not df.empty else 'NONE'}\nSample rows:\n{df_preview}"
            prompt = f"{context}\nUser: {user_input}\nAnswer concisely using the dataset context."
            try:
                resp = MODEL.generate_content(prompt)
                ans = resp.text if resp and getattr(resp, "text", None) else "I couldn't generate a response."
            except Exception:
                ans = "AI model invocation failed; here's a preview of your data instead."
            push_model(ans)
        else:
            if df.empty:
                push_model("No dataset found at path: " + DATA_PATH)
            else:
                push_model("I couldn't detect a specific intent. Here are sample rows from the data:")
                with st.chat_message("model"):
                    st.dataframe(df.head(10))
        # end fallback

# End of file
