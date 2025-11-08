# chatbot_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
from dotenv import load_dotenv
from typing import List, Optional

# Optional Gemini integration
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

# --- Utilities ---
def safe_lower_strip_replace(col: str) -> str:
    return col.strip().lower().replace(" ", "_")

# --- Data loading ---
@st.cache_data(ttl=600)  # cache for 10 minutes (tunable)
def load_data(file_path: str = "data/Revenue File.xlsx", sheet_name: str = "Revenue") -> pd.DataFrame:
    """Load and normalize the dataset. Returns empty DataFrame on failure."""
    if not os.path.exists(file_path):
        return pd.DataFrame()  # caller will handle empty DF
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    except Exception:
        # try default sheet or first sheet
        try:
            df = pd.read_excel(file_path)
        except Exception:
            return pd.DataFrame()
    # normalize column names
    df = df.rename(columns={c: safe_lower_strip_replace(c) for c in df.columns})
    df = df.dropna(how="all").copy()
    # ensure date and amount typed correctly if present
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    # convert object columns to str to avoid surprises
    for c in df.select_dtypes(["object"]).columns:
        df[c] = df[c].astype(str).str.strip()
    return df

# --- Business masks ---
def revenue_mask(df: pd.DataFrame) -> pd.Series:
    """
    Identify revenue records:
      - vouchertype == 'receipt'
      - amount_type == 'cr'
      - amount > 0
    If required columns are missing, returns a boolean Series of False.
    """
    if df.empty:
        return pd.Series([False] * 0, index=df.index)
    if "vouchertype" not in df.columns or "amount_type" not in df.columns or "amount" not in df.columns:
        # gracefully return False for all rows if essential columns are missing
        return pd.Series([False] * len(df), index=df.index)
    vt = df["vouchertype"].fillna("").str.lower().str.strip()
    at = df["amount_type"].fillna("").str.lower().str.strip()
    amt = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    return (vt == "receipt") & (at == "cr") & (amt > 0)

# --- Aggregation helpers ---
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
    grouped = grouped.rename(columns={"amount": "value"})[["month", "value"]]
    # ensure chronological order
    grouped = grouped.sort_values("month").reset_index(drop=True)
    return grouped

def compute_revenue_by_year(df_local: pd.DataFrame) -> pd.DataFrame:
    rev = df_local.loc[revenue_mask(df_local)].copy()
    if "date" not in rev.columns or rev.empty:
        return pd.DataFrame(columns=["year", "value"])
    grouped = rev.groupby(rev["date"].dt.year)["amount"].sum().reset_index()
    grouped.columns = ["year", "value"]
    grouped["year"] = grouped["year"].astype(int)
    grouped = grouped.sort_values("year").reset_index(drop=True)
    return grouped

def compute_revenue_by_month_for_year(df_local: pd.DataFrame, year: int, fill_empty_months: bool = True) -> pd.DataFrame:
    rev = df_local.loc[revenue_mask(df_local)].copy()
    if "date" not in rev.columns:
        return pd.DataFrame(columns=["month", "value"])
    revy = rev[rev["date"].dt.year == year]
    month_map = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    if revy.empty:
        if fill_empty_months:
            # return all months with zero values
            return pd.DataFrame({"month": list(month_map.values()), "value": [0] * 12})
        return pd.DataFrame(columns=["month", "value"])
    month_grp = revy.groupby(revy["date"].dt.month)["amount"].sum().reset_index()
    month_grp.columns = ["month_num", "value"]
    month_grp["month"] = month_grp["month_num"].map(month_map)
    # ensure all months present in chronological order (if requested)
    if fill_empty_months:
        df_all = pd.DataFrame({"month_num": list(range(1, 13))})
        df_all = df_all.merge(month_grp, on="month_num", how="left").fillna({"value": 0})
        df_all["month"] = df_all["month_num"].map(month_map)
        return df_all[["month", "value"]]
    return month_grp[["month", "value"]]

# --- Presentation helpers ---
def format_currency(val: float) -> str:
    try:
        return f"₹{val:,.0f}"
    except Exception:
        return f"₹{float(val):,.2f}"

def plot_chart(df_plot: pd.DataFrame, x_col: str, y_col: str, title: str, kind: str = "bar"):
    """Matplotlib plotting helper that outputs into Streamlit."""
    if df_plot.empty:
        st.write("No data to plot.")
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    plt.tight_layout(pad=3)
    # ensure x ordering is preserved
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

# --- Helpers for keyword extraction and matching ---
def extract_keywords_from_query(q: str, extra_exclude: Optional[set] = None) -> List[str]:
    exclude_words = {
        "revenue", "amount", "total", "show", "what", "is", "the", "for", "by", "month", "year",
        "in", "of", "from", "this", "that", "wise", "need", "and", "want", "data", "give", "me",
        "require", "tell", "get", "calculate", "display", "find", "list", "please", "all",
        "can", "could", "would", "should", "may", "will", "also",
        "bar", "chart", "barchart", "plot", "line", "graph", "please", "today", "yesterday",
        "show", "monthly", "annual", "annualy", "yearwise", "monthwise"
    }
    if extra_exclude:
        exclude_words |= extra_exclude
    words = [w for w in re.findall(r"[a-zA-Z]+", q) if w.lower() not in exclude_words and len(w) > 2]
    keywords = [w.lower() for w in words if not w.isdigit()]
    return keywords

def last_keywords_to_regex(keywords: List[str]) -> str:
    # escape and join with | for regex contains
    return "|".join(re.escape(k) for k in keywords)

# --- Load dataset ---
DATA_PATH = "data/Revenue File.xlsx"
df = load_data(DATA_PATH)

# --- Streamlit UI ---
st.set_page_config(page_title="Financial Data Chatbot", page_icon="💬", layout="centered")
st.title("Financial Data Chatbot 🔨🤖🔧")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_keywords" not in st.session_state:
    st.session_state["last_keywords"] = []

# show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat input ---
user_input = st.chat_input("Ask about your financial data...")

if user_input:
    # record user message
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    q = user_input.lower()

    # early intent detection
    viz_terms = {"bar", "chart", "plot", "barchart", "line", "graph"}
    wants_chart = any(t in q for t in viz_terms)
    wants_year = bool(re.search(r"\b(yearwise|year\s*wise|year|annual|annually)\b", q))
    wants_month = bool(re.search(r"\b(monthwise|month\s*wise|month|monthly)\b", q))

    # helper to show message
    def push_model(content: str, display_as_markdown: bool = True):
        with st.chat_message("model"):
            if display_as_markdown:
                st.markdown(content)
            else:
                st.write(content)
        st.session_state.messages.append({"role": "model", "content": content})

    # -------------- 1) priority: chart + time tokens --------------
    if wants_chart and wants_year:
        last_keywords = st.session_state.get("last_keywords", [])
        if last_keywords:
            key_str = last_keywords_to_regex(last_keywords)
            if "partyname" in df.columns:
                mask = revenue_mask(df) & df["partyname"].str.contains(key_str, case=False, na=False)
                subset = df[mask]
                if subset.empty:
                    push_model(f"⚠️ No data found for {' / '.join(last_keywords)}.")
                else:
                    grouped = subset.groupby(subset["date"].dt.year)["amount"].sum().reset_index().rename(columns={"date": "year", "amount": "value"})
                    grouped["year"] = grouped["year"].astype(int)
                    summary_text = build_summary(grouped, "year")
                    push_model(summary_text)
                    st.write("")
                    st.write(f"📈 Year-wise Revenue – {' / '.join(last_keywords)}")
                    plot_chart(grouped, "year", "value", f"Year-wise Revenue – {' / '.join(last_keywords)}", kind="bar")
            else:
                push_model("⚠️ 'partyname' column not found in dataset.")
        else:
            ydf = compute_revenue_by_year(df)
            if ydf.empty:
                push_model("No year-wise revenue data available.")
            else:
                summary_text = build_summary(ydf, "year")
                push_model(summary_text)
                st.write("")
                st.write("📈 Year-wise Revenue (All Data)")
                plot_chart(ydf, "year", "value", "Year-wise Revenue (All Data)", kind="bar")

    elif wants_chart and wants_month:
        last_keywords = st.session_state.get("last_keywords", [])
        if last_keywords:
            key_str = last_keywords_to_regex(last_keywords)
            if "partyname" in df.columns:
                mask = revenue_mask(df) & df["partyname"].str.contains(key_str, case=False, na=False)
                subset = df[mask]
                if subset.empty:
                    push_model(f"⚠️ No data found for {' / '.join(last_keywords)}.")
                else:
                    grouped = subset.groupby(subset["date"].dt.to_period("M"))["amount"].sum().reset_index()
                    grouped["month"] = grouped["date"].astype(str)
                    grouped = grouped.rename(columns={"amount": "value"})[["month", "value"]]
                    grouped = grouped.sort_values("month").reset_index(drop=True)
                    summary_text = build_summary(grouped, "month")
                    push_model(summary_text)
                    st.write("")
                    st.write(f"📈 Month-wise Revenue – {' / '.join(last_keywords)}")
                    plot_chart(grouped, "month", "value", f"Month-wise Revenue – {' / '.join(last_keywords)}", kind="bar")
            else:
                push_model("⚠️ 'partyname' column not found in dataset.")
        else:
            mdf = compute_revenue_by_month(df)
            if mdf.empty:
                push_model("No monthly revenue data available.")
            else:
                summary_text = build_summary(mdf, "month")
                push_model(summary_text)
                st.write("")
                st.write("📈 Monthly Revenue (All Data)")
                plot_chart(mdf, "month", "value", "Monthly Revenue (All Data)", kind="bar")

    # -------------- 2) follow-ups: year/month without explicit 'plot' --------------
    elif wants_year or wants_month:
        last_keywords = st.session_state.get("last_keywords", [])
        if last_keywords:
            key_str = last_keywords_to_regex(last_keywords)
            if "partyname" in df.columns:
                mask = revenue_mask(df) & df["partyname"].str.contains(key_str, case=False, na=False)
                subset = df[mask]
                if subset.empty:
                    push_model(f"⚠️ No data found for {' / '.join(last_keywords)}.")
                else:
                    if wants_month:
                        grouped = subset.groupby(subset["date"].dt.month)["amount"].sum().reset_index().rename(columns={"date": "month", "amount": "value"})
                        grouped["month"] = grouped["month"].map({1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                                                                 7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"})
                        label_kind = "month"
                        title = f"Month-wise Revenue – {' / '.join(last_keywords)}"
                    else:
                        grouped = subset.groupby(subset["date"].dt.year)["amount"].sum().reset_index().rename(columns={"date": "year", "amount": "value"})
                        grouped["year"] = grouped["year"].astype(int)
                        label_kind = "year"
                        title = f"Year-wise Revenue – {' / '.join(last_keywords)}"
                    grouped = grouped.sort_values(grouped.columns[0]).reset_index(drop=True)
                    summary_text = build_summary(grouped, label_kind)
                    push_model(summary_text)
                    st.write("")
                    st.write(f"📈 {title}")
                    plot_chart(grouped, grouped.columns[0], "value", title, kind="bar")
            else:
                push_model("⚠️ 'partyname' column not found in dataset.")
        else:
            if wants_month:
                grouped = compute_revenue_by_month(df)
                label_kind = "month"
                title = "Month-wise Revenue (All Data)"
            else:
                grouped = compute_revenue_by_year(df)
                label_kind = "year"
                title = "Year-wise Revenue (All Data)"
            if grouped.empty:
                push_model(f"No {label_kind}-wise data available.")
            else:
                summary_text = build_summary(grouped, label_kind)
                push_model(summary_text)
                st.write("")
                st.write(f"📈 {title}")
                plot_chart(grouped, grouped.columns[0], "value", title, kind="bar")

    # -------------- 3) generic total revenue queries --------------
    elif re.search(r"\b(total\s+revenue|what\s+is\s+the\s+total\s+revenue|total\s+revenue\s+this\s+month|total\s+revenue\s+in\s+\d{4})\b", q) \
         or ("revenue" in q and ("total" in q or "this month" in q or re.search(r"\b\d{4}\b", q)) and not re.search(r"\b(hostel|mess|canteen|contractor|party)\b", q)):
        if "this month" in q and "date" in df.columns:
            now = pd.Timestamp.now()
            dfm = df[(df["date"].dt.year == now.year) & (df["date"].dt.month == now.month)]
            val = dfm.loc[revenue_mask(dfm), "amount"].sum()
            resp_text = f"Total revenue for this month ({now.strftime('%Y-%m')}): {format_currency(val)}"
            push_model(resp_text)
        elif re.search(r"\b(20\d{2})\b", q):
            match = re.search(r"\b(20\d{2})\b", q)
            year = int(match.group(1))
            if "date" in df.columns:
                dfy = df[df["date"].dt.year == year]
                val = dfy.loc[revenue_mask(dfy), "amount"].sum()
                resp_text = f"Total revenue in {year}: {format_currency(val)}"
                push_model(resp_text)
            else:
                push_model("No date column available to compute year-specific revenue.")
        else:
            total = compute_total_revenue(df)
            resp_text = f"Total revenue (all data): {format_currency(total)}"
            push_model(resp_text)

    # -------------- 4) category / keyword-based revenue --------------
    elif "revenue" in q:
        keywords = extract_keywords_from_query(q)
        if keywords:
            st.session_state["last_keywords"] = keywords
        else:
            keywords = st.session_state.get("last_keywords", [])

        if not keywords:
            push_model("⚠️ Please specify a category or party (e.g., 'hostel', 'mess', 'canteen') or ask generic 'total revenue'.")
        else:
            key_str = last_keywords_to_regex(keywords)
            if "partyname" in df.columns:
                mask = revenue_mask(df) & df["partyname"].str.contains(key_str, case=False, na=False)
                subset = df[mask]
                if subset.empty:
                    push_model(f"⚠️ No transactions found for {' / '.join(keywords)}.")
                else:
                    total = subset["amount"].sum()
                    monthly = subset.groupby(subset["date"].dt.to_period("M"))["amount"].sum().reset_index().rename(columns={"date": "month", "amount": "value"})
                    monthly["month"] = monthly["month"].astype(str)
                    monthly = monthly.sort_values("month").reset_index(drop=True)
                    resp_text = f"📊 Total Revenue for {' / '.join(keywords)}: {format_currency(total)}"
                    # display
                    with st.chat_message("model"):
                        st.write(resp_text)
                        st.markdown(build_summary(monthly, "month"))
                        kind = "bar" if wants_chart else "line"
                        plot_chart(monthly, "month", "value", f"Monthly Trend – {' / '.join(keywords)}", kind=kind)
                    st.session_state.messages.append({"role":"model","content":resp_text})
            else:
                push_model("⚠️ 'partyname' column not found in dataset.")

    # -------------- 5) fallback: AI summary or data preview --------------
    else:
        if MODEL:
            # give dataset context (limited rows) to the model
            df_preview = df.head(5).to_string(index=False) if not df.empty else "Dataset is empty or missing."
            context = f"Dataset columns: {', '.join(df.columns) if not df.empty else 'NONE'}\nSample rows:\n{df_preview}"
            prompt = f"{context}\nUser: {user_input}\nAnswer concisely using the dataset context."
            try:
                resp = MODEL.generate_content(prompt)
                ans = resp.text if resp and getattr(resp, "text", None) else "I couldn't generate a response."
            except Exception:
                ans = "AI model invocation failed; here's a preview of your data instead."
            with st.chat_message("model"):
                st.markdown(ans)
            st.session_state.messages.append({"role":"model","content":ans})
        else:
            resp_text = "I couldn't detect a specific intent. Here are sample rows from the data:"
            with st.chat_message("model"):
                st.write(resp_text)
                if df.empty:
                    st.write("No dataset found at path: " + DATA_PATH)
                else:
                    st.dataframe(df.head(10))
            st.session_state.messages.append({"role":"model","content":resp_text})
