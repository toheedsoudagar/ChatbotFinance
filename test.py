# chatbot_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
from dotenv import load_dotenv
from typing import List, Optional

# --- Require Gemini AI ---
try:
    import google.generativeai as genai
except Exception as e:
    genai = None

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# If AI is required, ensure genai is available and key exists
if genai is None or not GOOGLE_API_KEY:
    st.set_page_config(page_title="Financial Data Chatbot (AI Required)", page_icon="💬", layout="centered")
    st.title("Financial Data Chatbot (AI Required)")
    st.error("Gemini AI integration is required to run this app.\n\n"
             "1) Install the 'google-generative-ai' package in your environment.\n"
             "2) Set GOOGLE_API_KEY in your environment or .env file.\n\n"
             "Once done, restart the app.")
    st.stop()

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)
MODEL = genai.GenerativeModel("gemini-2.5-flash")

# --- Utilities ---
def safe_lower_strip_replace(col: str) -> str:
    return col.strip().lower().replace(" ", "_")

# --- Data loading ---
@st.cache_data(ttl=600)
def load_data(file_path: str = "data/Revenue File.xlsx", sheet_name: str = "Revenue") -> pd.DataFrame:
    if not os.path.exists(file_path):
        return pd.DataFrame()
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    except Exception:
        try:
            df = pd.read_excel(file_path)
        except Exception:
            return pd.DataFrame()
    df = df.rename(columns={c: safe_lower_strip_replace(c) for c in df.columns})
    df = df.dropna(how="all").copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    for c in df.select_dtypes(["object"]).columns:
        df[c] = df[c].astype(str).str.strip()
    return df

# --- Business masks ---
def revenue_mask(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series([False] * 0, index=df.index)
    if "vouchertype" not in df.columns or "amount_type" not in df.columns or "amount" not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    vt = df["vouchertype"].fillna("").str.lower().str.strip()
    at = df["amount_type"].fillna("").str.lower().str.strip()
    amt = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    return (vt == "receipt") & (at == "cr") & (amt > 0)

def payments_mask(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series([False] * 0, index=df.index)
    if "vouchertype" not in df.columns or "amount_type" not in df.columns or "amount" not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    vt = df["vouchertype"].fillna("").str.lower().str.strip()
    at = df["amount_type"].fillna("").str.lower().str.strip()
    amt = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    return (vt == "payment") & (at == "dr") & (amt > 0)

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
            return pd.DataFrame({"month": list(month_map.values()), "value": [0] * 12})
        return pd.DataFrame(columns=["month", "value"])
    month_grp = revy.groupby(revy["date"].dt.month)["amount"].sum().reset_index()
    month_grp.columns = ["month_num", "value"]
    month_grp["month"] = month_grp["month_num"].map(month_map)
    if fill_empty_months:
        df_all = pd.DataFrame({"month_num": list(range(1, 13))})
        df_all = df_all.merge(month_grp, on="month_num", how="left").fillna({"value": 0})
        df_all["month"] = df_all["month_num"].map(month_map)
        return df_all[["month", "value"]]
    return month_grp[["month", "value"]]

def receipts_payments_net_for_period(df_local: pd.DataFrame, start_ts, end_ts):
    if "date" not in df_local.columns:
        return 0.0, 0.0, 0.0
    period_df = df_local[(df_local["date"] >= start_ts) & (df_local["date"] <= end_ts)]
    receipts = period_df.loc[revenue_mask(period_df), "amount"].sum()
    payments = period_df.loc[payments_mask(period_df), "amount"].sum()
    net = receipts - payments
    return receipts, payments, net

def top_entities(df: pd.DataFrame, entity_col: str, n: int = 5) -> pd.DataFrame:
    if entity_col not in df.columns or df.empty:
        return pd.DataFrame(columns=[entity_col, "revenue"])
    rev = df.loc[revenue_mask(df)].copy()
    top_df = (
        rev.groupby(entity_col)["amount"]
        .sum()
        .reset_index()
        .rename(columns={"amount": "revenue"})
        .sort_values("revenue", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )
    return top_df

def format_rank_table(df: pd.DataFrame, col: str) -> str:
    lines = [f"{i+1}. {row[col]} — {format_currency(row['revenue'])}" for i, row in df.iterrows()]
    return "\n".join(lines)

# --- Presentation helpers ---
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

def compute_yoy(grouped_year_df: pd.DataFrame) -> Optional[str]:
    if grouped_year_df.shape[0] < 2:
        return None
    last = grouped_year_df.iloc[-1]["value"]
    prev = grouped_year_df.iloc[-2]["value"]
    if prev == 0:
        return None
    pct = (last - prev) / prev * 100.0
    arrow = "↑" if pct >= 0 else "↓"
    return f"{arrow} {abs(pct):.1f}% vs {int(grouped_year_df.iloc[-2]['year'])}"

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
        ytext = compute_yoy(grouped)
        if ytext:
            summary += f"\n• Change (latest vs previous): {ytext}"
    else:
        summary = (f"📆 **Month-wise Summary:**\n"
                   f"• Total revenue across selected months: {format_currency(total_rev)}\n"
                   f"• Highest in {max_label}: {format_currency(max_val)}")
    return summary

# --- Keyword helpers ---
def extract_keywords_from_query(q: str, extra_exclude: Optional[set] = None) -> List[str]:
    exclude_words = {
        "revenue", "amount", "total", "show", "what", "is", "the", "for", "by", "month", "year",
        "in", "of", "from", "this", "that", "wise", "need", "and", "want", "data", "give", "me",
        "require", "tell", "get", "calculate", "display", "find", "list", "please", "all",
        "can", "could", "would", "should", "may", "will", "also",
        "bar", "chart", "barchart", "plot", "line", "graph", "today", "yesterday",
        "show", "monthly", "annual", "annually", "yearwise", "monthwise"
    }
    if extra_exclude:
        exclude_words |= extra_exclude
    words = [w for w in re.findall(r"[a-zA-Z]+", q) if w.lower() not in exclude_words and len(w) > 2]
    keywords = [w.lower() for w in words if not w.isdigit()]
    return keywords

def last_keywords_to_regex(keywords: List[str]) -> str:
    return "|".join(re.escape(k) for k in keywords)

# --- Load dataset ---
DATA_PATH = "data/Revenue File.xlsx"
df = load_data(DATA_PATH)

# --- Streamlit UI ---
st.set_page_config(page_title="Financial Data Chatbot ", page_icon="💬", layout="centered")
st.title("Financial Data Chatbot (AI Required)")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_keywords" not in st.session_state:
    st.session_state["last_keywords"] = []
if "last_context_type" not in st.session_state:
    st.session_state["last_context_type"] = None

# show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat input ---
user_input = st.chat_input("Ask about your financial data (AI required)...")

def ai_generate_answer(prompt: str) -> str:
    """Call Gemini to generate a textual answer. Returns text or fallback message."""
    try:
        resp = MODEL.generate_content(prompt)
        if resp and getattr(resp, "text", None):
            return resp.text
        # older clients may have choices etc - try to convert to string
        return str(resp)
    except Exception as e:
        return f"(AI generation failed: {str(e)})"

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    q = user_input.lower()

    # early intents
    viz_terms = {"bar", "chart", "plot", "barchart", "line", "graph"}
    wants_chart = any(t in q for t in viz_terms)
    wants_year = bool(re.search(r"\b(yearwise|year\s*wise|year|annual|annually)\b", q))
    wants_month = bool(re.search(r"\b(monthwise|month\s*wise|month|monthly)\b", q))

    def push_model(content: str, display_as_markdown: bool = True):
        """Send content to UI and save in session history."""
        with st.chat_message("model"):
            if display_as_markdown:
                st.markdown(content)
            else:
                st.write(content)
        st.session_state.messages.append({"role": "model", "content": content})

    # detect specific month + year like "April 2020"
    month_names = {
        "january":1, "february":2, "march":3, "april":4, "may":5, "june":6,
        "july":7, "august":8, "september":9, "october":10, "november":11, "december":12
    }
    month_match = next((m for m in month_names if m in q), None)
    year_match = re.search(r"\b(20\d{2})\b", q)
    if month_match and year_match:
        month_num = month_names[month_match]
        year_val = int(year_match.group(1))
        if "date" in df.columns:
            df_filtered = df[(df["date"].dt.year == year_val) & (df["date"].dt.month == month_num)]
            val = df_filtered.loc[revenue_mask(df_filtered), "amount"].sum()
            month_label = month_match.capitalize()
            # Use AI to craft a friendly answer
            prompt = (f"You are a data analyst. User asked: 'Revenue for {month_label} {year_val}'. "
                      f"Dataset has {len(df_filtered)} matching rows. Revenue (Receipt/Cr/amount>0) = {val:.2f}. "
                      f"Respond concisely with one short paragraph stating the amount and any note if zero or no data.")
            ai_text = ai_generate_answer(prompt)
            push_model(f"**AI:** {ai_text}")
            st.write("")
            chart_df = pd.DataFrame({"month":[f"{month_label} {year_val}"],"value":[val]})
            plot_chart(chart_df, "month", "value", f"Revenue for {month_label} {year_val}", kind="bar")
            st.stop()
        else:
            prompt = f"User asked for month/year but dataset has no 'date' column."
            ai_text = ai_generate_answer(prompt)
            push_model(f"**AI:** {ai_text}")
            st.stop()

    # Top-N entities intent (prefer before generic fallbacks)
    if any(w in q for w in ["top", "highest", "largest", "biggest", "most"]):
        n_match = re.search(r"\btop\s*(\d+)", q)
        n = int(n_match.group(1)) if n_match else 5

        if "party" in q and "partyname" in df.columns:
            top_df = top_entities(df, "partyname", n)
            prompt = (f"User asked for top {n} parties by revenue. Provide a short natural language list "
                      f"and include numeric amounts. Data (top rows):\n{top_df.to_string(index=False)}")
            ai_text = ai_generate_answer(prompt)
            # show table + AI text
            push_model("**Top parties (table):**\n\n" + top_df.to_string(index=False))
            push_model(f"**AI:** {ai_text}")
            st.stop()
        elif any(col in df.columns for col in ["ledger_name","ledger"]) and "ledger" in q:
            ledger_col = "ledger_name" if "ledger_name" in df.columns else "ledger"
            top_df = top_entities(df, ledger_col, n)
            prompt = (f"User asked for top {n} ledgers by revenue. Provide a concise natural language list "
                      f"and include numeric amounts. Data (top rows):\n{top_df.to_string(index=False)}")
            ai_text = ai_generate_answer(prompt)
            push_model("**Top ledgers (table):**\n\n" + top_df.to_string(index=False))
            push_model(f"**AI:** {ai_text}")
            st.stop()
        elif "partyname" in df.columns:
            top_df = top_entities(df, "partyname", n)
            prompt = (f"User asked for top {n} parties by revenue (unspecified). Provide a concise natural list "
                      f"and include numeric amounts. Data (top rows):\n{top_df.to_string(index=False)}")
            ai_text = ai_generate_answer(prompt)
            push_model("**Top parties (table):**\n\n" + top_df.to_string(index=False))
            push_model(f"**AI:** {ai_text}")
            st.stop()
        else:
            prompt = "User asked for top entities but dataset lacks partyname/ledger columns."
            ai_text = ai_generate_answer(prompt)
            push_model(f"**AI:** {ai_text}")
            st.stop()

    # receipts/payments/net queries
    if any(w in q for w in ["receipts", "payments", "net", "balance", "inflow", "outflow"]):
        # detect month+year or year
        year_m = re.search(r"\b(20\d{2})\b", q)
        month_m = next((m for m in month_names if m in q), None)
        if month_m and year_m:
            mnum = month_names[month_m]
            yv = int(year_m.group(1))
            start = pd.Timestamp(year=yv, month=mnum, day=1)
            end = (start + pd.offsets.MonthEnd(0))
            receipts, payments, net = receipts_payments_net_for_period(df, start, end)
            prompt = (f"User requested receipts/payments/net for {month_m.capitalize()} {yv}. "
                      f"Receipts={receipts:.2f}, Payments={payments:.2f}, Net={net:.2f}. "
                      "Provide a concise summary sentence + a 2-line recommendation if net is negative.")
            ai_text = ai_generate_answer(prompt)
            push_model(f"**AI:** {ai_text}")
            st.stop()
        elif year_m:
            yv = int(year_m.group(1))
            start = pd.Timestamp(year=yv, month=1, day=1)
            end = pd.Timestamp(year=yv, month=12, day=31, hour=23, minute=59, second=59)
            receipts, payments, net = receipts_payments_net_for_period(df, start, end)
            prompt = (f"User requested receipts/payments/net for {yv}. Receipts={receipts:.2f}, Payments={payments:.2f}, Net={net:.2f}. "
                      "Provide a concise summary sentence + short note on trend.")
            ai_text = ai_generate_answer(prompt)
            push_model(f"**AI:** {ai_text}")
            st.stop()
        else:
            receipts = df.loc[revenue_mask(df), "amount"].sum() if not df.empty else 0.0
            payments = df.loc[payments_mask(df), "amount"].sum() if not df.empty else 0.0
            net = receipts - payments
            prompt = (f"User requested overall receipts/payments/net. Receipts={receipts:.2f}, Payments={payments:.2f}, Net={net:.2f}. "
                      "Answer in a concise human-friendly paragraph.")
            ai_text = ai_generate_answer(prompt)
            push_model(f"**AI:** {ai_text}")
            st.stop()

    # chart + time tokens (year/month) requests
    if wants_chart and wants_year:
        ydf = compute_revenue_by_year(df)
        prompt = (f"User requested a year-wise revenue chart. Provide a one-sentence caption summarizing totals and latest year. "
                  f"Data:\n{ydf.to_string(index=False)}")
        ai_text = ai_generate_answer(prompt)
        push_model(f"**AI Caption:** {ai_text}")
        if not ydf.empty:
            plot_chart(ydf, "year", "value", "Year-wise Revenue (All Data)", kind="bar")
        st.stop()

    if wants_chart and wants_month:
        mdf = compute_revenue_by_month(df)
        prompt = (f"User requested a month-wise revenue chart. Provide a one-sentence caption summarizing trend and top month. "
                  f"Data (first/last rows):\n{mdf.head(3).to_string(index=False)}\n...\n{mdf.tail(3).to_string(index=False)}")
        ai_text = ai_generate_answer(prompt)
        push_model(f"**AI Caption:** {ai_text}")
        if not mdf.empty:
            plot_chart(mdf, "month", "value", "Monthly Revenue (All Data)", kind="bar")
        st.stop()

    # follow-ups for year/month (no plot token)
    if wants_year or wants_month:
        last_keywords = st.session_state.get("last_keywords", [])
        if not last_keywords:
            if wants_month:
                grouped = compute_revenue_by_month(df)
                label_kind = "month"
                title = "Month-wise Revenue (All Data)"
            else:
                grouped = compute_revenue_by_year(df)
                label_kind = "year"
                title = "Year-wise Revenue (All Data)"
            prompt = (f"User requested {label_kind}-wise summary. Provide 2-line summary. Data excerpt:\n{grouped.head(5).to_string(index=False)}")
            ai_text = ai_generate_answer(prompt)
            push_model(f"**AI:** {ai_text}")
            if not grouped.empty:
                plot_chart(grouped, grouped.columns[0], "value", title, kind="bar")
            st.stop()
        else:
            regex = last_keywords_to_regex(last_keywords)
            if "partyname" in df.columns:
                mask = revenue_mask(df) & df["partyname"].str.contains(regex, case=False, na=False)
                subset = df[mask]
                total = subset["amount"].sum() if not subset.empty else 0.0
                if wants_month:
                    grouped = subset.groupby(subset["date"].dt.month)["amount"].sum().reset_index().rename(columns={"date":"month","amount":"value"})
                    grouped["month"] = grouped["month"].map({1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"})
                    label_kind = "month"
                    title = f"Month-wise Revenue – {' / '.join(last_keywords)}"
                else:
                    grouped = subset.groupby(subset["date"].dt.year)["amount"].sum().reset_index().rename(columns={"date":"year","amount":"value"})
                    grouped["year"] = grouped["year"].astype(int)
                    label_kind = "year"
                    title = f"Year-wise Revenue – {' / '.join(last_keywords)}"
                prompt = (f"User requested {label_kind}-wise for {' / '.join(last_keywords)}. "
                          f"Total revenue = {total:.2f}. Provide a concise summary sentence and mention top period. Data excerpt:\n{grouped.head(5).to_string(index=False)}")
                ai_text = ai_generate_answer(prompt)
                push_model(f"**AI:** {ai_text}")
                if not grouped.empty:
                    plot_chart(grouped, grouped.columns[0], "value", title, kind="bar")
                st.stop()

    # generic total revenue queries
    if re.search(r"\b(total\s+revenue|what\s+is\s+the\s+total\s+revenue|total\s+revenue\s+this\s+month|total\s+revenue\s+in\s+\d{4})\b", q) \
       or ("revenue" in q and ("total" in q or "this month" in q or re.search(r"\b\d{4}\b", q)) and not re.search(r"\b(hostel|mess|canteen|contractor|party)\b", q)):
        if "this month" in q and "date" in df.columns:
            now = pd.Timestamp.now()
            dfm = df[(df["date"].dt.year == now.year) & (df["date"].dt.month == now.month)]
            val = dfm.loc[revenue_mask(dfm), "amount"].sum()
            prompt = f"User asked total revenue this month ({now.strftime('%Y-%m')}). Amount = {val:.2f}. Give short sentence."
            ai_text = ai_generate_answer(prompt)
            push_model(f"**AI:** {ai_text}")
            st.stop()
        elif re.search(r"\b(20\d{2})\b", q):
            match = re.search(r"\b(20\d{2})\b", q)
            year = int(match.group(1))
            if "date" in df.columns:
                dfy = df[df["date"].dt.year == year]
                val = dfy.loc[revenue_mask(dfy), "amount"].sum()
                prompt = f"User asked total revenue in {year}. Amount = {val:.2f}. Give short sentence with one observation."
                ai_text = ai_generate_answer(prompt)
                push_model(f"**AI:** {ai_text}")
                st.stop()
            else:
                ai_text = ai_generate_answer("User requested year-specific total but dataset has no date column.")
                push_model(f"**AI:** {ai_text}")
                st.stop()
        else:
            total = compute_total_revenue(df)
            prompt = f"User asked total revenue for all data. Amount = {total:.2f}. Provide a friendly one-line answer."
            ai_text = ai_generate_answer(prompt)
            push_model(f"**AI:** {ai_text}")
            st.stop()

    # category / keyword-based revenue
    if "revenue" in q:
        keywords = extract_keywords_from_query(q)
        if keywords:
            st.session_state["last_keywords"] = keywords
            st.session_state["last_context_type"] = None
        else:
            keywords = st.session_state.get("last_keywords", [])
        if not keywords:
            ai_text = ai_generate_answer("User asked for revenue but no category specified. Ask them to specify a party or ledger, or say 'total revenue'.")
            push_model(f"**AI:** {ai_text}")
            st.stop()
        regex = last_keywords_to_regex(keywords)
        if "partyname" in df.columns:
            mask = revenue_mask(df) & df["partyname"].str.contains(regex, case=False, na=False)
            subset = df[mask]
            if subset.empty:
                ai_text = ai_generate_answer(f"No transactions found for {' / '.join(keywords)} in dataset.")
                push_model(f"**AI:** {ai_text}")
                st.stop()
            total = subset["amount"].sum()
            monthly = subset.groupby(subset["date"].dt.to_period("M"))["amount"].sum().reset_index().rename(columns={"date":"month","amount":"value"})
            monthly["month"] = monthly["month"].astype(str)
            prompt = (f"User requested revenue for {' / '.join(keywords)}. Total={total:.2f}. "
                      f"Provide a concise AI summary and a short suggestion if trend is down. Data excerpt:\n{monthly.head(5).to_string(index=False)}")
            ai_text = ai_generate_answer(prompt)
            push_model(f"**AI:** {ai_text}")
            plot_chart(monthly, "month", "value", f"Monthly Trend – {' / '.join(keywords)}", kind="bar")
            st.stop()
        else:
            ai_text = ai_generate_answer("Dataset does not have a 'partyname' column to filter by party.")
            push_model(f"**AI:** {ai_text}")
            st.stop()

    # If nothing matched -> use AI to analyze / suggest from sample
    df_preview = df.head(10).to_string(index=False) if not df.empty else "Dataset empty or missing."
    prompt = f"User query: {user_input}\nDataset columns: {', '.join(df.columns) if not df.empty else 'NONE'}\nSample rows:\n{df_preview}\nProvide a concise answer or ask a clarifying question if needed."
    ai_text = ai_generate_answer(prompt)
    push_model(f"**AI:** {ai_text}")
    st.stop()
