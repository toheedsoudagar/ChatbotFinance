# chatbot_app.py
import streamlit as st
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import re
import numpy as np

# --- Load environment variables ---
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")
else:
    model = None

# --- Load Excel Data ---
def load_data(file_path: str, sheet_name: str = "Revenue") -> pd.DataFrame:
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df = df.dropna(how="all").copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    # ensure text columns are strings
    for c in df.select_dtypes(["object"]).columns:
        df[c] = df[c].astype(str)
    return df

# --- Business logic masks ---
def revenue_mask(df: pd.DataFrame) -> pd.Series:
    if "vouchertype" not in df.columns or "amount_type" not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    return (
        df["vouchertype"].str.lower().eq("receipt")
        & df["amount_type"].str.lower().eq("cr")
        & (df["amount"] > 0)
    )

# --- Chart helpers ---
def format_currency(val):
    return f"₹{val:,.0f}"

def plot_chart(df_plot, x_col, y_col, title, kind="bar"):
    fig, ax = plt.subplots(figsize=(9, 4))
    plt.tight_layout(pad=3)

    x_vals = df_plot[x_col].astype(str).tolist()
    y_vals = df_plot[y_col].tolist()

    if kind == "bar":
        bars = ax.bar(x_vals, y_vals, color="#4CAF50", alpha=0.95)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(format_currency(h),
                        xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0,5), textcoords="offset points",
                        ha="center", fontsize=9, fontweight="bold")
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

# --- Aggregation helpers ---
def compute_total_revenue(df_local):
    return df_local.loc[revenue_mask(df_local), "amount"].sum()

def compute_revenue_by_month(df_local):
    rev = df_local.loc[revenue_mask(df_local)].copy()
    if "date" not in rev.columns or rev.empty:
        return pd.DataFrame(columns=["month","value"])
    grouped = rev.groupby(rev["date"].dt.to_period("M"))["amount"].sum().reset_index()
    grouped["month"] = grouped["date"].astype(str)
    grouped = grouped.rename(columns={"amount":"value"})[["month","value"]]
    return grouped

def compute_revenue_by_year(df_local):
    rev = df_local.loc[revenue_mask(df_local)].copy()
    if "date" not in rev.columns or rev.empty:
        return pd.DataFrame(columns=["year","value"])
    grouped = rev.groupby(rev["date"].dt.year)["amount"].sum().reset_index()
    grouped.columns = ["year","value"]
    grouped["year"] = grouped["year"].astype(int)
    return grouped

def compute_revenue_by_month_for_year(df_local, year:int):
    rev = df_local.loc[revenue_mask(df_local)].copy()
    if "date" not in rev.columns:
        return pd.DataFrame(columns=["month","value"])
    revy = rev[rev["date"].dt.year == year]
    if revy.empty:
        return pd.DataFrame(columns=["month","value"])
    month_grp = revy.groupby(revy["date"].dt.month)["amount"].sum().reset_index()
    month_grp.columns = ["month_num","value"]
    month_grp["month"] = month_grp["month_num"].map({
        1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
        7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"
    })
    return month_grp[["month","value"]]

# --- Summary builder ---
def build_summary(grouped: pd.DataFrame, label_kind: str) -> str:
    if grouped.empty:
        return f"No data available for {label_kind}."
    total_rev = grouped["value"].sum()
    idx = grouped["value"].idxmax()
    max_label = str(grouped.iloc[idx,0])
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

# --- Load data ---
DATA_PATH = "data/Revenue File.xlsx"
df = load_data(DATA_PATH)

# --- Streamlit setup ---
st.set_page_config(page_title="Financial Data Chatbot", page_icon="💬", layout="centered")
st.title("Financial Data Chatbot")

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
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role":"user","content":user_input})
    q = user_input.lower()

    # detect visualization intent (bar/line/plot/chart/barchart)
    viz_terms = ["bar", "chart", "plot", "barchart", "barchart", "line", "graph"]
    wants_chart = any(t in q for t in viz_terms)

    # ---------- 1) Generic total revenue queries (no category required) ----------
    if re.search(r"\b(total\s+revenue|what\s+is\s+the\s+total\s+revenue|total\s+revenue\s+this\s+month|total\s+revenue\s+in\s+\d{4}|total\s+revenue\s+in\b)", q) \
       or ( "revenue" in q and ("total" in q or "this month" in q or re.search(r"\b\d{4}\b", q)) and not re.search(r"\b(hostel|mess|canteen|contractor|party)\b", q) ):

        # "this month"
        if "this month" in q and "date" in df.columns:
            now = pd.Timestamp.now()
            dfm = df[(df["date"].dt.year == now.year) & (df["date"].dt.month == now.month)]
            val = dfm.loc[revenue_mask(dfm), "amount"].sum()
            with st.chat_message("model"):
                st.write(f"Total revenue for this month ({now.strftime('%Y-%m')}): {format_currency(val)}")

        # explicit year "in 2024"
        elif re.search(r"\b(20\d{2})\b", q):
            match = re.search(r"\b(20\d{2})\b", q)
            year = int(match.group(1))
            if "date" in df.columns:
                dfy = df[df["date"].dt.year == year]
                val = dfy.loc[revenue_mask(dfy), "amount"].sum()
                with st.chat_message("model"):
                    st.write(f"Total revenue in {year}: {format_currency(val)}")
                    if wants_chart and "month" in q:
                        monthly = compute_revenue_by_month_for_year(df, year)
                        if monthly.empty:
                            st.write(f"No revenue data for {year}.")
                        else:
                            st.markdown(build_summary(monthly, "month"))
                            plot_chart(monthly, "month", "value", f"Monthly Revenue – {year}", kind="bar")
                    elif wants_chart:
                        ydf = compute_revenue_by_year(df)
                        if ydf.empty:
                            st.write("No year-wise revenue data.")
                        else:
                            st.markdown(build_summary(ydf, "year"))
                            plot_chart(ydf, "year", "value", "Revenue by Year", kind="bar")
        else:
            total = compute_total_revenue(df)
            with st.chat_message("model"):
                st.write(f"Total revenue (all data): {format_currency(total)}")
                if wants_chart:
                    rbm = compute_revenue_by_month(df)
                    if not rbm.empty:
                        st.markdown(build_summary(rbm, "month"))
                        plot_chart(rbm, "month", "value", "Monthly Revenue (all data)", kind="bar")

    # ---------- 2) Category / keyword-based revenue (dynamic keywords) ----------
    elif "revenue" in q:
        # improved exclude list to avoid filler words and UI terms
        exclude_words = {
            "revenue","amount","total","show","what","is","the","for","by","month","year",
            "in","of","from","this","that","wise","need","and","want","data","give","me",
            "require","tell","get","calculate","display","find","list","please","all",
            # modal verbs & common helpers
            "can","could","would","should","may","will","also",
            # visualization keywords (don't treat as categories)
            "bar","chart","barchart","plot","line","graph"
        }

        words = [w for w in re.findall(r"[a-zA-Z]+", q) if w.lower() not in exclude_words and len(w) > 2]
        keywords = [w.lower() for w in words if not w.isdigit()]

        # Save or reuse last keywords
        if keywords:
            st.session_state["last_keywords"] = keywords
        else:
            keywords = st.session_state.get("last_keywords", [])

        # If no keywords but user wants a chart -> show generic revenue chart
        if not keywords and wants_chart:
            rbm = compute_revenue_by_month(df)
            if rbm.empty:
                with st.chat_message("model"):
                    st.write("No monthly revenue data available to plot.")
            else:
                with st.chat_message("model"):
                    st.write("📊 Revenue (all data)")
                    st.markdown(build_summary(rbm, "month"))
                    plot_chart(rbm, "month", "value", "Monthly Revenue (all data)", kind="bar")
        elif not keywords:
            with st.chat_message("model"):
                st.write("⚠️ Please specify a category or party (e.g., 'hostel', 'mess', 'canteen') or ask generic 'total revenue'.")
        else:
            key_str = "|".join(re.escape(k) for k in keywords)
            if "partyname" in df.columns:
                mask = revenue_mask(df) & df["partyname"].str.contains(key_str, case=False, na=False)
                subset = df[mask]
                if subset.empty:
                    with st.chat_message("model"):
                        st.write(f"⚠️ No transactions found for {' / '.join(keywords)}.")
                else:
                    total = subset["amount"].sum()
                    monthly = (
                        subset.groupby(subset["date"].dt.to_period("M"))["amount"]
                        .sum()
                        .reset_index()
                        .rename(columns={"date":"month","amount":"value"})
                    )
                    monthly["month"] = monthly["month"].astype(str)
                    with st.chat_message("model"):
                        st.write(f"📊 **Total Revenue for {' / '.join(keywords)}:** {format_currency(total)}")
                        if not monthly.empty:
                            st.markdown(build_summary(monthly, "month"))
                            # choose bar if user explicitly asked for bar/chart, else line by default for trends
                            kind = "bar" if wants_chart else "line"
                            plot_chart(monthly, "month", "value", f"Monthly Trend – {' / '.join(keywords)}", kind=kind)
            else:
                with st.chat_message("model"):
                    st.write("⚠️ 'partyname' column not found in dataset.")

    # ---------- 3) Follow-ups using context memory (year wise / month wise) ----------
    elif any(kw in q for kw in ["year", "yearwise", "year wise", "annual", "monthwise", "month wise"]):
        last_keywords = st.session_state.get("last_keywords", [])
        if not last_keywords:
            with st.chat_message("model"):
                st.write("Please specify what category or party you meant (e.g., 'Hostel', 'Canteen').")
        else:
            key_str = "|".join(re.escape(k) for k in last_keywords)
            if "partyname" in df.columns:
                mask = revenue_mask(df) & df["partyname"].str.contains(key_str, case=False, na=False)
                subset = df[mask]
                if subset.empty:
                    with st.chat_message("model"):
                        st.write(f"⚠️ No data found for {' / '.join(last_keywords)}.")
                else:
                    if "month" in q:
                        grouped = (
                            subset.groupby(subset["date"].dt.month)["amount"]
                            .sum()
                            .reset_index()
                            .rename(columns={"date":"month","amount":"value"})
                        )
                        grouped["month"] = grouped["month"].map({
                            1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                            7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"
                        })
                        label_kind = "month"
                        title = f"Month-wise Revenue – {' / '.join(last_keywords)}"
                    else:
                        grouped = (
                            subset.groupby(subset["date"].dt.year)["amount"]
                            .sum()
                            .reset_index()
                            .rename(columns={"date":"year","amount":"value"})
                        )
                        grouped["year"] = grouped["year"].astype(int)
                        label_kind = "year"
                        title = f"Year-wise Revenue – {' / '.join(last_keywords)}"

                    with st.chat_message("model"):
                        st.markdown(build_summary(grouped, label_kind))
                        st.write("")  # spacing
                        st.write(f"📈 **{title}**")
                        plot_chart(grouped, grouped.columns[0], "value", title, kind="bar")
            else:
                with st.chat_message("model"):
                    st.write("⚠️ 'partyname' column not found in dataset.")

    # ---------- 4) Other queries: fallback to AI or preview ----------
    else:
        if model:
            context = f"Dataset columns: {', '.join(df.columns)}\nSample rows:\n{df.head(5).to_string(index=False)}"
            prompt = f"{context}\nUser: {user_input}\nAnswer concisely using the dataset context."
            resp = model.generate_content(prompt)
            ans = resp.text if resp and resp.text else "I couldn't generate a response."
            with st.chat_message("model"):
                st.markdown(ans)
        else:
            with st.chat_message("model"):
                st.write("I couldn't detect an intent. Here are sample rows from the data:")
                st.dataframe(df.head(10))

    # keep UI consistent
    st.session_state.messages.append({"role":"model","content":"(answered)"})
