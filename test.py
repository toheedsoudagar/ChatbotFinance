import streamlit as st
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import re

# --- Load environment variables ---
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# --- Load Excel Data ---
def load_data(file_path: str, sheet_name: str = "Revenue") -> pd.DataFrame:
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df.dropna(how="all", inplace=True)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

# --- Revenue Logic ---
def calculate_revenue(df: pd.DataFrame):
    """Apply business rule for revenue: Receipt + Cr + amount > 0"""
    mask = (
        (df["vouchertype"].str.lower() == "receipt")
        & (df["amount_type"].str.lower() == "cr")
        & (df["amount"] > 0)
    )
    rev = df[mask]
    total = rev["amount"].sum()

    by_year = (
        rev.groupby(rev["date"].dt.year)["amount"]
        .sum()
        .reset_index()
        .rename(columns={"date": "year", "amount": "revenue"})
    )

    by_month = (
        rev.groupby(rev["date"].dt.to_period("M"))["amount"]
        .sum()
        .reset_index()
        .rename(columns={"date": "month", "amount": "revenue"})
    )
    by_month["month"] = by_month["month"].astype(str)
    return total, by_year, by_month

# --- Load Data ---
DATA_PATH = "data/Revenue File.xlsx"
df = load_data(DATA_PATH)
total_revenue, revenue_by_year, revenue_by_month = calculate_revenue(df)

# --- Initialize Gemini ---
model = genai.GenerativeModel("gemini-2.0-flash")

# --- Streamlit Setup ---
st.set_page_config(page_title="Financial Chatbot", page_icon="💬", layout="centered")
st.title(" Financial Data Chatbot (AI + Visual Insights)")

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Show previous chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Helper function: format large values nicely ---
def format_currency(val):
    return f"₹{val:,.0f}"

# --- Improved chart with numeric annotations and clean layout ---
def plot_revenue_chart(df, x_col, y_col, title, kind="bar"):
    fig, ax = plt.subplots(figsize=(8, 4))
    plt.tight_layout(pad=3)

    if kind == "bar":
        bars = ax.bar(df[x_col], df[y_col], color="#4CAF50", alpha=0.9)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                format_currency(height),
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                fontweight="bold",
                color="black",
            )
    else:
        ax.plot(df[x_col], df[y_col], marker="o", color="#2196F3", linewidth=2)
        for i, val in enumerate(df[y_col]):
            ax.text(i, val, format_currency(val), ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel(x_col.capitalize(), fontsize=10)
    ax.set_ylabel("Revenue (₹)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    st.pyplot(fig)

# --- Revenue mask function ---
def revenue_mask(df):
    return (
        (df["vouchertype"].str.lower() == "receipt")
        & (df["amount_type"].str.lower() == "cr")
        & (df["amount"] > 0)
    )

# --- Chat Input ---
user_input = st.chat_input("Ask about your financial data...")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    q = user_input.lower()
    show_chart = False
    chart_type = None

    # --- Detect revenue-related visual queries ---
    if "revenue" in q or "receipt" in q:
        # Check for "by month for year"
        if ("month" in q and re.search(r"\b(20\d{2})\b", q)):
            match = re.search(r"\b(20\d{2})\b", q)
            year = int(match.group(1))
            if "date" in df.columns:
                dfy = df[df["date"].dt.year == year]
                if dfy.empty:
                    with st.chat_message("model"):
                        st.write(f"⚠️ No transactions found for {year}. Please try another year present in the dataset.")
                else:
                    monthly_rev = (
                        dfy.loc[revenue_mask(dfy)]
                        .groupby(dfy["date"].dt.month)["amount"]
                        .sum()
                        .reset_index()
                    )
                    monthly_rev.columns = ["month_num", "revenue"]
                    monthly_rev["month"] = monthly_rev["month_num"].map({
                        1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                        7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"
                    })
                    with st.chat_message("model"):
                        st.write(f"📊 **Revenue by Month for {year}**")
                        plot_revenue_chart(
                            monthly_rev, "month", "revenue",
                            f"Monthly Revenue Trend – {year}", kind="bar"
                        )
            else:
                with st.chat_message("model"):
                    st.write("⚠️ No 'date' column found in the dataset.")
        # Generic monthly/yearly charts
        elif "month" in q:
            show_chart = True
            chart_type = "month"
        elif "year" in q:
            show_chart = True
            chart_type = "year"

        context = f"""
        The dataset contains financial transactions.

        Business rule for REVENUE:
        Include only rows where vouchertype = 'Receipt', amount_type = 'Cr', and amount > 0.

        Computed summaries:
        • Total Revenue: {total_revenue:,.2f}
        • Revenue by Year:
        {revenue_by_year.to_string(index=False)}
        • Revenue by Month:
        {revenue_by_month.to_string(index=False)}
        """

    else:
        context = f"""
        The dataset has {len(df)} rows and columns: {', '.join(df.columns)}.
        Example data:
        {df.head(5).to_string(index=False)}
        Use reasoning to answer generic questions about data.
        """

    # --- Generate AI response ---
    prompt = f"{context}\nUser: {user_input}"
    response = model.generate_content(prompt)
    answer = response.text if response and response.text else "I couldn’t generate a response."

    # --- Display AI Answer + Chart ---
    with st.chat_message("model"):
        st.markdown(answer)

        # Fallback for general charts
        if show_chart:
            if chart_type == "year" and not revenue_by_year.empty:
                st.write("📊 **Revenue by Year (₹)**")
                plot_revenue_chart(revenue_by_year, "year", "revenue", "Revenue by Year")

            elif chart_type == "month" and not revenue_by_month.empty:
                st.write("📈 **Revenue by Month (₹)**")
                plot_revenue_chart(revenue_by_month, "month", "revenue", "Revenue by Month", kind="line")

    st.session_state.messages.append({"role": "model", "content": answer})
