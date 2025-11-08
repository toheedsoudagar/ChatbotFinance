import streamlit as st
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt

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
st.title("💬 Financial Data Chatbot (AI + Visual Insights)")

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Show previous chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Helper function: format large values nicely ---
def format_currency(val):
    return f"₹{val:,.0f}"

# --- Draw charts with annotations ---
def plot_revenue_chart(df, x_col, y_col, title, kind="bar"):
    fig, ax = plt.subplots(figsize=(8, 4))
    if kind == "bar":
        bars = ax.bar(df[x_col], df[y_col], color="#4CAF50")
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                format_currency(height),
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color="black",
            )
    else:
        ax.plot(df[x_col], df[y_col], marker="o", color="#2196F3", linewidth=2)
        for i, val in enumerate(df[y_col]):
            ax.text(i, val, format_currency(val), ha="center", va="bottom", fontsize=9)

    ax.set_xlabel(x_col.capitalize())
    ax.set_ylabel("Revenue (₹)")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    st.pyplot(fig)

# --- Chat Input ---
user_input = st.chat_input("Ask about your financial data...")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    user_text = user_input.lower()
    show_chart = False
    chart_type = None

    # --- Detect revenue-related visual queries ---
    if "revenue" in user_text or "receipt" in user_text:
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

        Use these values to answer revenue-related questions.
        """

        # Identify chart intent
        if "month" in user_text:
            show_chart = True
            chart_type = "month"
        elif "year" in user_text:
            show_chart = True
            chart_type = "year"

    else:
        # General data context
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

        if show_chart:
            if chart_type == "year" and not revenue_by_year.empty:
                st.write("📊 **Revenue by Year (₹)**")
                plot_revenue_chart(revenue_by_year, "year", "revenue", "Revenue by Year")

            elif chart_type == "month" and not revenue_by_month.empty:
                st.write("📈 **Revenue by Month (₹)**")
                plot_revenue_chart(revenue_by_month, "month", "revenue", "Revenue by Month", kind="line")

    st.session_state.messages.append({"role": "model", "content": answer})
