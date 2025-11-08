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

# --- Revenue Rule ---
def revenue_mask(df):
    return (
        (df["vouchertype"].str.lower() == "receipt")
        & (df["amount_type"].str.lower() == "cr")
        & (df["amount"] > 0)
    )

# --- Chart Helper ---
def format_currency(val):
    return f"₹{val:,.0f}"

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

# --- Load Data ---
DATA_PATH = "data/Revenue File.xlsx"
df = load_data(DATA_PATH)

# --- Initialize Gemini ---
model = genai.GenerativeModel("gemini-2.0-flash")

# --- Streamlit Setup ---
st.set_page_config(page_title="Financial Chatbot", page_icon="💬", layout="centered")
st.title("💬 Financial Data Chatbot (AI + Dynamic Category Search)")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat Input ---
user_input = st.chat_input("Ask about your financial data...")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    q = user_input.lower()

    # ========== 1️⃣ Dynamic Keyword-Based Revenue by Party ========== #
    # Automatically detect if user is asking for revenue of a category or party
    if "revenue" in q and not re.search(r"\b(20\d{2})\b", q):
        # Extract keyword candidates (e.g., hostel, mess, contractor, rent, etc.)
        exclude_words = {
            "revenue", "amount", "total", "show", "what", "is", "the", "for",
            "by", "month", "year", "in", "of", "from", "this", "that"
        }
        words = [w for w in re.findall(r"[a-zA-Z]+", q) if w not in exclude_words and len(w) > 2]
        keywords = [w for w in words if not w.isdigit()]
        keyword_str = "|".join(keywords)

        if keywords:
            # Filter data dynamically
            if "partyname" in df.columns:
                mask = revenue_mask(df) & df["partyname"].str.contains(keyword_str, case=False, na=False)
                subset = df[mask]
                total = subset["amount"].sum()
                if subset.empty:
                    with st.chat_message("model"):
                        st.write(f"⚠️ No transactions found for '{' / '.join(keywords)}'.")
                else:
                    monthly = (
                        subset.groupby(subset["date"].dt.to_period("M"))["amount"]
                        .sum()
                        .reset_index()
                        .rename(columns={"date": "month", "amount": "revenue"})
                    )
                    monthly["month"] = monthly["month"].astype(str)
                    with st.chat_message("model"):
                        st.write(f"📊 **Total Revenue for '{' / '.join(keywords)}':** {format_currency(total)}")
                        if not monthly.empty:
                            plot_revenue_chart(monthly, "month", "revenue", f"Revenue Trend for {' / '.join(keywords)}", kind="bar")
            else:
                with st.chat_message("model"):
                    st.write("⚠️ 'partyname' column not found in dataset.")
        else:
            # No valid keyword found, fallback to general reasoning
            context = f"""
            The dataset has {len(df)} rows and columns: {', '.join(df.columns)}.
            Example data:
            {df.head(5).to_string(index=False)}
            """
            prompt = f"{context}\nUser: {user_input}"
            response = model.generate_content(prompt)
            with st.chat_message("model"):
                st.markdown(response.text if response and response.text else "I couldn’t generate a response.")

    # ========== 2️⃣ Year-specific "Revenue by Month" ========== #
    elif ("month" in q and re.search(r"\b(20\d{2})\b", q)):
        match = re.search(r"\b(20\d{2})\b", q)
        year = int(match.group(1))
        if "date" in df.columns:
            dfy = df[df["date"].dt.year == year]
            if dfy.empty:
                with st.chat_message("model"):
                    st.write(f"⚠️ No transactions found for {year}.")
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
                    st.write(f"📈 **Revenue by Month for {year}**")
                    plot_revenue_chart(monthly_rev, "month", "revenue", f"Monthly Revenue – {year}", kind="bar")
        else:
            with st.chat_message("model"):
                st.write("⚠️ No 'date' column found in dataset.")

    # ========== 3️⃣ Fallback / Other Queries ========== #
    else:
        context = f"""
        The dataset has {len(df)} rows and columns: {', '.join(df.columns)}.
        Example data:
        {df.head(5).to_string(index=False)}
        Use reasoning to answer generic questions about data.
        """
        prompt = f"{context}\nUser: {user_input}"
        response = model.generate_content(prompt)
        answer = response.text if response and response.text else "I couldn’t generate a response."
        with st.chat_message("model"):
            st.markdown(answer)

    st.session_state.messages.append({"role": "model", "content": "(answered)"})
