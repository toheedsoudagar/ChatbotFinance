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
def revenue_mask(df):
    return (
        (df["vouchertype"].str.lower() == "receipt")
        & (df["amount_type"].str.lower() == "cr")
        & (df["amount"] > 0)
    )

# --- Chart Helpers ---
def format_currency(val):
    return f"₹{val:,.0f}"

def plot_chart(df, x_col, y_col, title, kind="bar"):
    fig, ax = plt.subplots(figsize=(8, 4))
    plt.tight_layout(pad=3)

    if kind == "bar":
        bars = ax.bar(df[x_col], df[y_col], color="#4CAF50", alpha=0.9)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                format_currency(h),
                xy=(bar.get_x() + bar.get_width()/2, h),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                fontweight="bold"
            )
    else:
        ax.plot(df[x_col], df[y_col], marker="o", color="#2196F3", linewidth=2)
        for i, v in enumerate(df[y_col]):
            ax.text(i, v, format_currency(v), ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel(x_col.capitalize(), fontsize=10)
    ax.set_ylabel("Amount (₹)", fontsize=10)
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
st.title("💬 Financial Data Chatbot (AI + Context Memory + Summaries)")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_keywords" not in st.session_state:
    st.session_state["last_keywords"] = []

# --- Show previous chat ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat Input ---
user_input = st.chat_input("Ask about your financial data...")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    q = user_input.lower()

    # ========== 1️⃣ Dynamic Keyword-Based Revenue ========== #
    if "revenue" in q or any(w in q for w in ["total", "amount", "show"]):
        exclude_words = {
            "revenue","amount","total","show","what","is","the","for","by","month","year",
            "in","of","from","this","that","wise","need","and","want","data","give","me",
            "require","tell","get","calculate","display","find","list","please","all"
        }

        # Tokenize and filter out filler words
        words = [w for w in re.findall(r"[a-zA-Z]+", q) if w.lower() not in exclude_words and len(w) > 2]
        keywords = [w.lower() for w in words if not w.isdigit()]

        # Save or reuse last context keywords
        if keywords:
            st.session_state["last_keywords"] = keywords
        else:
            keywords = st.session_state.get("last_keywords", [])

        if keywords:
            key_str = "|".join(keywords)
            if "partyname" in df.columns:
                mask = revenue_mask(df) & df["partyname"].str.contains(key_str, case=False, na=False)
                subset = df[mask]
                if subset.empty:
                    with st.chat_message("model"):
                        st.write(f"⚠️ No transactions found for '{' / '.join(keywords)}'.")
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
                        st.write(f"📊 **Total Revenue for '{' / '.join(keywords)}':** {format_currency(total)}")
                        if not monthly.empty:
                            plot_chart(monthly, "month", "value", f"Monthly Trend – {' / '.join(keywords)}", kind="bar")
            else:
                with st.chat_message("model"):
                    st.write("⚠️ 'partyname' column not found in dataset.")
        else:
            with st.chat_message("model"):
                st.write("⚠️ Please specify a valid category or party name.")

    # ========== 2️⃣ Follow-up: Year-wise / Month-wise ========== #
    elif any(kw in q for kw in ["year", "yearwise", "year wise", "annual", "monthwise", "month wise"]):
        last_keywords = st.session_state.get("last_keywords", [])
        if not last_keywords:
            with st.chat_message("model"):
                st.write("Please specify what category or party you meant (e.g., 'Hostel', 'Canteen').")
        else:
            key_str = "|".join(last_keywords)
            if "partyname" in df.columns:
                mask = revenue_mask(df) & df["partyname"].str.contains(key_str, case=False, na=False)
                subset = df[mask]
                if subset.empty:
                    with st.chat_message("model"):
                        st.write(f"⚠️ No data found for '{' / '.join(last_keywords)}'.")
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
                        title = f"Month-wise Revenue – {' / '.join(last_keywords)}"
                    else:
                        grouped = (
                            subset.groupby(subset["date"].dt.year)["amount"]
                            .sum()
                            .reset_index()
                            .rename(columns={"date":"year","amount":"value"})
                        )
                        title = f"Year-wise Revenue – {' / '.join(last_keywords)}"

                    with st.chat_message("model"):
                        # --- 🧮 Summary before chart ---
                        total_rev = grouped["value"].sum()
                        max_row = grouped.loc[grouped["value"].idxmax()]
                        max_label = str(max_row[grouped.columns[0]])
                        max_val = max_row["value"]

                        if "year" in grouped.columns[0].lower():
                            summary = (
                                f"📅 **Year-wise Summary:**\n"
                                f"• Total revenue across all years: {format_currency(total_rev)}\n"
                                f"• Highest in {max_label}: {format_currency(max_val)}"
                            )
                        else:
                            summary = (
                                f"📆 **Month-wise Summary:**\n"
                                f"• Total revenue across all months: {format_currency(total_rev)}\n"
                                f"• Highest in {max_label}: {format_currency(max_val)}"
                            )

                        st.markdown(summary)
                        st.write("")  # spacing
                        st.write(f"📈 **{title}**")
                        plot_chart(grouped, grouped.columns[0], "value", title, kind="bar")
            else:
                with st.chat_message("model"):
                    st.write("⚠️ 'partyname' column not found in dataset.")

    # ========== 3️⃣ General reasoning fallback ========== #
    else:
        context = f"""
        The dataset has {len(df)} rows and columns: {', '.join(df.columns)}.
        Example data:
        {df.head(5).to_string(index=False)}
        """
        prompt = f"{context}\nUser: {user_input}"
        response = model.generate_content(prompt)
        answer = response.text if response and response.text else "I couldn’t generate a response."
        with st.chat_message("model"):
            st.markdown(answer)

    st.session_state.messages.append({"role": "model", "content": "(answered)"})
