# chatbot_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import os
from typing import Tuple

# --- Config ---
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)
    MODEL = genai.GenerativeModel("gemini-2.0-flash")
else:
    MODEL = None  # summaries will be local templates if no API key

DATA_PATH = "data/Revenue File.xlsx"
TABLE_NAME = "transactions"

# --- Utility: load & normalize data ---
def load_data(fp: str) -> pd.DataFrame:
    df = pd.read_excel(fp, sheet_name="Revenue")
    df.columns = (
        df.columns.str.strip()
                   .str.lower()
                   .str.replace(" ", "_")
                   .str.replace(r"[^\w_]", "", regex=True)
    )
    df = df.dropna(how="all")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # ensure amount numeric
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    # normalize text fields safely
    for col in df.select_dtypes(["object"]).columns:
        df[col] = df[col].astype(str)
    return df

df = load_data(DATA_PATH)

# --- Business logic filters ---
def revenue_mask(df: pd.DataFrame) -> pd.Series:
    # vouchertype = Receipt, amount_type = Cr, amount > 0
    if "vouchertype" not in df.columns or "amount_type" not in df.columns:
        return pd.Series([False]*len(df))
    return (
        df["vouchertype"].str.lower().eq("receipt")
        & df["amount_type"].str.lower().eq("cr")
        & (df["amount"] > 0)
    )

def payments_mask(df: pd.DataFrame) -> pd.Series:
    # vouchertype = Payment, amount_type = Dr, amount > 0
    if "vouchertype" not in df.columns or "amount_type" not in df.columns:
        return pd.Series([False]*len(df))
    return (
        df["vouchertype"].str.lower().eq("payment")
        & df["amount_type"].str.lower().eq("dr")
        & (df["amount"] > 0)
    )

# --- Aggregation helpers ---
def total_revenue(df: pd.DataFrame) -> float:
    return df.loc[revenue_mask(df), "amount"].sum()

def revenue_by_year(df: pd.DataFrame) -> pd.DataFrame:
    rev = df.loc[revenue_mask(df)].copy()
    if rev.empty or "date" not in rev.columns:
        return pd.DataFrame(columns=["year", "revenue"])
    res = rev.groupby(rev["date"].dt.year)["amount"].sum().reset_index()
    res.columns = ["year", "revenue"]
    return res

def revenue_by_month(df: pd.DataFrame) -> pd.DataFrame:
    rev = df.loc[revenue_mask(df)].copy()
    if rev.empty or "date" not in rev.columns:
        return pd.DataFrame(columns=["month", "revenue"])
    res = rev.groupby(rev["date"].dt.to_period("M"))["amount"].sum().reset_index()
    res["month"] = res["date"].astype(str)
    res = res.rename(columns={"amount": "revenue"})[["month", "revenue"]]
    return res

def revenue_by_party(df: pd.DataFrame, n=10) -> pd.DataFrame:
    col = "partyname" if "partyname" in df.columns else df.columns[0]
    rev = df.loc[revenue_mask(df)]
    out = rev.groupby(col)["amount"].sum().reset_index().sort_values("amount", ascending=False)
    out = out.rename(columns={col: "party", "amount": "revenue"})
    return out.head(n)

def revenue_by_ledger(df: pd.DataFrame) -> pd.DataFrame:
    key = "ledger_name" if "ledger_name" in df.columns else "ledger"
    rev = df.loc[revenue_mask(df)]
    return rev.groupby(key)["amount"].sum().reset_index().rename(columns={key: "ledger", "amount": "revenue"}).sort_values("revenue", ascending=False)

# Payments / expenses
def payments_by_ledger(df: pd.DataFrame) -> pd.DataFrame:
    key = "ledger_name" if "ledger_name" in df.columns else "ledger"
    pay = df.loc[payments_mask(df)]
    return pay.groupby(key)["amount"].sum().reset_index().rename(columns={key: "ledger", "amount": "payments"}).sort_values("payments", ascending=False)

def top_expenses(df: pd.DataFrame, year=None, n=10) -> pd.DataFrame:
    pay = df.loc[payments_mask(df)].copy()
    if year and "date" in df.columns:
        pay = pay[pay["date"].dt.year == int(year)]
    if "narration" in pay.columns:
        out = pay.groupby("narration")["amount"].sum().reset_index().sort_values("amount", ascending=False)
        out = out.rename(columns={"amount": "total_paid"})
    else:
        out = pay.groupby("ledger_name")["amount"].sum().reset_index().sort_values("amount", ascending=False).rename(columns={"amount": "total_paid", "ledger_name": "ledger"})
    return out.head(n)

# Receipts vs payments monthly/yearly
def receipts_payments_net_month(df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in df.columns:
        return pd.DataFrame(columns=["month","receipts","payments","net"])
    receipts = df.loc[revenue_mask(df)].groupby(df["date"].dt.to_period("M"))["amount"].sum().reset_index(name="receipts")
    payments = df.loc[payments_mask(df)].groupby(df["date"].dt.to_period("M"))["amount"].sum().reset_index(name="payments")
    merged = pd.merge(receipts, payments, on="date", how="outer").fillna(0)
    merged["net"] = merged["receipts"] - merged["payments"]
    merged["month"] = merged["date"].astype(str)
    return merged[["month","receipts","payments","net"]]

def receipts_payments_net_year(df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in df.columns:
        return pd.DataFrame(columns=["year","receipts","payments","net"])
    receipts = df.loc[revenue_mask(df)].groupby(df["date"].dt.year)["amount"].sum().reset_index(name="receipts")
    payments = df.loc[payments_mask(df)].groupby(df["date"].dt.year)["amount"].sum().reset_index(name="payments")
    merged = pd.merge(receipts, payments, left_on="date", right_on="date", how="outer").fillna(0)
    merged = merged.rename(columns={"date":"year"})
    merged["net"] = merged["receipts"] - merged["payments"]
    return merged[["year","receipts","payments","net"]]

# Generic searches / filters
def find_transactions_over(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    return df[df["amount"].abs() > threshold].sort_values("amount", ascending=False)

def find_narration_contains(df: pd.DataFrame, term: str) -> pd.DataFrame:
    if "narration" not in df.columns:
        return pd.DataFrame()
    mask = df["narration"].str.contains(term, case=False, na=False)
    return df[mask].sort_values("date", ascending=False)

def transactions_for_ledger(df: pd.DataFrame, ledger_name: str) -> pd.DataFrame:
    ledger_col = None
    if "ledger_name" in df.columns:
        ledger_col = "ledger_name"
    elif "ledger" in df.columns:
        ledger_col = "ledger"
    else:
        # fallback try to find close match
        for c in df.columns:
            if "ledger" in c:
                ledger_col = c
                break
    if not ledger_col:
        return pd.DataFrame()
    mask = df[ledger_col].str.lower().str.contains(ledger_name.lower(), na=False)
    return df[mask].sort_values("date", ascending=False)

# --- Visualization helpers (professional charts) ---
def format_currency(val):
    return f"₹{val:,.0f}"

def plot_bar_with_values(x, y, xlabel, ylabel, title, rotate_x=45):
    fig, ax = plt.subplots(figsize=(9,4))
    bars = ax.bar(x, y, color="#2b8cbe")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.tick_params(axis="x", rotation=rotate_x)
    # annotate
    for bar in bars:
        h = bar.get_height()
        ax.annotate(format_currency(h), xy=(bar.get_x()+bar.get_width()/2, h), xytext=(0,4), textcoords="offset points", ha="center", fontsize=9)
    # format y ticks
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"₹{v:,.0f}"))
    st.pyplot(fig)

def plot_line_with_values(x, y, xlabel, ylabel, title, rotate_x=45):
    fig, ax = plt.subplots(figsize=(9,4))
    ax.plot(x, y, marker='o', color="#7fc97f", linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.tick_params(axis="x", rotation=rotate_x)
    for i, val in enumerate(y):
        ax.text(i, val, format_currency(val), ha="center", va="bottom", fontsize=9)
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"₹{v:,.0f}"))
    st.pyplot(fig)

# --- Natural summary (use Gemini if available) ---
def make_summary(text_prompt: str) -> str:
    if MODEL is None:
        return text_prompt  # fallback to the prebuilt text
    resp = MODEL.generate_content(text_prompt)
    return resp.text or text_prompt

# --- Streamlit UI ---
st.set_page_config(page_title="Financial Chatbot (Advanced)", layout="centered")
st.title("💬 Financial Chatbot — compute & visualize from your data")

if "messages" not in st.session_state:
    st.session_state.messages = []

# display history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_input = st.chat_input("Ask (examples in your prompt list)...")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role":"user","content":user_input})

    q = user_input.lower()

    # --- Route common intents ---
    if "total revenue" in q or ("total" in q and "revenue" in q) or ("this month" in q and "revenue" in q):
        # detect month/year if present
        if "this month" in q:
            if "date" in df.columns:
                today = pd.Timestamp.now()
                dfm = df[df["date"].dt.year.eq(today.year) & df["date"].dt.month.eq(today.month)]
                val = dfm.loc[revenue_mask(dfm), "amount"].sum()
                out_table = pd.DataFrame([{"period": f"{today.year}-{today.month:02d}", "revenue": val}])
                with st.chat_message("model"):
                    st.write(f"Total revenue for this month: {format_currency(val)}")
                    plot_bar_with_values(out_table["period"], out_table["revenue"], "Period", "Revenue (₹)", "Revenue (month)")
            else:
                with st.chat_message("model"):
                    st.write("No date column to evaluate 'this month'.")
        elif "2024" in q or "2023" in q or "in 2024" in q:
            year = None
            for y in ["2024","2023","2022","2021"]:
                if y in q:
                    year = int(y)
                    break
            if year and "date" in df.columns:
                dfy = df[df["date"].dt.year == year]
                val = dfy.loc[revenue_mask(dfy), "amount"].sum()
                with st.chat_message("model"):
                    st.write(f"Total revenue in {year}: {format_currency(val)}")
            else:
                val = total_revenue(df)
                with st.chat_message("model"):
                    st.write(f"Total revenue: {format_currency(val)}")

    elif "revenue by year" in q or "quarter" in q or "quarterly" in q or "compare this month" in q:
        # quarterly trend or compare months
        if "quarter" in q or "quarterly" in q:
            # quarterly aggregation
            rev = df.loc[revenue_mask(df)].copy()
            if "date" in rev.columns and not rev.empty:
                rev["qtr"] = rev["date"].dt.to_period("Q")
                qtr = rev.groupby("qtr")["amount"].sum().reset_index()
                qtr["qtr"] = qtr["qtr"].astype(str)
                with st.chat_message("model"):
                    st.write("Quarterly revenue trend:")
                    plot_bar_with_values(qtr["qtr"], qtr["amount"], "Quarter", "Revenue (₹)", "Revenue by Quarter")
            else:
                with st.chat_message("model"):
                    st.write("No date data to calculate quarterly trend.")
        elif "compare this month" in q or "last month" in q:
            if "date" in df.columns:
                now = pd.Timestamp.now()
                this = df[df["date"].dt.to_period("M") == pd.Period(now, "M")]
                last = df[df["date"].dt.to_period("M") == pd.Period(now - pd.offsets.MonthBegin(1), "M")]
                val_this = this.loc[revenue_mask(this), "amount"].sum()
                val_last = last.loc[revenue_mask(last), "amount"].sum()
                pct = ((val_this - val_last) / val_last * 100) if val_last != 0 else np.nan
                with st.chat_message("model"):
                    st.write(f"This month: {format_currency(val_this)}  — Last month: {format_currency(val_last)}")
                    if not np.isnan(pct):
                        st.write(f"Change vs last month: {pct:.2f}%")
            else:
                with st.chat_message("model"):
                    st.write("No date column available to compare months.")
        else:
            ydf = revenue_by_year(df)
            if not ydf.empty:
                with st.chat_message("model"):
                    st.write("Revenue by year:")
                    plot_bar_with_values(ydf["year"].astype(str), ydf["revenue"], "Year", "Revenue (₹)", "Revenue by Year")
            else:
                with st.chat_message("model"):
                    st.write("No revenue by year data.")

    elif "mdc canteen" in q or "mDC canteen".lower() in q.lower() or "mdc" in q and ("canteen" in q):
        # example: revenue from MDC Canteen last month
        # find partyname matches
        term = "mdc canteen"
        # detect month phrase
        target_month = None
        if "last month" in q and "date" in df.columns:
            pm = pd.Timestamp.now() - pd.offsets.MonthBegin(1)
            dfm = df[df["date"].dt.to_period("M") == pd.Period(pm, "M")]
        else:
            dfm = df
        mask_party = dfm["partyname"].str.contains(term, case=False, na=False) if "partyname" in dfm.columns else pd.Series([False]*len(dfm))
        val = dfm.loc[mask_party & revenue_mask(dfm), "amount"].sum()
        with st.chat_message("model"):
            st.write(f"Revenue from MDC Canteen: {format_currency(val)} (filtered by available data)")

    elif "hittachi rent" in q or "hitachi rent" in q:
        # show revenue/payments for Hittachi Rent in 2023
        term = "hittachi"
        year = 2023
        mdf = df[df["date"].dt.year == year] if "date" in df.columns else df
        mask_rev = mdf["narration"].str.contains(term, case=False, na=False) if "narration" in mdf.columns else pd.Series([False]*len(mdf))
        rev_val = mdf.loc[mask_rev & revenue_mask(mdf), "amount"].sum()
        pay_val = mdf.loc[mask_rev & payments_mask(mdf), "amount"].sum()
        with st.chat_message("model"):
            st.write(f"Hittachi Rent revenue in {year}: {format_currency(rev_val)}; payments: {format_currency(pay_val)}")

    elif "top 5" in q and ("party" in q or "parties" in q):
        out = revenue_by_party(df, n=5)
        if not out.empty:
            with st.chat_message("model"):
                st.write("Top 5 revenue-generating parties:")
                st.dataframe(out)
        else:
            with st.chat_message("model"):
                st.write("No party revenue data available.")

    elif "payments" in q or "expenses" in q or "paid" in q:
        # various payments-related queries
        if "top 10" in q or "top 5" in q:
            year = None
            for y in ["2024","2023","2022","2021"]:
                if y in q:
                    year = int(y); break
            out = top_expenses(df, year=year, n=10)
            with st.chat_message("model"):
                st.write("Top expenses:")
                st.dataframe(out)
        elif "hittachi rent" in q or "hittachi" in q:
            out = df.loc[df["narration"].str.contains("hittachi", case=False, na=False) & payments_mask(df)]
            with st.chat_message("model"):
                st.write("Payments for Hittachi-related narrations:")
                st.dataframe(out[["date","partyname","ledger_name","narration","amount"]])
        elif "who received the highest payment last month" in q:
            if "date" in df.columns:
                pm = pd.Timestamp.now() - pd.offsets.MonthBegin(1)
                dfm = df[df["date"].dt.to_period("M") == pd.Period(pm, "M")]
                out = dfm.loc[payments_mask(dfm)].groupby("partyname")["amount"].sum().reset_index().sort_values("amount", ascending=False).head(1)
                with st.chat_message("model"):
                    st.write(out if not out.empty else "No payments found last month.")
            else:
                with st.chat_message("model"):
                    st.write("No date column to filter last month.")

    elif "receipts payments" in q or "net balance" in q or "receipts, payments" in q:
        # Show Receipts, Payments and Net balance for a specific month/year
        # Try to parse month textual reference e.g., "march 2023"
        import re
        m = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{4})", q)
        if m:
            mon = m.group(1)
            year = int(m.group(2))
            mon_num = pd.to_datetime(mon, format="%b").month
            dfm = df[(df["date"].dt.year == year) & (df["date"].dt.month == mon_num)]
        elif "march 2023" in q:
            dfm = df[(df["date"].dt.year == 2023) & (df["date"].dt.month == 3)]
        else:
            # fallback use whole data or last month
            dfm = df
        receipts = dfm.loc[revenue_mask(dfm), "amount"].sum()
        payments = dfm.loc[payments_mask(dfm), "amount"].sum()
        net = receipts - payments
        with st.chat_message("model"):
            st.write(f"Receipts: {format_currency(receipts)}")
            st.write(f"Payments: {format_currency(payments)}")
            st.write(f"Net balance: {format_currency(net)}")

    elif "which month had the highest net inflow" in q or "highest net" in q:
        rp = receipts_payments_net_month(df)
        if not rp.empty:
            best = rp.loc[rp["net"].idxmax()]
            with st.chat_message("model"):
                st.write(f"Highest net inflow month: {best['month']} with {format_currency(best['net'])}")
        else:
            with st.chat_message("model"):
                st.write("No monthly net data available.")

    elif "which month has the highest revenue" in q or "month with the highest revenue" in q:
        rbm = revenue_by_month(df)
        if not rbm.empty:
            best = rbm.loc[rbm["revenue"].idxmax()]
            with st.chat_message("model"):
                st.write(f"Month with highest revenue: {best['month']} — {format_currency(best['revenue'])}")
                # show chart too
                plot_bar_with_values(rbm["month"], rbm["revenue"], "Month", "Revenue (₹)", "Monthly Revenue")
        else:
            with st.chat_message("model"):
                st.write("No monthly revenue data available.")

    elif "show monthly revenue trend" in q or "monthly revenue trend" in q:
        rbm = revenue_by_month(df)
        if not rbm.empty:
            with st.chat_message("model"):
                st.write("Monthly revenue trend:")
                plot_line_with_values(rbm["month"], rbm["revenue"], "Month", "Revenue (₹)", "Monthly Revenue Trend")
        else:
            with st.chat_message("model"):
                st.write("No monthly revenue data available.")

    elif "ledger" in q and ("cash" in q or "ledger" in q):
        # Show all transactions for Ledger 'Cash' or named ledger
        term = "cash" if "cash" in q else q.split("ledger")[-1].strip()
        out = transactions_for_ledger(df, term)
        with st.chat_message("model"):
            st.write(f"Transactions for ledger matching '{term}':")
            st.dataframe(out.head(200))

    elif "find transactions above" in q or "above ₹" in q or "above rs" in q:
        import re
        m = re.search(r"above\s*[₹rs]*\s*([\d,]+)", q.replace(",", ""))
        if m:
            th = float(m.group(1).replace(",", ""))
            out = find_transactions_over(df, th)
            with st.chat_message("model"):
                st.write(f"Transactions with absolute amount > {format_currency(th)}:")
                st.dataframe(out[["date","partyname","vouchertype","narration","amount"]].head(200))
        else:
            with st.chat_message("model"):
                st.write("Please specify a numeric threshold, e.g., 'Find transactions above ₹50,000'.")

    elif "narration" in q or "rent" in q:
        out = find_narration_contains(df, "rent")
        with st.chat_message("model"):
            st.write("Entries where narration contains 'Rent':")
            st.dataframe(out[["date","partyname","narration","amount"]].head(200))

    else:
        # fallback: give a short dataframe preview + let Gemini summarize if available
        sample = df.head(10).to_string(index=False)
        context = f"Dataset sample:\n{sample}\nUser question: {user_input}\nAnswer succinctly based on the data sample."
        if MODEL:
            summary = make_summary(context)
            with st.chat_message("model"):
                st.markdown(summary)
        else:
            with st.chat_message("model"):
                st.write("I couldn't detect a specific intent. Here are sample rows from the data:")
                st.dataframe(df.head(10))

    # store assistant placeholder to keep chat UI in sync
    st.session_state.messages.append({"role":"model","content":"(answered)"})


