import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import math
import os
from PIL import Image


# -----------------------------
# Utility: Load users
# -----------------------------
def load_users(file_path: str = "assets/user.txt") -> list[str]:
    """Load list of users from a text file."""
    try:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                users = [line.strip() for line in f.readlines() if line.strip()]
            return users
        else:
            st.error("⚠️ user.txt file not found!")
            return []
    except Exception as e:
        st.error(f"Error while reading user.txt: {e}")
        return []


# -----------------------------
# Student Score Analysis
# -----------------------------
def calculate_average(scores: list[float]) -> float:
    """Calculate average score from a list of scores."""
    return sum(scores) / len(scores) if scores else 0


def percentage_distribution(scores: list[float]) -> dict[str, int]:
    """Return score distribution in bins."""
    bins = {"90-100": 0, "80-89": 0, "70-79": 0, "60-69": 0, "<60": 0}
    for score in scores:
        if score >= 90:
            bins["90-100"] += 1
        elif score >= 80:
            bins["80-89"] += 1
        elif score >= 70:
            bins["70-79"] += 1
        elif score >= 60:
            bins["60-69"] += 1
        else:
            bins["<60"] += 1
    return bins


def student_analysis():
    """Page: Analyze student scores from Excel file."""
    st.header("📊 Student Score Analysis")

    uploaded_file = st.file_uploader(
        "Upload an Excel file (must contain a column named 'Score')",
        type=["xlsx"]
    )

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        if "Score" not in df.columns:
            st.error("⚠️ Column 'Score' not found in uploaded file.")
            return

        scores = df["Score"].dropna().astype(float).tolist()

        if scores:
            st.write(
                f"Total students: **{len(scores)}**  |  "
                f"Average score: **{round(calculate_average(scores), 2)}**"
            )

            dist = percentage_distribution(scores)
            labels = list(dist.keys())
            values = list(dist.values())

            # Pie chart
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
            ax.axis("equal")
            st.pyplot(fig)

            # Histogram
            st.subheader("Score Distribution Histogram")
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.hist(scores, bins=10, edgecolor="black")
            ax.set_xlabel("Score")
            ax.set_ylabel("Number of Students")
            ax.set_title("Histogram of Scores")
            st.pyplot(fig)


# -----------------------------
# Factorial Calculator
# -----------------------------
def factorial_calculator():
    """Page: Factorial calculator with login system."""
    st.header("➗ Factorial Calculator")

    # Show login/logout state
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = ""

    if not st.session_state.logged_in:
        username = st.text_input("Enter username:")
        if st.button("Login"):
            users = load_users()
            if username in users:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("❌ Unauthorized user")
    else:
        st.success(f"Welcome, {st.session_state.username}!")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.rerun()

        st.divider()

        number = st.number_input(
            "Enter a number:",
            min_value=0,
            max_value=900,
            step=1
        )
        if st.button("Calculate Factorial"):
            result = math.factorial(number)
            st.write(f"Factorial of {number} is **{result}**")


# -----------------------------
# Main App with Tabs
# -----------------------------
def main():
    st.title("🎓 Multi-Tool Dashboard")

    tab1, tab2 = st.tabs(["📊 Student Analysis", "➗ Factorial Calculator"])

    with tab1:
        student_analysis()
    with tab2:
        factorial_calculator()


if __name__ == "__main__":
    main()
