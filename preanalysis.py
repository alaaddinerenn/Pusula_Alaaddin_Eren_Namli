import pandas as pd
import streamlit as st

def upload_and_preview_file(uploaded_file):
    """
    Handles file upload and preview.
    Automatically removes unnamed index columns.
    """
    # Read the file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Remove columns starting with "Unnamed"
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    st.success("File uploaded successfully!")
    st.subheader("Data Preview")
    st.write(df)

    return df

def display_data_info(df):
    """
    Displays data information and duplicate check.
    """
    st.subheader("Data Information")
    st.info(f"Number of rows: {df.shape[0]} Number of columns: {df.shape[1]}")

    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        st.warning(f"{duplicate_count} duplicate rows found.")
    else:
        st.info("No duplicate rows found.")

    column_info = pd.DataFrame({
        "Data Type": df.dtypes.astype(str)
    })
    st.write(column_info)

def show_basic_statistics(df):
    """
    Displays basic statistics.
    """
    st.subheader("Basic Statistics")
    st.write(df.describe())

def show_missing_data(df):
    """
    Performs missing data analysis and displays the result.
    """
    st.subheader("Missing Data Analysis")
    missing_data = df.isnull().sum()
    total_missing = missing_data.sum()

    if total_missing == 0:
        st.info("No missing data found ✅")
    else:
        missing_percentage = (missing_data / len(df)) * 100
        missing_data_table = pd.DataFrame({
            "Column Name": missing_data.index,
            "Missing Data Count": missing_data.values,
            "Missing Data (%)": missing_percentage.values
        })
        missing_data_table["Missing Data (%)"] = missing_data_table["Missing Data (%)"].map("{:.2f}".format)
        st.write(missing_data_table)