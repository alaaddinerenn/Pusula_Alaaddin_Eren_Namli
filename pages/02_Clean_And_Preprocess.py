from email.policy import default
import streamlit as st
from preanalysis import upload_and_preview_file, display_data_info, show_basic_statistics, show_missing_data
from preprocessing import handle_missing_and_duplicates, compare, parse_and_clean_columns, encode_categorical_columns
from utils import replace_turkish_characters, manipulate_manually, scale_columns

# Title
st.title("Data Analysis Web Application")

# Initialize session state variables
if "cleaned" not in st.session_state:
    st.session_state["cleaned"] = False

if "already_cleaned" not in st.session_state:
    st.session_state["already_cleaned"] = False

if "cleaned_df" not in st.session_state:
    st.session_state["cleaned_df"] = None

if "df" not in st.session_state:
    st.session_state["df"] = None

if "parsed_and_cleaned" not in st.session_state:
    st.session_state["parsed_and_cleaned"] = False

if "encoded" not in st.session_state:
    st.session_state["encoded"] = False

# File upload
uploaded_file = st.file_uploader("Upload a file", type=["csv", "xlsx"])
if uploaded_file is not None:
    try:
        # File upload and preview
        df = upload_and_preview_file(uploaded_file)
        st.session_state["df"] = df

        # Display data information and basic statistics
        display_data_info(df)
        show_basic_statistics(df)

        # Missing data analysis
        show_missing_data(df)

        if not st.session_state.cleaned and not st.session_state.already_cleaned:
            # Data preprocessing (handling missing data and duplicates)
            # Compare before and after cleaning
            st.session_state.cleaned_df, st.session_state.cleaned, st.session_state.already_cleaned = handle_missing_and_duplicates(df)

        if st.session_state.cleaned and not st.session_state.already_cleaned:
            # Show comparison
            compare(st.session_state.df, st.session_state.cleaned_df)

        if st.session_state.cleaned or st.session_state.already_cleaned:
            df = st.session_state.cleaned_df

        # Add a multiselect for users to select columns
        st.subheader("View Unique Values of Columns")
        selected_unique_cols = st.multiselect("Select columns to view unique values:", df.columns.tolist())

        if selected_unique_cols:
            for col in selected_unique_cols:
                unique_values = df[col].unique()
                st.write(f"Unique values in column {col}:")
                st.write(unique_values)

        # Add a checkbox to convert Turkish characters to English characters
        st.subheader("Convert Turkish Characters to English Characters")
        if st.checkbox("Convert Turkish characters to English", value=True):
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].apply(replace_turkish_characters)

            st.success("Turkish characters in all columns have been converted to English characters.")
            st.write(df)

        # Convert values in specific columns to lowercase for consistency
        if 'Alerji' in df.columns and 'TedaviAdi' in df.columns and 'Tanilar' in df.columns:
            st.subheader("Convert Values in Selected Columns to Lowercase")
            selected_lower_cols = st.multiselect("Select columns to convert to lowercase:", df.columns.tolist(), default=['Alerji', 'TedaviAdi', 'Tanilar'], key="lower_case_cols")
            if selected_lower_cols:
                for col in selected_lower_cols:
                    if col in df.columns:
                        df[col] = df[col].str.lower()
            st.success("Values in the selected columns have been converted to lowercase.")
            st.write(df[selected_lower_cols])

        # Manual data manipulation for consistency
        if 'Alerji' in df.columns and 'TedaviAdi' in df.columns and 'Tanilar' in df.columns:
            st.subheader("Manual Data Manipulation for Consistency")
            df = manipulate_manually(df)
            st.success("Values in the selected columns have been manually manipulated for consistency.")
            st.session_state["df"] = df

        # Column renaming and cleaning button and process
        case_option = st.radio("Select format for column names:", ("snake_case", "camelCase", "PascalCase"), index=2)
        df = parse_and_clean_columns(df, case=case_option)
        st.session_state["df"] = df

        df = encode_categorical_columns(df)
        st.session_state["df"] = df

        df = scale_columns(df)
        st.session_state["df"] = df

    except Exception as e:
        # Display error message to the user in case of an error
        st.error(f"An error occurred while reading the file: {e}")
else:
    # Display information message when no file is uploaded
    st.info("Please upload a file.")