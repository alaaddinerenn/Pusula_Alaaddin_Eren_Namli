import streamlit as st

st.set_page_config(page_title="About", layout="centered")

st.title("About the Project")

st.markdown(
    """
    ## Project Purpose
    This project aims to provide detailed **exploratory data analysis (EDA)** for health data. 
    Users can perform various analyses on the uploaded dataset, fill in missing values, and make the data more meaningful.

    ---

    ## Features
    - **Data Upload:** Upload CSV or Excel files for analysis.
    - **General Information:** Analyze dataset size, column types, and missing values.
    - **Numerical and Categorical Analysis:** Perform detailed analysis and visualization for numerical and categorical variables.
    - **Correlation Analysis:** Examine relationships between numerical variables using a correlation matrix.
    - **Data Cleaning:** Detect and clean erroneous or outlier values in the dataset.
    - **Handling Missing Values:** Fill in missing values using various methods (mean, median, mode, etc.).
    - **Encoding:** Convert categorical variables into numerical values (Label Encoding, One-Hot Encoding).
    - **Standardization and Normalization:** Scale and normalize numerical variables.
    - **Duplicate Detection:** Identify and clean duplicate rows in the dataset.

    ---

    ## Usage
    1. **Data Upload:** Upload a CSV or Excel file on the main page.
    2. **Analysis Selection:** Choose the type of analysis from the relevant page.
    3. **Review Results:** Examine the results through visualizations and tables.

    ---

    ## Contact
    For feedback or suggestions about this project, please get in touch:
    - **Developer:** Alaaddin Eren Namlı
    - **Email:** alaaddin.namli@example.com

    """
)