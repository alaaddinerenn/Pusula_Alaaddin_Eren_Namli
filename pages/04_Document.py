import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from preanalysis import display_data_info, show_basic_statistics, show_missing_data

st.set_page_config(page_title="Document Analysis", layout="centered")

# -------------------------------
# 1. Dataset Initialization
# -------------------------------
# Center align the title
st.markdown("""
<h1 style="text-align: center;">📄 Document</h1>
""", unsafe_allow_html=True)

# Name and Email
st.subheader("Developer: Alaaddin Eren Namlı")
st.subheader('Email: nealaadin@gmail.com')

# Add a project description and preprocessing steps below the title
st.markdown("""
<div style="text-align: justify; font-family: Arial, sans-serif; line-height: 1.6;">
<h3>About This Project</h3>
<p>This project provides users with a platform to analyze datasets and perform preprocessing steps before predictive modeling. The application consists of two main sections: <strong>EDA (Exploratory Data Analysis)</strong> and <strong>Clean & Preprocess</strong>. These sections are implemented to support any dataset. Users can upload their files in CSV or Excel format, analyze the data, handle missing values, and make the data more meaningful. Additionally, users can convert categorical variables into numerical ones (using Label Encoding or One-Hot Encoding) and scale numerical variables (using Standardization or Normalization). On this page, I will explain step by step how I analyzed the data, the transformations I applied, and the different datasets I obtained. Using these datasets, I will answer specific questions and create visualizations. You can also generate the datasets I obtained using the <strong>Clean & Preprocess</strong> section and analyze them in the <strong>EDA</strong> section.</p>
</div>
""", unsafe_allow_html=True)




st.header('Preprocessing Steps')
st.subheader('Step 1: Data Upload and Initial Inspection')
df = pd.read_excel("data/Talent_Academy_Case_DT_2025.xlsx")
st.write("Initial Data Preview:")
st.dataframe(df)
show_basic_statistics(df)
display_data_info(df)
st.markdown(
    """
    <div style="text-align: justify;">
    First, I examined the data. I noticed that some columns contained missing values and others had meaningless data. Then, I realized that there were too many duplicate records. In the Alerji column, there was inconsistency due to case sensitivity issues. The TedaviAdi column contained too many different values and was written inconsistently. The TedaviSuresi and UygulamaSuresi columns were not numeric, but I realized that converting them into numeric values would make them more functional.
    </div>
    """,
    unsafe_allow_html=True
)

st.subheader('Step 2: Cleaning the Data and Removing Duplicates')
show_missing_data(df)
st.markdown(
    """
    <div style="text-align: justify;">
    Then, using the <strong>Clean & Preprocess</strong> section, I handled the missing values and duplicate records. I dropped missing rows in the Cinsiyet, KanGrubu, Bolum and UygulamaYerleri columns, as these columns are crucial for analysis. For other columns with missing values, I filled Alerji and KronikHastalik with 'Yok' (None) and filled Tanilar column with 'tanimlanmamis' (undefined). I also removed duplicate records, which significantly reduced the dataset size.
    </div>
    """,
    unsafe_allow_html=True
)
clean_df = pd.read_csv("data/clean.csv", index_col=0)
st.subheader("Cleaned Data Preview:")
st.write(clean_df)

st.subheader('Step 3: Converting Turkish Characters to English Characters')
display_data_info(clean_df)
st.markdown(
    """
    <div style="text-align: justify;">
    I converted Turkish characters to English characters using the <strong>Clean & Preprocess</strong> page to ensure consistency in text data.
    </div>
    """,
    unsafe_allow_html=True
)
eng_df = pd.read_csv("data/eng.csv", index_col=0)
st.subheader("Converted Data Preview:")
st.write(eng_df)

st.subheader('Step 4: Converting Columns to Lowercase for Consistency and Manual Data Manipulation')
st.markdown(
    """
    <div style="text-align: justify;">
    I converted values in the Alerji, TedaviAdi, and Tanilar columns to lowercase using the <strong>Clean & Preprocess</strong> page to ensure consistency in text data. Also, I performed manual data manipulation to correct inconsistencies in the TedaviAdi and Tanilar columns.
    </div>
    """,
    unsafe_allow_html=True
)
lower_df = pd.read_csv("data/lower.csv", index_col=0)
st.subheader("Converted Data Preview:")
st.write(lower_df)

st.subheader('Step 5: Converting Columns to Appropriate Data Types')
st.markdown(
    """
    <div style="text-align: justify;">
    I converted the TedaviSuresiSeans and UygulamaSuresi columns to numeric types using the <strong>Clean & Preprocess</strong> page, which allowed for more effective analysis.
    </div>
    """,
    unsafe_allow_html=True
)
converted_df = pd.read_csv("data/converted.csv", index_col=0)
st.subheader("Converted Data Preview:")
st.write(converted_df)

st.subheader('Step 6: Encoding Categorical Features')
st.markdown(
    """
    <div style="text-align: justify;">
    I applied One-Hot Encoding (MultiLabelBinarizer) to the categorical columns using the <strong>Clean & Preprocess</strong> page. This transformation allowed me to convert categorical variables into a format suitable for machine learning algorithms.
    </div>
    """,
    unsafe_allow_html=True
)
encoded_df = pd.read_csv("data/encoded.csv", index_col=0)
st.subheader("Converted Data Preview:")
st.write(encoded_df)

st.subheader('Step 7: Standardizing or Normalizing Numerical Features')
st.markdown(
    """
    <div style="text-align: justify;">
    I scaled the numerical columns using Standardization (StandardScaler) via the <strong>Clean & Preprocess</strong> page. This step ensured that numerical features were on a similar scale, which is important for distance-based machine learning algorithms.
    </div>
    """,
    unsafe_allow_html=True
)
scaled_df = pd.read_csv("data/scaled.csv", index_col=0)
st.subheader("Converted Data Preview:")
st.write(scaled_df)

st.markdown(
    """
    <div style="text-align: justify;">
    Here is the final dataset after all preprocessing steps. You can use this dataset for further analysis or predictive modeling.
    </div>
    """,
    unsafe_allow_html=True
)


# Load the dataset directly using pandas
# Replace 'your_file_path.csv' with the actual file path
df = pd.read_csv("data/expanded.csv", index_col=0) # The first column is an index

# Questions Section
# -------------------------------
st.markdown("<h2 style='text-align: center;'>📌 Data Analysis with Questions</h2>", unsafe_allow_html=True)

# Question 1: What is the most common diagnosis?
st.markdown("### 1. What is the most common diagnosis?")
if df is not None and 'Tanilar' in df.columns:
    diagnosis_counts = df['Tanilar'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=diagnosis_counts.values, y=diagnosis_counts.index, ax=ax, palette="viridis")
    ax.set_title("Top 10 Most Common Diagnoses")
    ax.set_xlabel("Count")
    ax.set_ylabel("Diagnosis")
    st.pyplot(fig)
    st.write("The most common diagnosis is:", diagnosis_counts.idxmax())
else:
    st.info("The column 'Tanilar' is not available in the dataset.")

# Question 2: What is the average treatment duration?
st.markdown("### 2. What is the average treatment duration?")
if 'TedaviSuresiSeans' in df.columns:
    st.write(df['TedaviSuresiSeans'].describe())
    avg_duration = df['TedaviSuresiSeans'].mean()
    st.write(f"The average treatment duration is {avg_duration:.2f} sessions.")

# Question 3: Are there any correlations between numerical variables?
st.markdown("### 3. Are there any correlations between numerical variables?")
numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
if len(numeric_cols) > 1:
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)
else:
    st.info("Not enough numerical columns for correlation analysis.")

# Question 4: What is the distribution of allergies?
st.markdown("### 4. What is the distribution of allergies?")
if 'Alerji' in df.columns:
    allergy_counts = df['Alerji'].value_counts().head(10)
    fig, ax = plt.subplots()
    sns.barplot(x=allergy_counts.values, y=allergy_counts.index, ax=ax)
    ax.set_title("Top 10 Allergies")
    st.pyplot(fig)
else:
    st.info("The column 'Alerji' is not available in the dataset.")

# Question 5: What are the most common allergies by gender?
st.markdown("### 5. What are the most common allergies by gender?")
if 'Cinsiyet' in df.columns and 'Alerji' in df.columns:
    gender_allergy_counts = df.groupby(['Cinsiyet', 'Alerji']).size().reset_index(name='Count')
    top_gender_allergies = gender_allergy_counts.sort_values(by='Count', ascending=False).groupby('Cinsiyet').head(10)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=top_gender_allergies,
        x='Count',
        y='Alerji',
        hue='Cinsiyet',
        dodge=True,
        palette="Set2",
        ax=ax
    )
    ax.set_title("Top Allergies by Gender")
    ax.set_xlabel("Count")
    ax.set_ylabel("Allergy")
    st.pyplot(fig)

    # Display a DataFrame with counts and percentages
    gender_allergy_summary = (
        gender_allergy_counts
        .groupby(['Cinsiyet', 'Alerji'])['Count']
        .sum()
        .reset_index()
    )
    gender_allergy_summary['Percentage'] = (
        gender_allergy_summary['Count'] / gender_allergy_summary['Count'].sum() * 100
    )

    # Separate DataFrames for males and females
    male_allergy_summary = (
        gender_allergy_summary[gender_allergy_summary['Cinsiyet'] == 'Erkek']
        .sort_values(by='Count', ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    female_allergy_summary = (
        gender_allergy_summary[gender_allergy_summary['Cinsiyet'] == 'Kadin']
        .sort_values(by='Count', ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    # Adjust percentage formatting to .3f
    male_allergy_summary['Percentage'] = male_allergy_summary['Percentage'].map(lambda x: f"{x:.3f}")
    female_allergy_summary['Percentage'] = female_allergy_summary['Percentage'].map(lambda x: f"{x:.3f}")

    # Display side-by-side DataFrames
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Male Allergies**")
        st.dataframe(male_allergy_summary)

    with col2:
        st.markdown("**Female Allergies**")
        st.dataframe(female_allergy_summary)

    
    st.write("The table above shows the total count and percentage of each allergy by gender, sorted by the highest counts.")
    st.write("According to the chart, we can observe that women tend to have pollen, novalgin, gripin and sucuk allergies, men ,on the other hand, tend to have dust(toz) allergy.")

else:
    st.info("The columns 'Cinsiyet' and/or 'Alerji' are not available in the dataset.")


