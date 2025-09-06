# Pusula Case Project

## Alaaddin Eren Namlı - nealaadin@gmail.com


## About the Project
This project is designed to provide users with a platform for analyzing datasets and performing preprocessing steps before predictive modeling. The application consists of two main sections:

1. **EDA (Exploratory Data Analysis):** Allows users to explore their datasets, visualize data distributions, and identify patterns.
2. **Clean & Preprocess:** Enables users to clean their datasets by handling missing values, encoding categorical variables, and scaling numerical features.


The project supports CSV and Excel file formats, making it versatile for various datasets. Users can perform tasks such as:
- Handling missing data
- Encoding categorical variables (Label Encoding, One-Hot Encoding)
- Scaling numerical features (Standardization, Normalization)
- Visualizing data insights

## Note
The document for the case project corresponds to the **Document** page in this project.

## Features
- **Data Upload:** Upload datasets in CSV or Excel format.
- **Data Cleaning:** Handle missing values, remove duplicates, and standardize text data.
- **Data Transformation:** Convert columns to appropriate data types and encode categorical variables.
- **Visualization:** Generate charts and graphs to understand data distributions and relationships.

## Dependencies
The project requires the following Python libraries:
- `streamlit`
- `pandas`
- `matplotlib`
- `seaborn`
- `openpyxl`
- `scikit-learn`

You can install the dependencies using the following command:
```bash
pip install -r requirements.txt
```

## How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/pusula-case.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the Streamlit application:
   ```bash
   streamlit run Home_Page.py
   ```
4. Open the provided URL in your browser to access the application.

## Folder Structure
```
Pusula_Alaaddin_Eren_Namli/
├── pages/
│   ├── 01_EDA.py
│   ├── 02_Clean_And_Preprocess.py
│   ├── 03_About.py
│   ├── 04_Document.py
├── data/
│   ├── Talent_Academy_Case_DT_2025.xlsx
│   ├── clean.csv
│   ├── eng.csv
│   ├── lower.csv
│   ├── converted.csv
│   ├── encoded.csv
│   ├── scaled.csv
│   ├── expanded.csv
├── README.md
├── requirements.txt
├── Home_Page.py
├── preanalysis.py
├── preprocessing.py
├── utils.py
```




