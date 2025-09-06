import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler

def file_loader(uploaded_file):
    """
    Reads the uploaded file and returns a DataFrame.
    Supported file types: CSV, Excel.
    """
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")

# Replace Turkish characters with English characters
def replace_turkish_characters(value):
    if isinstance(value, str):
        replacements = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
        return value.translate(replacements)
    return value

def format_dorsalji(value):
    if isinstance(value, str) and value.startswith('dorsalji-'):
        parts = value.split('+')
        if len(parts) == 1:
            return value
        return ','.join([f'dorsalji-{parts[i]}' for i in range(1, len(parts))])
    return value

def manipulate_manually(df):
    """
    Performs manual operations on the dataset.
    Fixes inconsistencies.
    """
    df['TedaviAdi'] = df['TedaviAdi'].str.replace('dorsalji -', 'dorsalji-', regex=False)
    df['TedaviAdi'] = df['TedaviAdi'].apply(format_dorsalji)
    df['TedaviAdi'] = df['TedaviAdi'].replace('dorsalji +servikal myelomalazi', 'dorsalji,servikal myelomalazi')
    df['TedaviAdi'] = df['TedaviAdi'].replace('sağ+ sol humerus kırığı ', 'sağ ve sol humerus kırığı ')
    df['TedaviAdi'] = df['TedaviAdi'].replace('dirsek eklem çıkığı+kontraktürü', 'dirsek eklem çıkığı, dirsek eklem kontraktürü')
    df['TedaviAdi'] = df['TedaviAdi'].replace('sağ kalça ve bacak kuvvetsizliği+ağrısı', 'sağ kalça ve bacak kuvvetsizliği, sağ kalça ve bacak ağrısı')
    df['TedaviAdi'] = df['TedaviAdi'].replace('el bi̇leği̇ tendi̇ni̇t+gangli̇on', 'el bi̇leği̇ tendi̇ni̇t, el bi̇leği̇ gangli̇on')
    df['TedaviAdi'] = df['TedaviAdi'].str.replace('gonartroz-meniskopati', 'gonartroz,meniskopati', regex=True)
    df['TedaviAdi'] = df['TedaviAdi'].str.replace('-erken rehabilitasyon', ',erken rehabilitasyon', regex=True)
    df['TedaviAdi'] = df['TedaviAdi'].replace('dorsalji-dorsal', 'dorsalji,dorsal')
    df['TedaviAdi'] = df['TedaviAdi'].replace('dorsalji- koksiks', 'dorsalji,koksiks')
    df['TedaviAdi'] = df['TedaviAdi'].str.replace('+', ',', regex=False)
    df['TedaviAdi'] = df['TedaviAdi'].str.strip()
    df['Tanilar'] = df['Tanilar'].str.replace(',,', ',', regex=False)
    df['Tanilar'] = df['Tanilar'].str.replace('"', '', regex=False)
    df['Tanilar'] = df['Tanilar'].str.strip()
    str_cols = df.select_dtypes(include='object').columns
    df[str_cols] = df[str_cols].apply(lambda x: x.str.strip())
    st.success("Manual data manipulation completed.")
    st.write(df)
    return df

def scale_columns(df):
    """
    Scales selected columns based on user input and scaling method.

    Args:
        df (pd.DataFrame): The dataframe to process.

    Returns:
        pd.DataFrame: The scaled dataframe.
    """
    st.subheader("Scale Columns")
    st.info("Select one or more columns and choose a scaling method.")

    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

    if not numeric_cols:
        st.warning("No numerical columns found.")
        return df

    selected_cols = st.multiselect("Select columns to scale:", numeric_cols)
    scaler_option = st.radio(
        "Select a scaling method:",
        ("Standard Scaler", "MinMax Scaler", "Robust Scaler", "MaxAbs Scaler")
    )

    if st.button("Scale Selected Columns"):
        scaler = None
        if scaler_option == "Standard Scaler":
            scaler = StandardScaler()
        elif scaler_option == "MinMax Scaler":
            scaler = MinMaxScaler()
        elif scaler_option == "Robust Scaler":
            scaler = RobustScaler()
        elif scaler_option == "MaxAbs Scaler":
            scaler = MaxAbsScaler()

        if scaler and selected_cols:
            df[selected_cols] = scaler.fit_transform(df[selected_cols])
            st.success(f"Columns {', '.join(selected_cols)} have been scaled using {scaler_option}.")
            st.write(df)

    return df


