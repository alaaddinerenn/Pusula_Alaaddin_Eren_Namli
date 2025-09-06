import re
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from preanalysis import show_basic_statistics
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

def handle_missing_and_duplicates(df):
    """
    Provides options for handling missing data and duplicates.
    For numeric columns: Mean, Median, Zero, User Input, KNNImputer.
    For categorical columns: Mode, User Input.
    """
    st.subheader("Handle Missing Data and Duplicates")

    # Save the original dataset before processing
    before_df = df.copy()

    # 🔹 Handle duplicates
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        st.warning(f"{duplicate_count} duplicate rows found.")
        duplicate_col = st.selectbox("Select a column for duplicate check:", options=["Entire dataset"] + df.columns.tolist(), key="duplicate_col")
        if duplicate_col == "Entire dataset":
            if st.checkbox("Remove duplicate rows for the entire dataset"):
                df = df.drop_duplicates()
                st.success("Duplicate rows for the entire dataset have been removed.")
        else:
            if st.checkbox("Remove duplicate rows based on the selected column"):
                df = df.drop_duplicates(subset=[duplicate_col])
                st.success(f"Duplicate rows based on column {duplicate_col} have been removed.")
    else:
        st.info("No duplicate rows found.")

    # 🔹 Handle missing data
    missing_data = df.isnull().sum()
    missing_columns = missing_data[missing_data > 0].index.tolist()

    if missing_columns:
        st.warning(f"Missing data found in {len(missing_columns)} columns.")
        actions = {}
        user_inputs = {}

        for col in missing_columns:
            st.write(f"Column {col} has {missing_data[col]} missing values.")
            col_type = df[col].dtype

            if pd.api.types.is_numeric_dtype(col_type):
                actions[col] = st.selectbox(
                    f"Select an action for column {col}:",
                    (
                        "Do nothing",
                        "Fill with mean",
                        "Fill with median",
                        "Fill with zero",
                        "Fill with user input",
                        "Fill with KNNImputer"
                    ),
                    key=f"action_{col}"
                )
            else:  # categorical
                actions[col] = st.selectbox(
                    f"Select an action for column {col}:",
                    (
                        "Do nothing",
                        "Fill with mode",
                        "Fill with user input",
                        "Drop rows"
                    ),
                    key=f"action_{col}"
                )

            # Input field for user input option
            if actions[col] == "Fill with user input":
                user_inputs[col] = st.text_input(f"Enter a value for column {col}:", key=f"user_input_{col}")

        if st.button("Apply selected actions"):
            for col, action in actions.items():
                if action == "Fill with mean":
                    imputer = SimpleImputer(strategy="mean")
                    df[[col]] = imputer.fit_transform(df[[col]])
                elif action == "Fill with median":
                    imputer = SimpleImputer(strategy="median")
                    df[[col]] = imputer.fit_transform(df[[col]])
                elif action == "Fill with zero":
                    df[col] = df[col].fillna(0)
                elif action == "Fill with mode":
                    imputer = SimpleImputer(strategy="most_frequent")
                    df[[col]] = imputer.fit_transform(df[[col]])
                elif action == "Fill with user input":
                    if col in user_inputs and user_inputs[col] != "":
                        df[col] = df[col].fillna(user_inputs[col])
                elif action == "Fill with KNNImputer":
                    imputer = KNNImputer(n_neighbors=5)
                    df[[col]] = imputer.fit_transform(df[[col]])

                elif action == "Drop rows":
                    df = df.dropna(subset=[col])

            st.success("Selected actions have been successfully applied.")
            return df, True, False

    else:
        st.info("No missing data found.")
        return df, False, True

    return df, False, False

def compare(df_before: pd.DataFrame, df_after: pd.DataFrame):
    """
    Öncesi ve sonrası dataframe'leri sayısal ve kategorik açıdan karşılaştırır.
    """
    st.subheader("🔍 Comparison Before and After Data Cleaning")

    st.subheader("Temizlenmiş Veri Önizlemesi")
    st.write(df_after)

    st.subheader("Temizlenmiş Veri Bilgileri")
    st.info(f"Satır sayısı: {df_after.shape[0]} Sütun sayısı: {df_after.shape[1]}")

    duplicate_count = df_after.duplicated().sum()
    if duplicate_count > 0:
        st.warning(f"{duplicate_count} adet duplicate satır bulundu.")
    else:
        st.info("Duplicate satır bulunmamakta.")

    column_info = pd.DataFrame({
        "Sütun İsmi": df_after.columns,
        "Veri Tipi": df_after.dtypes.astype(str)
    })
    st.write(column_info)
    show_basic_statistics(df_after)

    if "selected_num_cols" not in st.session_state:
        st.session_state["selected_num_cols"] = []
    if "selected_cat_cols" not in st.session_state:
        st.session_state["selected_cat_cols"] = []

    tab_num, tab_cat = st.tabs(["📊 Numerical Data", "📋 Categorical Data"])

    # Target kolonları analiz dışında bırak
    target_cols = [c for c in df_before.columns if c in ['label', 'target'] or c.startswith('target')]
    dfb = df_before.drop(columns=target_cols, errors='ignore')
    dfa = df_after.drop(columns=target_cols, errors='ignore')

    numeric_cols = [c for c in dfb.columns if pd.api.types.is_numeric_dtype(dfb[c])]
    categorical_cols = [c for c in dfb.columns if c not in numeric_cols]

    # --- NUMERICAL TAB ---
    with tab_num:
        st.markdown("### 🧱 Missing Values")
        na_before_num = dfb[numeric_cols].isna().sum()
        na_after_num = dfa[numeric_cols].isna().sum()
        na_df_num = pd.DataFrame({
            "Before": na_before_num,
            "After": na_after_num
        })
        st.dataframe(na_df_num)
        st.bar_chart(na_df_num)

        st.markdown("### 📏 Basic Statistical Summary")
        if len(numeric_cols) == 0:
            st.info("No numerical columns found.")
        else:
            selected_num_cols = st.multiselect(
                "Select numerical columns for summary",
                numeric_cols,
                default=st.session_state["selected_num_cols"] or numeric_cols[:5],
                key="num_cols_summary"
            )
            st.session_state["selected_num_cols"] = selected_num_cols

            if selected_num_cols:
                st.markdown("#### Before")
                st.dataframe(dfb[selected_num_cols].describe().round(3))
                st.markdown("#### After")
                st.dataframe(dfa[selected_num_cols].describe().round(3))

        st.markdown("### 📉 Histogram Comparison")
        if len(numeric_cols) == 0:
            st.info("No numerical columns found.")
        else:
            selected_hist_cols = st.multiselect("Select numerical columns for histogram", numeric_cols, default=numeric_cols[:3], key="hist_num_cols")
            for col in selected_hist_cols:
                fig, axes = plt.subplots(1,2, figsize=(10,4))
                sns.histplot(dfb[col].dropna(), ax=axes[0], color='orange', kde=True)
                axes[0].set_title(f"Before: {col}")
                sns.histplot(dfa[col].dropna(), ax=axes[1], color='blue', kde=True)
                axes[1].set_title(f"After: {col}")
                st.pyplot(fig)

    # --- CATEGORICAL TAB ---
    with tab_cat:
        st.markdown("### 🧱 Missing Values")
        na_before_cat = dfb[categorical_cols].isna().sum()
        na_after_cat = dfa[categorical_cols].isna().sum()
        na_df_cat = pd.DataFrame({
            "Before": na_before_cat,
            "After": na_after_cat
        })
        st.dataframe(na_df_cat)
        st.bar_chart(na_df_cat)

        if len(categorical_cols) == 0:
            st.info("No categorical columns found.")
            return

        selected_cat_cols = st.multiselect(
            "Select categorical columns",
            categorical_cols,
            default=st.session_state["selected_cat_cols"] or categorical_cols[:3],
            key="cat_cols_selection"
        )
        st.session_state["selected_cat_cols"] = selected_cat_cols

        for col in selected_cat_cols:
            st.markdown(f"## {col}")

            # Unique value count
            unique_before = dfb[col].nunique(dropna=False)
            unique_after = dfa[col].nunique(dropna=False)
            st.markdown(f"**Unique Values:** Before: {unique_before} | After: {unique_after}")

            # En sık görülen kategorilerin yüzdesi
            freq_before_pct = dfb[col].fillna("NaN").value_counts(normalize=True).head(5) * 100
            freq_after_pct = dfa[col].fillna("NaN").value_counts(normalize=True).head(5) * 100

            all_categories = freq_before_pct.index.union(freq_after_pct.index)

            freq_before_pct = freq_before_pct.reindex(all_categories).fillna(0)
            freq_after_pct = freq_after_pct.reindex(all_categories).fillna(0)

            # Tam dataset üzerinden yüzdeler
            full_before_pct = dfb[col].fillna("NaN").value_counts(normalize=True) * 100
            for cat in all_categories:
                if freq_before_pct[cat] == 0 and freq_after_pct[cat] > 0:
                    freq_before_pct[cat] = full_before_pct.get(cat, 0)

            freq_pct_df = pd.DataFrame({
                "Before (%)": freq_before_pct,
                "After (%)": freq_after_pct
            }).fillna(0)

            freq_pct_df = freq_pct_df.map(lambda x: f"{x:.2f}%")

            st.markdown("**Most Frequent Categories (Percentage %)**")
            st.dataframe(freq_pct_df)

            # Kategorik frekans karşılaştırması
            top_n = 10
            before_top = dfb[col].fillna("NaN").value_counts().head(top_n).index
            after_top = dfa[col].fillna("NaN").value_counts().head(top_n).index
            all_top_categories = sorted(set(before_top) | set(after_top))

            freq_before_full = dfb[col].fillna("NaN").value_counts().reindex(all_top_categories, fill_value=0)
            freq_after_full = dfa[col].fillna("NaN").value_counts().reindex(all_top_categories, fill_value=0)

            freq_before_top = dfb[col].fillna("NaN").value_counts().head(top_n).reindex(all_top_categories, fill_value=0)
            freq_after_top = dfa[col].fillna("NaN").value_counts().head(top_n).reindex(all_top_categories, fill_value=0)

            for cat in all_top_categories:
                if freq_before_top[cat] == 0 and freq_before_full[cat] > 0:
                    freq_before_top[cat] = freq_before_full[cat]
                if freq_after_top[cat] == 0 and freq_after_full[cat] > 0:
                    freq_after_top[cat] = freq_after_full[cat]

            others_before = dfb[col].fillna("NaN").value_counts().drop(index=all_top_categories, errors='ignore').sum()
            others_after = dfa[col].fillna("NaN").value_counts().drop(index=all_top_categories, errors='ignore').sum()

            freq_before_top = pd.concat([freq_before_top, pd.Series({"Others": others_before})])
            freq_after_top = pd.concat([freq_after_top, pd.Series({"Others": others_after})])

            freq_df = pd.DataFrame({
                "Before": freq_before_top,
                "After": freq_after_top
            }).astype(int)

            st.markdown(f"**Frequency Distribution (Top {top_n} + Others)**")
            st.dataframe(freq_df)

            # Bar chart
            freq_df_sorted = freq_df.sort_values("Before", ascending=False)

            fig, ax = plt.subplots(figsize=(10, 4))
            x = np.arange(len(freq_df_sorted.index))
            bar_width = 0.35

            ax.bar(x - bar_width/2, freq_df_sorted["Before"], width=bar_width, label="Before", color='orange')
            ax.bar(x + bar_width/2, freq_df_sorted["After"], width=bar_width, label="After", color='blue')

            ax.set_xticks(x)
            ax.set_xticklabels(freq_df_sorted.index, rotation=45, ha='right')
            ax.set_ylabel("Frequency")
            ax.set_title(f"{col} - Categorical Value Frequency Comparison")
            ax.legend()

            st.pyplot(fig)

            # Mode karşılaştırma
            mode_before = dfb[col].mode().iloc[0] if not dfb[col].mode().empty else "None"
            mode_after = dfa[col].mode().iloc[0] if not dfa[col].mode().empty else "None"
            st.markdown(f"**Most Frequent Value (Mode):** Before: {mode_before} | After: {mode_after}")

            set_before = set(dfb[col].dropna().unique())
            set_after = set(dfa[col].dropna().unique())
            removed = set_before - set_after

            # Removed categories
            removed = set_before - set_after
            removed_str = ", ".join(map(str, removed)) if removed else "None"
            st.markdown(f"**Removed Categories:** {removed_str}")

def parse_and_clean_columns(df, case="snake_case"):
    """
    Kullanıcının seçtiği kolonlarda sayısal değerleri ayıklar ve kolon isimlerini birim ile birlikte günceller.

    Args:
        df (pd.DataFrame): İşlenecek veri çerçevesi.
        case (str): Kolon isimlerinin formatı. "snake_case", "camelCase" veya "PascalCase" olabilir.

    Örnek:
    'tedavi süresi' sütunu -> 'tedavi_suresi_dk' (snake_case)
    'tedavi süresi' sütunu -> 'tedaviSuresiDk' (camelCase)
    'tedavi süresi' sütunu -> 'TedaviSuresiDk' (PascalCase)
    Hücrelerdeki '10 dakika' -> 10
    """
    df = df.copy()

    st.subheader("Dönüştürmek istediğiniz kolonları seçin")
    st.info("Seçtiğiniz kolonlarda sayısal değerleri ayıklayabilir ve kolon isimlerini birim ile birlikte güncelleyebilirsiniz.")    

    selected_cols = st.multiselect("Kolon seçin", df.columns.tolist())

    if selected_cols:
        if st.button("Kolonları Dönüştür") or st.session_state.get("parse_and_cleaned", False):
            new_cols = {}
            for col in selected_cols:
                # Hücrelerden sayı ve birimi ayıkla
                numbers = df[col].astype(str).str.extract(r"(\d+)").astype(float)
                units = df[col].astype(str).str.extract(r"\d+\s*(\D+)").fillna("").astype(str)
                unit_name = units.iloc[0, 0].strip().lower() if not units.empty else ""

                # Yeni kolon adı oluştur
                if unit_name:
                    if case == "snake_case":
                        new_col_name = f"{col}_{unit_name}"
                    elif case == "camelCase":
                        new_col_name = f"{col}{unit_name.capitalize()}"
                    elif case == "PascalCase":
                        new_col_name = f"{col}{unit_name.capitalize()}"
                else:
                    new_col_name = col

                # Yeni kolonu ekle
                df[new_col_name] = numbers
                new_cols[col] = new_col_name

            # Eski kolonları sil
            df = df.drop(columns=new_cols.keys())
            st.success(f"{len(new_cols)} kolon dönüştürüldü: {list(new_cols.values())}")
            st.session_state["parse_and_cleaned"] = True
            st.subheader("Güncellenmiş Veri Önizlemesi")
            st.write(df)
            column_info = pd.DataFrame({
                "Sütun İsmi": df.columns,
                "Veri Tipi": df.dtypes.astype(str)
            })
            st.write(column_info)
    else:
        st.info("Henüz kolon seçilmedi. Dönüştürme uygulanmadı.")

    return df

def encode_categorical_columns(df):
    """
    Provides options for encoding categorical data.
    Users can choose between Label Encoding or MultiLabelBinarizer (One-Hot Encoding).
    Supports rows with multiple values separated by ",".

    Args:
        df (pd.DataFrame): The dataframe to process.

    Returns:
        pd.DataFrame: The encoded dataframe.
    """
    st.subheader("Encode Categorical Data")
    st.info("Select an option to encode categorical data.")

    categorical_cols = [col for col in df.columns if pd.api.types.is_object_dtype(df[col])]

    if not categorical_cols:
        st.info("No categorical columns found.")
        return df

    selected_cols = st.multiselect("Select categorical columns to encode:", categorical_cols)

    if selected_cols:
        encoding_method = st.radio("Select an encoding method:", ("Label Encoding", "MultiLabelBinarizer (One-Hot Encoding)"), index=1)

        if st.button("Encode Selected Columns") or st.session_state.get("encoded", False):
            for col in selected_cols:
                if encoding_method == "Label Encoding":
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                elif encoding_method == "MultiLabelBinarizer (One-Hot Encoding)":
                    # Split values by ',' and apply MultiLabelBinarizer
                    mlb = MultiLabelBinarizer()
                    df[col] = df[col].fillna("").apply(lambda x: x.split(",") if isinstance(x, str) else [])
                    one_hot = pd.DataFrame(mlb.fit_transform(df[col]), columns=[f"{col}_{cls}" for cls in mlb.classes_], index=df.index)
                    df = pd.concat([df.drop(columns=[col]), one_hot], axis=1)

            st.success("Selected columns have been successfully encoded.")
            st.session_state["encoded"] = True
            st.subheader("Encoded Data")
            st.write(df)
    else:
        st.info("No columns selected. Encoding not applied.")

    return df