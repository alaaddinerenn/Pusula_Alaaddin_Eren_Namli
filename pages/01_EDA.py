import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from preanalysis import upload_and_preview_file

st.set_page_config(page_title="EDA", layout="wide")

# -------------------------------
# 1. Dataset Upload
# -------------------------------
st.title("📊 Exploratory Data Analysis")

df = None
uploaded_file = st.file_uploader("Upload a file", type=["csv", "xlsx"])
if uploaded_file is not None:
    # File upload and preview
    df = upload_and_preview_file(uploaded_file)
    st.session_state["df"] = df

    # -------------------------------
    # 2. General Information
    # -------------------------------
    st.subheader("📌 General Information")
    st.write("**Shape:**", df.shape)
    st.write("**Data Types:**")
    st.write(df.dtypes)

    # Missing Values
    st.write("**Missing Values:**")
    missing_data = df.isnull().sum()
    missing_percentages = (missing_data / len(df)) * 100
    missing_df = pd.DataFrame({"Missing Count": missing_data, "Percentage (%)": missing_percentages.round(2)})

    if missing_data.sum() > 0:
        st.dataframe(missing_df)
    else:
        st.info("No missing values found.")

    # Duplicate Rows
    st.write("**Duplicate Rows:**")
    duplicate_rows = df[df.duplicated()]
    if not duplicate_rows.empty:
        st.write(f"**{len(duplicate_rows)} duplicate rows found.**")
        st.dataframe(duplicate_rows)
    else:
        st.info("No duplicate rows found.")

    # Unique Values
    st.subheader("View Unique Values of Columns")
    selected_unique_cols = st.multiselect("Select columns to view unique values:", df.columns.tolist(), default=['Alerji', 'TedaviAdi', 'Tanilar'])

    if selected_unique_cols:
        for col in selected_unique_cols:
            unique_values = df[col].unique()
            st.write(f"Unique values in column {col}:")
            st.write(unique_values)
    
    # Allow user to select columns to explode
    st.subheader("Select Columns to Explode")
    st.info("Choose columns with comma-separated values to split and explode.")
    columns_to_explode = st.multiselect("Select columns to explode:", df.columns.tolist(), default=['Tanilar', 'Alerji', 'KronikHastalik', 'TedaviAdi'])
    df_exploded = None
    if st.checkbox("Apply Explode Operation"):
        df_exploded = df.copy()
        for col in columns_to_explode:
            df_exploded[col] = df_exploded[col].str.split(',').explode(col)

        st.subheader("Expanded DataFrame")
        st.write(df_exploded)

    # -------------------------------
    # 3. Numerical Column Analysis
    # -------------------------------
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    st.subheader("📈 Numerical Variables")
    if len(numeric_cols) > 0:
        selected_cols = st.multiselect("Select numerical variables:", numeric_cols)

        if selected_cols:
            st.subheader("Descriptive Statistics")
            st.write(df[selected_cols].describe())

            graph_options = st.multiselect(
                "Select graph types:",
                ["Histogram", "KDE", "Boxplot", "Histogram with KDE"],
                default=["Histogram"]
            )

            for col in selected_cols:
                st.subheader(f"{col} - Graphs")
                fig, axes = plt.subplots(1, len(graph_options), figsize=(6 * len(graph_options), 4))

                if len(graph_options) == 1:
                    axes = [axes]  # Ensure axes is iterable

                for i, graph in enumerate(graph_options):
                    if graph == "Histogram":
                        sns.histplot(df[col].dropna(), kde=False, ax=axes[i])
                        axes[i].set_title(f"{col} - Histogram")
                    elif graph == "KDE":
                        sns.kdeplot(df[col].dropna(), ax=axes[i])
                        axes[i].set_title(f"{col} - KDE")
                    elif graph == "Boxplot":
                        sns.boxplot(x=df[col], ax=axes[i])
                        axes[i].set_title(f"{col} - Boxplot")
                    elif graph == "Histogram with KDE":
                        sns.histplot(df[col].dropna(), kde=True, ax=axes[i])
                        axes[i].set_title(f"{col} - Histogram with KDE")

                st.pyplot(fig)
        else:
            st.info("Please select one or more numerical variables.")
    else:
        st.info("No numerical variables found.")

    # -------------------------------
    # 4. Categorical Column Analysis
    # -------------------------------
    st.subheader("📊 Categorical Variables")
    if len(categorical_cols) > 0:
        selected_categorical_cols = st.multiselect("Select categorical variables:", categorical_cols)

        if selected_categorical_cols:
            graph_options = st.multiselect(
                "Select graph types:",
                ["Countplot", "Pie Chart", "Frequency Distribution"],
                default=["Countplot"]
            )

            for col in selected_categorical_cols:
                st.subheader(f"{col} - Graphs")

                # Use exploded DataFrame for specific columns
                if col in ["KronikHastalik", "TedaviAdi", "Alerji"]:
                    data_source = df_exploded
                else:
                    data_source = df

                if "Frequency Distribution" in graph_options:
                    value_counts = data_source[col].fillna("None").value_counts()
                    top_10 = value_counts.head(10)
                    others = value_counts.iloc[10:].sum()

                    if others > 0:
                        top_10["Others"] = others

                    percentages = (top_10 / top_10.sum()) * 100
                    freq_df = pd.DataFrame({"Count": top_10, "Percentage (%)": percentages.round(2)})
                    st.write(f"{col} - Frequency Distribution (Top 10 + Others)")
                    st.dataframe(freq_df)

                other_graphs = [graph for graph in graph_options if graph != "Frequency Distribution"]

                if other_graphs:
                    fig, axes = plt.subplots(1, len(other_graphs), figsize=(6 * len(other_graphs), 4))

                    if len(other_graphs) == 1:
                        axes = [axes]  # Ensure axes is iterable

                    for i, graph in enumerate(other_graphs):
                        value_counts = data_source[col].fillna("None").value_counts()
                        top_10 = value_counts.head(10)
                        others = value_counts.iloc[10:].sum()

                        if others > 0:
                            top_10["Others"] = others

                        if graph == "Countplot":
                            if top_10.index.str.len().max() > 15:  # Check if label names are long
                                sns.barplot(y=top_10.index, x=top_10.values, ax=axes[i])
                                axes[i].set_title(f"{col} - Horizontal Bar Plot (Top 10 + Others)")
                            else:
                                sns.countplot(x=pd.Categorical(data_source[col].fillna("None"), categories=top_10.index, ordered=True), ax=axes[i])
                                axes[i].set_title(f"{col} - Countplot (Top 10 + Others)")
                                axes[i].tick_params(axis='x', rotation=45)
                        elif graph == "Pie Chart":
                            percentages = (top_10 / top_10.sum()) * 100
                            if len(top_10) > 5:
                                axes[i].pie(top_10, startangle=90)
                            else:
                                labels_with_percentages = [f"{label} ({percent:.2f}%)" for label, percent in zip(top_10.index, percentages)]
                                axes[i].pie(top_10, labels=labels_with_percentages, startangle=90)
                            axes[i].set_title(f"{col} - Pie Chart (Top 10 + Others)")

                    st.pyplot(fig)
        else:
            st.info("Please select one or more categorical variables.")
    else:
        st.info("No categorical variables found.")

    # -------------------------------
    # 5. Numerical + Categorical Analysis
    # -------------------------------
    st.subheader("📌 Numerical + Categorical Analysis")
    if len(categorical_cols) > 0 and len(numeric_cols) > 0:
        cat_col = st.selectbox("Select a categorical variable:", categorical_cols, key="cat")
        num_col = st.selectbox("Select a numerical variable:", numeric_cols, key="num")

        plot_options = st.multiselect(
            "Select graph types:",
            ["Boxplot", "Violinplot", "Swarmplot"],
            default=["Boxplot"]
        )

        for plot in plot_options:
            fig, ax = plt.subplots(figsize=(6, 4))
            if plot == "Boxplot":
                sns.boxplot(x=df[cat_col], y=df[num_col], ax=ax)
                ax.set_title(f"Distribution of {num_col} by {cat_col} - Boxplot")
            elif plot == "Violinplot":
                sns.violinplot(x=df[cat_col], y=df[num_col], ax=ax)
                ax.set_title(f"Distribution of {num_col} by {cat_col} - Violinplot")
            elif plot == "Swarmplot":
                sns.swarmplot(x=df[cat_col], y=df[num_col], ax=ax)
                ax.set_title(f"Distribution of {num_col} by {cat_col} - Swarmplot")
            plt.xticks(rotation=45)
            st.pyplot(fig)
    else:
        st.info("No suitable columns for numerical + categorical analysis.")

    # -------------------------------
    # 6. Numerical + Numerical Analysis
    # -------------------------------
    st.subheader("📌 Numerical + Numerical Analysis")
    if len(numeric_cols) > 1:
        x_axis = st.selectbox("Select a numerical variable for X axis:", numeric_cols, key="x_axis")
        y_axis = st.selectbox("Select a numerical variable for Y axis:", [col for col in numeric_cols if col != x_axis], key="y_axis")

        plot_options = st.multiselect(
            "Select graph types:",
            ["Scatterplot", "Hexbin", "Correlation"],
            default=["Scatterplot"]
        )

        for plot in plot_options:
            if plot == "Correlation":
                correlation = df[[x_axis, y_axis]].corr().iloc[0, 1]
                st.write(f"**Correlation between {x_axis} and {y_axis}:** {correlation:.2f}")
            else:
                fig, ax = plt.subplots(figsize=(6, 4))
                if plot == "Scatterplot":
                    sns.scatterplot(x=df[x_axis], y=df[y_axis], ax=ax)
                    ax.set_title(f"{x_axis} and {y_axis} - Scatterplot")
                elif plot == "Hexbin":
                    hb = ax.hexbin(df[x_axis], df[y_axis], gridsize=30, cmap="Blues")
                    fig.colorbar(hb, ax=ax, label="Density")
                    ax.set_title(f"{x_axis} and {y_axis} - Hexbin")

                ax.set_xlabel(x_axis)
                ax.set_ylabel(y_axis)
                st.pyplot(fig)
    else:
        st.info("At least two numerical columns are required for numerical + numerical analysis.")

    # -------------------------------
    # 7. Correlation Analysis
    # -------------------------------
    st.subheader("📌 Correlation Analysis")
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)
    else:
        st.info("Not enough numerical columns for correlation analysis.")

    
else:
    st.info("Please upload a CSV file.")
