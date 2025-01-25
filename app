import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import sqlite3
import re
import google.generativeai as genai
from streamlit_option_menu import option_menu

# Configure Google Generative AI
genai.configure(api_key="AIzaSyCWf0aLUCsFx2ec8sW6sWh5MhhzkfOlf1w")
model = genai.GenerativeModel('gemini-1.5-flash')

# Split user prompt into queries
def split_user_prompt(prompt):
    """Splits the user input into separate queries based on conjunctions like 'and'."""
    queries = re.split(r'\band\b', prompt)
    return [query.strip() for query in queries if query.strip()]

# Create SQLite database from DataFrame
def create_sqlite_db(df):
    conn = sqlite3.connect(':memory:')  # Use an in-memory database
    df.to_sql('data', conn, index=False, if_exists='replace')
    return conn

# Execute SQL query
def execute_sql_query(conn, sql_query):
    try:
        return pd.read_sql_query(sql_query, conn)
    except Exception as e:
        st.error(f"Error executing SQL query: {e}")
        return None

# Load dataset, handle user prompt, and consolidate data
def load_and_process_data():
    uploaded_file = st.file_uploader("Upload your data file", type=["csv", "xlsx", "xls"])
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.write("Uploaded Data:")
        st.dataframe(df)

        # Convert DataFrame to SQLite database
        conn = create_sqlite_db(df)

        st.subheader("AI Chart Generator")
        user_prompt = st.text_input("Turn Your Data into Actionable Insights Instantly")

        if user_prompt:
            # Split the user input into separate queries based on conjunctions like "and"
            queries = split_user_prompt(user_prompt)

            # Process each split query
            consolidated_df = None
            columns_to_keep = set()
            for i, user_query in enumerate(queries):
                # Prepare the text for the model
                prompt = f"""
                Convert the following natural language prompt into a valid SQL query:

                User's analysis request: "{user_query}"

                The SQL query should be based on the following table schema:
                {', '.join([str(col) for col in df.columns])}

                Ensure the query references the table as "data" since that's the table name in the SQLite database.
                Provide only the SQL query, without any additional text or explanation, give the sql query to dispaly only the measures are dimension that are in user prompt .
                """

                # Send the prompt to the model to generate the SQL query
                try:
                    response = model.generate_content(prompt)
                    sql_query = response.text.strip()
                    sql_query = sql_query.replace("```", "").replace("sql", "").strip()
                    sql_query = re.sub(r'\s+', ' ', sql_query)  # Ensure no extra spaces are present
                except Exception as e:
                    st.error(f"Error while processing the SQL query: {e}")
                    continue  # Skip further processing for this query

                # Execute the SQL query
                result_df = execute_sql_query(conn, sql_query)

                if result_df is not None and not result_df.empty:
                    # Keep track of columns used in the query
                    columns_to_keep.update(result_df.columns)
                    if consolidated_df is None:
                        consolidated_df = result_df
                    else:
                        consolidated_df = pd.merge(consolidated_df, result_df, how='outer')

            if consolidated_df is not None:
                # Filter the consolidated_df to keep only the columns mentioned in the user queries
                consolidated_df = consolidated_df[list(columns_to_keep)]
                st.write("### Consolidated Table Based on User Query")
                st.dataframe(consolidated_df)
                return consolidated_df, list(columns_to_keep)
            else:
                st.warning("No data available for the given query.")
    return None, None

# Clean and consolidate data
def clean_data(df, columns_to_keep):
    cleaned_df = df[columns_to_keep].copy()
    # Drop columns with all missing values
    cleaned_df = cleaned_df.dropna(axis=1, how='all')
    # Consolidate columns with Yes/No, Male/Female, or similar values
    for column in cleaned_df.columns:
        if cleaned_df[column].dtype == object:  # Identify categorical columns
            unique_values = cleaned_df[column].dropna().unique()
            if set(unique_values).issubset({'Yes', 'No', 'Male', 'Female'}):
                cleaned_df[column] = cleaned_df[column].apply(lambda x: 1 if x in ['Yes', 'Male'] else (0 if x in ['No', 'Female'] else None))
    # Remove duplicate rows
    cleaned_df = cleaned_df.drop_duplicates()
    # Consolidate identical rows by summing numeric values
    cleaned_df = cleaned_df.groupby(list(cleaned_df.columns)).sum().reset_index()
    st.write("### Cleaned and Consolidated Table")
    st.dataframe(cleaned_df)
    return cleaned_df

# Visualize data automatically
def visualize_data(df):
    st.write("## Data Visualizations")
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = df.select_dtypes(include=['object']).columns

    # Chart Customization - User selects which chart to view
    with st.sidebar:
        selected_chart = option_menu(
            "Select Chart to View",
            [
                "Bar Chart", "Line Chart", "Pie Chart", "Scatter Plot", "Box Plot", "Correlation Heatmap", "Histogram", "Bubble Chart", "Sunburst Chart", "Treemap", "Funnel Chart", "Density Heatmap", "Violin Plot", "Strip Plot", "Facet Grid"
            ],
            icons=[
                "bar-chart", "line-chart", "pie-chart", "scatter-chart", "box", "heat-map", "histogram", "chart-bubble", "sun", "tree", "funnel", "density", "violin", "dots-vertical", "grid"
            ],
            menu_icon="cast",
            default_index=0,
            orientation="horizontal",
            styles={
                "container": {"text-align": "right"},
                "nav-link": {"font-size": "14px", "padding": "10px", "margin": "5px"}
            }
        )
    
    generate_chart(selected_chart, df, numeric_columns, categorical_columns)

# Generate the selected chart
def generate_chart(selected_chart, df, numeric_columns, categorical_columns):
    if selected_chart == "Bar Chart":
        if len(categorical_columns) >= 1 and len(numeric_columns) >= 1:
            for cat_col in categorical_columns:
                st.write(f"### Bar Chart for {cat_col} with Multiple Measures")
                fig = px.bar(df, x=cat_col, y=numeric_columns, barmode='group', color_discrete_sequence=px.colors.qualitative.Set1)
                st.plotly_chart(fig)
        else:
            st.warning("Bar Chart requires at least 1 categorical column and 1 numeric column.")

    elif selected_chart == "Line Chart":
        if len(categorical_columns) >= 1 and len(numeric_columns) >= 1:
            for cat_col in categorical_columns:
                for num_col in numeric_columns:
                    st.write(f"### Line Chart of {num_col} over {cat_col}")
                    fig = px.line(df, x=cat_col, y=num_col, markers=True)
                    st.plotly_chart(fig)
        else:
            st.warning("Line Chart requires at least 1 categorical column and 1 numeric column.")

    elif selected_chart == "Pie Chart":
       if len(categorical_columns) >= 1:
        cat_col = categorical_columns[0]  # Select the first categorical column
        count_df = df[cat_col].value_counts().reset_index()  # Count the occurrences of each value
        count_df.columns = [cat_col, 'count']  # Rename columns for clarity
        
        st.write(f"### Pie Chart of {cat_col}")
        fig = px.pie(count_df, names=cat_col, values='count', title=f"Pie Chart of {cat_col}", labels={cat_col: cat_col})
        st.plotly_chart(fig)
       else:
        st.warning("Pie Chart requires at least 1 categorical column.")

    elif selected_chart == "Scatter Plot":
        if len(numeric_columns) >= 2:
            for i, x_col in enumerate(numeric_columns):
                for y_col in numeric_columns[i + 1:]:
                    st.write(f"### Scatter Plot of {y_col} vs {x_col}")
                    fig = px.scatter(df, x=x_col, y=y_col)
                    st.plotly_chart(fig)
        else:
            st.warning("Scatter Plot requires at least 2 numeric columns.")

    elif selected_chart == "Box Plot":
        if len(categorical_columns) >= 1 and len(numeric_columns) >= 1:
            for cat_col in categorical_columns:
                for num_col in numeric_columns:
                    st.write(f"### Box Plot of {num_col} grouped by {cat_col}")
                    fig = px.box(df, x=cat_col, y=num_col)
                    st.plotly_chart(fig)
        else:
            st.warning("Box Plot requires at least 1 categorical column and 1 numeric column.")

    elif selected_chart == "Correlation Heatmap" and len(numeric_columns) > 1:
        st.write("### Correlation Heatmap")
        plt.figure(figsize=(12, 8))
        sns.heatmap(df[numeric_columns].corr(), annot=True, cmap="viridis")
        plt.title("Correlation Heatmap")
        st.pyplot(plt)
    elif selected_chart == "Correlation Heatmap":
        st.warning("Correlation Heatmap requires at least 2 numeric columns.")

    elif selected_chart == "Histogram":
        if len(numeric_columns) >= 1:
            for num_col in numeric_columns:
                st.write(f"### Histogram of {num_col}")
                fig = px.histogram(df, x=num_col)
                st.plotly_chart(fig)
        else:
            st.warning("Histogram requires at least 1 numeric column.")

    elif selected_chart == "Bubble Chart":
        if len(numeric_columns) >= 3:
            st.write("### Bubble Chart")
            fig = px.scatter(df, x=numeric_columns[0], y=numeric_columns[1], size=numeric_columns[2], color=numeric_columns[0])
            st.plotly_chart(fig)
        else:
            st.warning("Bubble Chart requires at least 3 numeric columns.")

    elif selected_chart == "Sunburst Chart":
        if len(categorical_columns) >= 2:
            st.write("### Sunburst Chart")
            fig = px.sunburst(df, path=categorical_columns[:2], values=numeric_columns[0] if len(numeric_columns) > 0 else None)
            st.plotly_chart(fig)
        else:
            st.warning("Sunburst Chart requires at least 2 categorical columns.")

    elif selected_chart == "Treemap":
        if len(categorical_columns) >= 2:
            st.write("### Treemap")
            fig = px.treemap(df, path=categorical_columns[:2], values=numeric_columns[0] if len(numeric_columns) > 0 else None)
            st.plotly_chart(fig)
        else:
            st.warning("Treemap requires at least 2 categorical columns.")

    elif selected_chart == "Funnel Chart":
        if len(categorical_columns) >= 1 and len(numeric_columns) > 0:
            st.write("### Funnel Chart")
            fig = px.funnel(df, x=categorical_columns[0], y=numeric_columns[0])
            st.plotly_chart(fig)
        else:
            st.warning("Funnel Chart requires at least 1 categorical column and 1 numeric column.")

    elif selected_chart == "Density Heatmap":
        if len(numeric_columns) >= 2:
            st.write("### Density Heatmap")
            fig = px.density_heatmap(df, x=numeric_columns[0], y=numeric_columns[1])
            st.plotly_chart(fig)
        else:
            st.warning("Density Heatmap requires at least 2 numeric columns.")

    elif selected_chart == "Violin Plot":
        if len(categorical_columns) >= 1 and len(numeric_columns) >= 1:
            for cat_col in categorical_columns:
                for num_col in numeric_columns:
                    st.write(f"### Violin Plot of {num_col} by {cat_col}")
                    fig = px.violin(df, x=cat_col, y=num_col)
                    st.plotly_chart(fig)
        else:
            st.warning("Violin Plot requires at least 1 categorical column and 1 numeric column.")

    elif selected_chart == "Strip Plot":
        if len(categorical_columns) >= 1 and len(numeric_columns) >= 1:
            for cat_col in categorical_columns:
                for num_col in numeric_columns:
                    st.write(f"### Strip Plot of {num_col} by {cat_col}")
                    fig = px.strip(df, x=cat_col, y=num_col)
                    st.plotly_chart(fig)
        else:
            st.warning("Strip Plot requires at least 1 categorical column and 1 numeric column.")

    elif selected_chart == "Facet Grid":
        if len(categorical_columns) >= 1 and len(numeric_columns) >= 1:
            st.write("### Facet Grid")
            fig = px.scatter(df, x=numeric_columns[0], y=numeric_columns[0], facet_col=categorical_columns[0])
            st.plotly_chart(fig)
        else:
            st.warning("Facet Grid requires at least 1 categorical column and 1 numeric column.")

# Main function
def main():
    st.title("Comprehensive Data Visualization App")
    df, columns_to_keep = load_and_process_data()
    if df is not None and columns_to_keep is not None:
        cleaned_df = clean_data(df, columns_to_keep)
        visualize_data(cleaned_df)

if __name__ == "__main__":
    main()
