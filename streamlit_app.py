import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Display the NCAA logo with a specific width
ncaa_logo_url = "https://www.pngall.com/wp-content/uploads/13/NCAA-Logo-PNG.png"

# Title of the web app with adjusted width
st.title('CBB Data Analysis')
st.image(ncaa_logo_url, caption='NCAA Logo', width=100)

# Load the dataset
def load_data(url):
    return pd.read_csv(url)

url_24 = "https://raw.githubusercontent.com/AviHaritsa/cbb-analysis/main/cbb_data_csv.csv"
url_23 = "https://raw.githubusercontent.com/AviHaritsa/cbb-analysis/main/cbb_data_csv_1.csv"
url_22 = "https://raw.githubusercontent.com/AviHaritsa/cbb-analysis/main/cbb_data_csv_2.csv"
df_24 = load_data(url_24)
df_23 = load_data(url_23)
df_22 = load_data(url_22)

# Standardize column names
df_24.columns = df_24.columns.str.upper()
df_23.columns = df_23.columns.str.upper()
df_22.columns = df_22.columns.str.upper()

# Ensure consistent column names across all DataFrames
df_23.rename(columns={'EFG_O': 'EFG%', 'EFG_D': 'EFGD%'}, inplace=True)
df_22.rename(columns={'EFG_O': 'EFG%', 'EFG_D': 'EFGD%'}, inplace=True)

# Add a 'Year' column to each DataFrame
df_24['YEAR'] = 2024
df_23['YEAR'] = 2023
df_22['YEAR'] = 2022

# Concatenate DataFrames
df_all = pd.concat([df_24, df_23, df_22], ignore_index=True)

# Filter function for DataFrame allowing filtering by conference, then teams if necessary
def filter_dataframe(df, key_prefix):
    conference = st.sidebar.selectbox(f'Select Conference ({key_prefix}):', ['All'] + list(df['CONF'].unique()), key=f'conference_{key_prefix}')
    if conference == 'All':
        filtered_df = df
    else:
        filtered_df = df[df['CONF'] == conference]
    
    team = st.sidebar.selectbox(f'Select Team ({key_prefix}):', ['All'] + list(df['TEAM'].unique()), key=f'team_{key_prefix}')
    if team == 'All':
        return filtered_df
    return filtered_df[filtered_df['TEAM'] == team]

# Apply filter for each DataFrame
filtered_df_24 = filter_dataframe(df_24, '2024')
filtered_df_23 = filter_dataframe(df_23, '2023')
filtered_df_22 = filter_dataframe(df_22, '2022')
filtered_df_all = filter_dataframe(df_all, 'All')

# Filter function for DataFrame allowing filtering by numeric range
def numeric_range_filter(df, variable, key_prefix):
    min_val = df[variable].min()
    max_val = df[variable].max()
    min_range, max_range = st.sidebar.slider(f'Select Range for {variable} ({key_prefix}):', min_val, max_val, (min_val, max_val), key=f'range_{variable}_{key_prefix}')
    return df[(df[variable] >= min_range) & (df[variable] <= max_range)]

# Apply numeric range filter for each DataFrame
filtered_df_24 = numeric_range_filter(filtered_df_24, 'G', '2024')
filtered_df_24 = numeric_range_filter(filtered_df_24, 'W', '2024')
# Repeat the process for other numeric variables and DataFrames

# Correlation Scatter Plot Page
st.sidebar.title('Navigation')
page = st.sidebar.selectbox("Go to", ["Datasets", "Visualizations"])

if page == "Visualizations":
    # Select variables for correlation
    x_variable = st.sidebar.selectbox('Corr X Var:', ['G', 'W', 'ADJOE', 'ADJDE', 'EFG%', 'EFGD%', 'TOR', 'TORD', 'ORB', 'DRB', 'FTR', 'FTRD', '2P_O', '2P_D', '3P_O', '3P_D', 'ADJ_T', 'WAB', 'SEED'])
    y_variable = st.sidebar.selectbox('Corr Y Var:', ['G', 'W', 'ADJOE', 'ADJDE', 'EFG%', 'EFGD%', 'TOR', 'TORD', 'ORB', 'DRB', 'FTR', 'FTRD', '2P_O', '2P_D', '3P_O', '3P_D', 'ADJ_T', 'WAB', 'SEED'])

# Plot correlation scatter plot for the combined DataFrame
    if len(filtered_df_all) > 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.scatterplot(data=filtered_df_all, x=x_variable, y=y_variable, ax=ax)
        ax.set_title(f'Correlation between {x_variable} and {y_variable} for Combined Teams Data (2022-2024)')
        ax.set_xlabel(x_variable)
        ax.set_ylabel(y_variable)
        st.pyplot(fig)

# Histogram
    variable = st.sidebar.selectbox('Select Variable for Histogram:', ['G', 'W', 'ADJOE', 'ADJDE', 'EFG%', 'EFGD%', 'TOR', 'TORD', 'ORB', 'DRB', 'FTR', 'FTRD', '2P_O', '2P_D', '3P_O', '3P_D', 'ADJ_T', 'WAB', 'SEED'])
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(filtered_df_all[variable], bins=30, kde=True, ax=ax)
    ax.set_title(f'Histogram of {variable}')
    ax.set_xlabel(variable)
    st.pyplot(fig)

# Boxplot
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(x='YEAR', y=variable, data=filtered_df_all, ax=ax)
    ax.set_title(f'Boxplot of {variable} by Year')
    ax.set_xlabel('Year')
    ax.set_ylabel(variable)
    st.pyplot(fig)

# Team Comparison
    st.write("## Team Comparison")
    teams_to_compare = st.multiselect('Select Teams to Compare', df_all['TEAM'].unique())

    if teams_to_compare:
        comparison_df = df_all[df_all['TEAM'].isin(teams_to_compare)]
        st.write(comparison_df)
        
        comparison_metric = st.selectbox('Select Metric for Comparison:', ['G', 'W', 'ADJOE', 'ADJDE', 'EFG%', 'EFGD%', 'TOR', 'TORD', 'ORB', 'DRB', 'FTR', 'FTRD', '2P_O', '2P_D', '3P_O', '3P_D', 'ADJ_T', 'WAB', 'SEED'])

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x='TEAM', y=comparison_metric, data=comparison_df, ax=ax)
        ax.set_title(f'Comparison of {comparison_metric} among selected Teams')
        ax.set_xlabel('Team')
        ax.set_ylabel(comparison_metric)
        st.pyplot(fig)
        '''
    else:
        # Display filtered DataFrames in columns
        st.write("## Data Overview")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("### 2024 Teams")
            st.write(filtered_df_24.head())

        with col2:
            st.write("### 2023 Teams")
            st.write(filtered_df_23.head())

        with col3:
            st.write("### 2022 Teams")
            st.write(filtered_df_22.head())

    # Add other visualizations and filters as needed
'''

# Display filtered DataFrames in columns
st.write("## Data Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.write("### 2024 Teams")
    st.write(filtered_df_24.head())

with col2:
    st.write("### 2023 Teams")
    st.write(filtered_df_23.head())

with col3:
    st.write("### 2022 Teams")
    st.write(filtered_df_22.head())

# Download filtered data
    st.sidebar.markdown("## Download Filtered Data")
    csv = filtered_df_all.to_csv(index=False)
    st.sidebar.download_button(label="Download CSV", data=csv, file_name='filtered_data.csv', mime='text/csv')
