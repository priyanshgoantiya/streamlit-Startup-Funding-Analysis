import streamlit as st
import pandas as pd
from itertools import chain
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
import seaborn as sns
import numpy as np

st.set_page_config(layout='wide', page_title="Startup Analysis")

# Load data
df = pd.read_csv('startup_funding (2).csv')
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Investors'] = df['Investors'].astype(str)
df.dropna(subset=['Round'], inplace=True)
df.drop_duplicates(['Startup', 'Date'], inplace=True)

# Function for overall analysis
def load_Overall_analysis():
    st.title("Overall Analysis")

    # Total Invested Amount
    total = round(df['amount'].sum())
    # max Funding Startup
    max_Funding_Startup = df.groupby("Startup")['amount'].max().sort_values(ascending=False).head(1).values[0]
    # Avg Funding Amount
    Avg_funding_amount = df.groupby("Startup")['amount'].sum().mean()
    # TOTAL FUNDED STARTUP
    TOTAL_FUNDED_STARTUP = df['Startup'].nunique()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Invested Amount", f"{total:.2f} CR")

    with col2:
        st.metric("Max Funding Startup", f"{max_Funding_Startup:.2f} CR")

    with col3:
        st.metric("AVG Funding Amount", f"{Avg_funding_amount:.2f} CR")

    with col4:
        st.metric("TOTAL FUNDED STARTUP", TOTAL_FUNDED_STARTUP)

    st.subheader("MOM Graph")
    selected_option = st.selectbox('Select Type', ['Total', 'Count'])
    if selected_option == 'Total':
        temp_df = df.groupby(['Year', 'Month'])['amount'].sum().reset_index()
    else:
        temp_df = df.groupby(['Year', 'Month'])['amount'].count().reset_index()

    st.write("DataFrame after groupby operation:")
    st.write(temp_df)

    # Plotting MOM graph
    temp_df.columns = ['Year', 'Month', 'amount']
    temp_df['x_axis'] = temp_df['Month'].astype(str) + '-' + temp_df['Year'].astype(str)
    fig, ax = plt.subplots()
    ax.plot(temp_df['x_axis'], temp_df['amount'], marker='o')
    ax.set_xlabel("Month and Respective Year", fontsize=12)
    ax.set_ylabel("Amount (IN crore)", fontsize=12)
    ax.set_title('MOM graph investment amount')
    step = max(1, len(temp_df) // 20)
    ax.set_xticks(range(0, len(temp_df), step))
    ax.set_xticklabels(temp_df['x_axis'][::step], rotation=90, fontsize=12)
    plt.yticks(fontsize=12)

    st.pyplot(fig)
    plt.close()

    # Sector Graph
    st.subheader("Sector Wise Investment")
    selected_option1 = st.selectbox('Select Type', ['total', 'count'])
    if selected_option1 == 'total':
        temp_df1 = df.groupby('Vertical')['amount'].sum().reset_index().sort_values(by='amount', ascending=False).head(20)
    else:
        temp_df1 = df.groupby('Vertical')['amount'].count().reset_index().sort_values(by='amount', ascending=False).head(20)

    st.write("DataFrame after groupby operation:")
    st.write(temp_df1)

    fig1, ax1 = plt.subplots(figsize=(12, 18))
    bars = ax1.bar(temp_df1['Vertical'], temp_df1['amount'], color='skyblue')
    ax1.set_xlabel('Vertical')
    ax1.set_ylabel('Amount')
    plt.xticks(rotation=45, ha='right')
    ax1.set_title('Investment Distribution by Sector')
    total_amount = temp_df1['amount'].sum()

    for bar in bars:
        height = bar.get_height()
        percentage = (height / total_amount) * 100
        ax1.annotate(f'{percentage:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height), ha='center')

    st.pyplot(fig1)
    plt.close()

    # Other sections and plots follow similarly...

# Sidebar and main options
option = st.sidebar.selectbox('Select One', ['Overall Analysis', 'Startup Analysis', 'Investor Analysis'])

if option == 'Overall Analysis':
    load_Overall_analysis()

elif option == 'Startup Analysis':
    selected_startup = st.sidebar.selectbox('Select Startup', sorted(df['Startup'].unique().tolist()))
    btn1 = st.sidebar.button("Find Startup details")
    if btn1:
        # Call function for startup analysis
        Startup_analysis(selected_startup)

else:
    selected_investor = st.sidebar.selectbox('Select Investor', sorted(unique_investors))
    btn2 = st.sidebar.button("Find Investor details")
    if btn2:
        # Call function for investor details
        load_investor_details(selected_investor)
