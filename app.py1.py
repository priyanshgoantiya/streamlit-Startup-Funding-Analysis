import streamlit as st
import pandas as pd
from itertools import chain
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans

st.set_page_config(layout='wide',page_title="Startup Analysis")
df=pd.read_csv('startup_funding (2).csv')
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Investors'] = df['Investors'].astype(str)

# Split the 'Investors' column by comma
split_investors = df['Investors'].str.split(',')

# Flatten the list of lists using chain.from_iterable
flat_investors = chain.from_iterable(split_investors)

# Strip whitespace from each investor name
investors_list = [investor.strip() for investor in flat_investors]

# Get unique investors and sort them
unique_investors = sorted(set(investors_list))


def load_investor_details(Investors):

    st.title(Investors)
    # last 5 investment of the investors
    last5_df=df[df['Investors'].str.contains(Investors)].sort_values('Date').head()[['Date','Startup','Vertical','City','Round'	,'amount']]
    st.subheader('Most recent investment')
    st.dataframe(last5_df)

    col1,col2,col3,col4=st.columns(4)
    # biggest Investments
    with col1:
        biggest_investment = df[df['Investors'].str.contains(Investors)].groupby('Startup')['amount'].sum().sort_values(ascending=False).head(20)
        st.subheader("biggest_investment")
        fig, ax = plt.subplots(figsize=(12,12))
        ax.bar(biggest_investment.index, biggest_investment.values)
        ax.set_xlabel("Startup",fontsize=12)
        ax.set_ylabel("Amount(IN crore)",fontsize=12)
        ax.set_title(f'Biggest Investment by {Investors}')
        plt.xticks(rotation=90, fontsize=12)
        plt.yticks(fontsize=12)
        st.pyplot(fig)
    # Sector Invested In
    with col2:
        investment_by_vertical=df[df['Investors'].str.contains(Investors)].groupby('Vertical')['amount'].sum().head(10)
        fig1, ax1 = plt.subplots()
        st.subheader("Sector Invested In")
        ax1.pie(investment_by_vertical, labels=investment_by_vertical.index, autopct='%1.1f%%', pctdistance=0.75)
        ax1.set_title(f'Investments by {Investors} by Vertical')
        st.pyplot(fig1)
    # Round Invested in
    with col3:
        investment_by_Round=df[df['Investors'].str.contains(Investors)].groupby('Round')['amount'].sum()
        fig2, ax2 = plt.subplots()
        st.subheader("Round Invested in")
        ax2.pie(investment_by_Round, labels=investment_by_Round.index, autopct='%1.1f%%', pctdistance=0.85)
        ax2.set_title(f'Investments by {Investors} by Vertical')
        st.pyplot(fig2)
    # City With Most Investment
    with col4:
        investment_by_city=df[df['Investors'].str.contains(Investors)].groupby('City')['amount'].sum()
        fig3, ax3 = plt.subplots()
        st.subheader("City With Most Investment")
        ax3.pie(investment_by_city, labels=investment_by_city.index, autopct='%1.1f%%', pctdistance=0.85)
        ax3.set_title(f'Investments by {Investors} by Vertical')
        st.pyplot(fig3)
    # Year With Most Investment
    df['Year'] = df['Date'].dt.year
    # Yearly investment analysis
    yearly_investment = df[df['Investors'].str.contains(Investors)].groupby('Year')['amount'].sum()
    fig4, ax4 = plt.subplots()
    st.subheader(f"Yearly Investment Trend {Investors}")
    ax4.plot(yearly_investment.index, yearly_investment.values, marker='o')
    st.pyplot(fig4)



    # similar investors
    categorical_features = ["City", "Round", "Vertical"]
    numerical_features = ["amount"]

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(), categorical_features)
        ]
    )

    # Transform the data
    X = preprocessor.fit_transform(df)

    # KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X)

    # Streamlit app
    st.title('Investor Clustering and Similar Investors')

    # Sidebar for selecting investor
    investor = st.sidebar.selectbox('Select an Investor', df['Investors'].unique())

    # Display selected investor and their cluster
    st.subheader(f"Selected Investor: {investor}")
    investor_data = df[df['Investors'] == investor]
    st.write(investor_data)

    # Find similar investors
    if not investor_data.empty:
        cluster = investor_data['Cluster'].iloc[0]
        similar_investors = df[df['Cluster'] == cluster]['Investors'].tolist()

        # Display similar investors
        st.subheader(f"Investors similar to {investor}:")
        for inv in similar_investors:
            st.write(inv)

        # Count similar investors
        num_similar_investors = len(similar_investors)
        st.write(f"Number of investors similar to {investor}: {num_similar_investors}")
    else:
        st.warning("No data available for the selected investor.")


st.sidebar.title("startup Funding Analysis")
option =st.sidebar.selectbox('Select One',['Overall Analysis','startup Analysis','Investor Analysis'])

if option =='Overall Analysis':
    st.title('Overall Analysis')
elif option=='startup Analysis':
    st.sidebar.selectbox('Select Startup',sorted(df['Startup'].unique().tolist()))
    btn1=st.sidebar.button("Find Startup details")
    st.title('startup Analysis')
else:
    selected_Investors=st.sidebar.selectbox('Select One',unique_investors)
    btn2 = st.sidebar.button("Find Investor details")
    if btn2:
        load_investor_details(selected_Investors)



