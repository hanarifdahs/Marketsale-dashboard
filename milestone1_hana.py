import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from scipy import stats
from statsmodels.stats.weightstats import ztest

st.set_page_config(
    page_title="Marketsale Dashboard",
    page_icon="💰",
    layout='wide'
)

DATA_URL = (
    'https://raw.githubusercontent.com/hanarifdahs/datasets/main/supermarket_clean.csv')


@st.cache  # untuk mauskin ke cache data jadi ga nge get data dr awal di function
def load_data():
    df = pd.read_csv(DATA_URL)
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df['Year'] = pd.DatetimeIndex(df['Date']).year
    df['Month'] = pd.DatetimeIndex(df['Date']).month
    df['Day'] = pd.DatetimeIndex(df['Date']).day
    df['Hour'] = pd.DatetimeIndex(df['Time']).hour
    df.columns = df.columns.str.replace(' ', '_')
    df.drop(['Unnamed:_0'], inplace=True, axis=1)
    return df


df = load_data()

st.sidebar.title('Navigation')
select_page = st.sidebar.radio('Go to', ['Home', 'Hypothesis Testing'])


def gross_PL():
    gross = df.groupby(by='Product_line').mean().gross_income
    fig = px.bar(gross, x=gross.index, y='gross_income')
    st.plotly_chart(fig)


def qtt_PL():
    qtt = df.groupby(by='Product_line').sum().Quantity
    fig = px.bar(qtt, x=qtt.index, y='Quantity')
    st.plotly_chart(fig)


def rating_branch():
    fig = plt.figure(figsize=(10, 6))
    plt.title('Rating by Branch')
    sns.barplot(x=df['Branch'], y=df['Rating'])
    st.pyplot(fig)


def gross_branch():
    gross = df.groupby('Branch').sum().gross_income
    fig = px.bar(gross, x=gross.index, y='gross_income')
    st.plotly_chart(fig)


def custType_branch():
    fig = plt.figure(figsize=(10, 6))
    plt.title('Rating by Branch')
    sns.countplot(x=df['Branch'], hue=df['Customer_type'])
    st.pyplot(fig)


def hour_sales():
    hour = df.groupby('hour').mean().Total
    fig = px.line(hour, x=hour.index, y='Total')
    # fig = px.bar(df, x='Time', y='Total',
    #                    color='City', hover_name='City')
    st.plotly_chart(fig)
    # hour_df = df.groupby('hour')


def overall_sales():
    sales = df_query.groupby(by=['Date']).sum()[['Total']]
    fig_sale_day = px.line(
        sales,
        x=sales.index,
        y='Total'
    )

    st.plotly_chart(fig_sale_day)


st.header('Supermarket Performance Sales')
st.markdown("---")
if select_page == 'Home':
    ###FILTER####
    branch = st.sidebar.multiselect(
        'Select Branch', options=df['City'].unique(), default=df['City'].unique())
    PL = st.sidebar.multiselect('Select Product Line', options=df['Product_line'].unique(
    ), default=df['Product_line'].unique())
    cust_type = st.sidebar.multiselect(
        'Select Customer Type', options=df['Customer_type'].unique(), default=df['Customer_type'].unique())
    gender = st.sidebar.multiselect(
        'Select Gender', options=df['Gender'].unique(), default=df['Gender'].unique())
    payment = st.sidebar.multiselect(
        'Select Payment Type', options=df['Payment'].unique(), default=df['Payment'].unique())

    df_query = df.loc[
        (df['City'].isin(branch)) &
        (df['Gender'].isin(gender)) &
        (df['Customer_type'].isin(cust_type)) &
        (df['Payment'].isin(payment)) &
        (df['Product_line'].isin(PL))
    ]
    # Summary
    with st.expander('Show Raw Data'):
        st.subheader("Raw Data")
        st.write(df)
    total_sales = round(df_query['Total'].sum(), 2)
    avg_total_sales = round(df_query['Total'].mean(), 2)
    grossincome = round(df_query['gross_income'].sum(), 2)
    product_sold = df['Quantity'].sum()
    rating = round(df_query['Rating'].mean(), 1)
    avg_gross_income = round(df_query['gross_income'].mean(), 2)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader('Total Sales')
        st.subheader(f"US ${total_sales:,}")

    with col2:
        st.subheader('Average Total Sales')
        st.subheader(f"US $ {avg_total_sales:,}")

    with col3:
        st.subheader('Gross Income')
        st.subheader(f"US $ {grossincome:,}")

    col4, col5, col6 = st.columns(3)
    with col4:
        st.subheader('Product Sold')
        st.subheader(f"{product_sold} Units")
    with col5:
        st.subheader('Rating')
        st.subheader(f"{rating}")
    with col6:
        st.subheader('Average Gross Income')
        st.subheader(f"US ${avg_gross_income:,}")

    st.markdown("---")
    # Chart
    st.header("Charts")
    st.subheader("Total Sales")
    overall_sales()
    with st.container():
        select_product = st.selectbox(
            "Select", ['Gross Income', 'Unit Sold'])
        if select_product == 'Gross Income':
            st.subheader('Gross income by Product Line')
            gross_PL()
        else:
            st.subheader('Unit Sold by Product Line')
            qtt_PL()

    with st.container():
        select_branch = st.selectbox(
            "Select", ['Rating', 'Gross Income', 'Customer Type'])
        if select_branch == 'Rating':
            st.subheader('Rating of All Branch ')
            rating_branch()
        elif select_branch == 'Gross Income':
            st.subheader('Gross income over 3 Months')
            gross_branch()
        else:
            st.subheader('Member and Non-Member Customer by Branch')
            custType_branch()

    with st.container():
        st.subheader('Which Was the Best Hour for Overall Sales?')
        hour_sales()
        # st.subheader('Sales Trend of Each Branch Within 3 Months')
        # month_sales()
else:
    st.subheader('Hypothesis Testing')
    st.markdown('''Hypothesis Testing uses Z-Test consider we know 
             the standard deviation of the population
             and the sample test is greater than 30. ''')
    st.markdown('''We want to know 
             is the average sales is significantly different between member
             and non-member customer''')
    st.markdown('''shall we make the null hypothesis and alternative hypothesis

**H0: μ_Member = μ_NonMember**

**H1: μ_Member != μ_Non_Member**''')

    member = df[df['Customer_type'] == 'Member']['Total'].sample(90)
    non = df[df['Customer_type'] == 'Normal']['Total'].sample(90)

    st.write('>The Average Sales of Member Customer is',
             round(member.mean(), 2))
    st.write('>The Average Sales of Non-Member Customer is',
             round(non.mean(), 2))
    alpha = 0.05
    z, p = ztest(member, non)

    st.write('The P-Value we get from Z-Test is', round(p, 4),
             'and we compare it with the significance level (α), we set the **α = 0.05**')

    member_df = member.to_frame()
    non_df = non.to_frame()

    member_pop = np.random.normal(
        member_df.Total.mean(), member_df.Total.std(), 10000)
    non_pop = np.random.normal(non_df.Total.mean(), non_df.Total.std(), 10000)

    ci = stats.norm.interval(
        0.90, member_df.Total.mean(), member_df.Total.std())
    fig_ht = plt.figure(figsize=(16, 5))
    sns.distplot(
        member_pop, label='Member Population', color='blue')
    sns.distplot(
        non_pop, label='Non-Member Population', color='red')

    plt.axvline(member_df.Total.mean(), color='blue',
                label='Member mean', linewidth=1)
    plt.axvline(non_df.Total.mean(), color='red',
                label='Non-Member', linewidth=0.75)

    plt.axvline(ci[1], color='green', linestyle='dashed',
                label='Confidence Threshold of 95%')
    plt.axvline(ci[0], color='green', linestyle='dashed')

    plt.axvline(member_pop.mean() + z * member_pop.std(), color='black',
                linestyle='dashed', label='Alternative Hypothesis')
    plt.axvline(member_pop.mean() - z * member_pop.std(),
                color='black', linestyle='dashed')

    plt.legend()
    st.pyplot(fig_ht)

    if(p < alpha):
        st.write('''We conclude that we **Reject Null Hypothesis** 
                 because we have enough evidence to reject the null 
                 hypothesis. Proven by the P-value is less than
                 the significance level or the confidence threshold. 
                 Means that **the mean of sales is significantly
                 different between member and non-member customer**''')
    else:
        st.write('''We conclude that we **Fail to Reject Null Hypothesis**
                 because we don\'t have enough evidence to reject the null 
                 hypothesis. Proven by the P-value is greater than
                 the significance level or the confidence threshold. 
                 Means that **the mean of sales is not significantly
                 different between member and non-member customer**''')
