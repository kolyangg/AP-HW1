import streamlit as st
import pandas as pd
import plotly.express as px
import json

import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np

# PAGE CONFIG
st.set_page_config(page_title="Plotting Demo", page_icon="📈")


# Prepare data
bank_data = pd.read_csv('bank_data.csv')

features_financial = ['PERSONAL_INCOME', 'CREDIT', 'FST_PAYMENT']
features_categorial = ['AGE', 'WORK_TIME', 'TERM']
features_bank = ['TARGET', 'LOAN_NUM_TOTAL', 'LOAN_NUM_CLOSED']
features_personal = ['GENDER', 'CHILD_TOTAL', 'DEPENDANTS', 'SOCSTATUS_WORK_FL', 'SOCSTATUS_PENS_FL', 'OWN_AUTO']
features_columns = [*features_financial, *features_categorial, *features_bank, *features_personal]

features_boolean = ['GENDER', 'SOCSTATUS_WORK_FL', 'SOCSTATUS_PENS_FL', 'OWN_AUTO']
features_categorial = ['TERM', 'CHILD_TOTAL', 'DEPENDANTS', 'LOAN_NUM_TOTAL',	'LOAN_NUM_CLOSED']

features = bank_data[features_columns]

# Нормализация данных по 99% персентилю
features_adj = features[features['CREDIT'] <= features['CREDIT'].quantile(0.99)]
features_adj = features_adj[features_adj['PERSONAL_INCOME'] <= features_adj['PERSONAL_INCOME'].quantile(0.99)]
features_adj = features_adj[features_adj['FST_PAYMENT'] <= features_adj['FST_PAYMENT'].quantile(0.99)]
features_adj = features_adj[features_adj['WORK_TIME'] <= features_adj['WORK_TIME'].quantile(0.99)]




st.set_option('deprecation.showPyplotGlobalUse', False)

def home_page():
    st.title("Распределение признаков")
    st.write("Графики распределений")

    # CHART1
    st.subheader('Размер кредита')
    fig1, ax1 = plt.subplots()
    sns.histplot(features_adj['CREDIT'], ax = ax1)
    st.pyplot(fig1)

    # CHART2
    st.subheader('Размер дохода')
    fig2, ax2 = plt.subplots()
    sns.histplot(features_adj['PERSONAL_INCOME'], ax = ax2, bins = 20)
    st.pyplot(fig2)
    
    # CHART3
    st.subheader('Первый платёж')
    fig3, ax3 = plt.subplots()
    sns.histplot(features_adj['FST_PAYMENT'], ax = ax3, bins = 20)
    st.pyplot(fig3)


def page1():
    st.title("Совместное распределение признаков")
    columns_small = ['TARGET', 'PERSONAL_INCOME', 'CREDIT', 'AGE', 'LOAN_NUM_CLOSED', 'CHILD_TOTAL', 'SOCSTATUS_WORK_FL', 'SOCSTATUS_PENS_FL']
    features_adj_small = features_adj[columns_small]

    # CHART1
    st.subheader("Совместное распределение")
    fig1 = sns.pairplot(data = features_adj_small, vars = ['PERSONAL_INCOME', 'CREDIT', 'AGE', 'SOCSTATUS_WORK_FL', 'SOCSTATUS_PENS_FL'], hue = 'TARGET')
    st.pyplot(fig1)

def page2():
    st.title("Корреляция признаков")
    list_for_corr = ['TARGET', 'PERSONAL_INCOME', 'CREDIT', 'AGE', 'LOAN_NUM_CLOSED', 'CHILD_TOTAL', 'SOCSTATUS_WORK_FL', 'SOCSTATUS_PENS_FL']
    features_corr = features[list_for_corr].corr()

    # CHART1
    st.subheader("Матрица корреляций признаков")
    sns.heatmap(features_corr, cmap="Blues", annot=True, annot_kws={"size": 10})
    st.pyplot()

def page3():
    st.title("Зависимости целевой переменной и признаков")
    columns_small = ['TARGET', 'PERSONAL_INCOME', 'CREDIT', 'AGE', 'LOAN_NUM_CLOSED', 'CHILD_TOTAL', 'SOCSTATUS_WORK_FL', 'SOCSTATUS_PENS_FL']
    features_adj_small = features_adj[columns_small]

    # CHART1
    st.subheader('Target и размер дохода')
    fig1, ax1 = plt.subplots()
    sns.boxplot(x=features_adj['TARGET'], y=features_adj['PERSONAL_INCOME'])
    plt.title('Target vs PERSONAL_INCOME')
    plt.xlabel('Target')
    plt.ylabel('PERSONAL_INCOME')
    st.pyplot(fig1)
    
    # CHART2
    st.subheader('Target и возраст')
    fig2, ax2 = plt.subplots()
    sns.boxplot(x=features_adj['TARGET'], y=features_adj['AGE'])
    plt.title('Target vs AGE')
    plt.xlabel('Target')
    plt.ylabel('AGE')
    st.pyplot(fig2)


    # CHART3
    st.subheader('Target и пол')
    df = features_adj
    grouped = df.groupby('GENDER')['TARGET'].value_counts(normalize=True).unstack().fillna(0) * 100
    grouped.plot(kind='bar', stacked=True, color=['burlywood', 'beige'])
    plt.title('Target % by gender')
    plt.xlabel('Gender')
    plt.ylabel('Target %')
    plt.xticks(ticks = list(range(len(grouped.index))), labels=['Female', 'Male'], rotation=0)
    plt.legend(title='Target', labels=['Zero (0)', 'One (1)'])
    plt.show()
    st.pyplot()

    # CHART4
    st.subheader('Target и SOCSTATUS_WORK')
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    # Pie chart for A = 1
    grouped.loc[1].plot(kind='pie', ax=axes[0], autopct='%1.0f%%', startangle=90, colors=['burlywood', 'beige'])
    axes[0].set_ylabel('')
    axes[0].set_title('SOCSTATUS_WORK_FL = 1')

    # Pie chart for A = 0
    grouped.loc[0].plot(kind='pie', ax=axes[1], autopct='%1.0f%%', startangle=90, colors=['burlywood', 'beige'])
    axes[1].set_ylabel('')
    axes[1].set_title('SOCSTATUS_WORK_FL = 0')

    axes[0].legend(title='Target value', loc='upper left', bbox_to_anchor=(-0.1, 1))
    plt.show()
    st.pyplot()


    # CHART5
    st.subheader('Target и SOCSTATUS_PENS_FL')
    df = features_adj
    grouped = df.groupby('SOCSTATUS_PENS_FL')['TARGET'].value_counts(normalize=True).unstack().fillna(0) * 100
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    # Pie chart for A = 1
    grouped.loc[1].plot(kind='pie', ax=axes[0], autopct='%1.0f%%', startangle=90, colors=['burlywood', 'beige'])
    axes[0].set_ylabel('')
    axes[0].set_title('SOCSTATUS_PENS_FL = 1')

    # Pie chart for A = 0
    grouped.loc[0].plot(kind='pie', ax=axes[1], autopct='%1.0f%%', startangle=90, colors=['burlywood', 'beige'])
    axes[1].set_ylabel('')
    axes[1].set_title('SOCSTATUS_PENS_FL = 0')

    axes[0].legend(title='Target value', loc='upper left', bbox_to_anchor=(-0.1, 1))
    plt.show()
    st.pyplot()


    # CHART6
    st.subheader('Target и количество детей')
    df = features_adj
    grouped = df.groupby('CHILD_TOTAL')['TARGET'].value_counts(normalize=True).unstack().fillna(0) * 100
    grouped.plot(kind='bar', stacked=True, color=['skyblue', 'salmon'])

    plt.title('Target % by number of children')
    plt.xlabel('Number of children')
    plt.ylabel('Target %')
    plt.xticks(ticks = list(range(len(grouped.index))), labels=list(grouped.index), rotation=0)
    plt.legend(title='Target', labels=['Zero (0)', 'One (1)'])
    plt.show()
    st.pyplot()


    # CHART7
    st.subheader('Target и количество автомобилей')
    df = features_adj
    grouped = df.groupby('OWN_AUTO')['TARGET'].value_counts(normalize=True).unstack().fillna(0) * 100
    grouped.plot(kind='bar', stacked=True, color=['skyblue', 'salmon'])

    plt.title('Target % by number of cars')
    plt.xlabel('Number of cars')
    plt.ylabel('Target %')
    plt.xticks(ticks = list(range(len(grouped.index))), labels=list(grouped.index), rotation=0)
    plt.legend(title='Target', labels=['Zero (0)', 'One (1)'])
    plt.show()
    st.pyplot()



def page4():
    st.title("Основные статистики числовых столбцов")

    # TABLE1
    num_columns = ['PERSONAL_INCOME', 'CREDIT', 'FST_PAYMENT', 'AGE', 'WORK_TIME', 'TERM']
    df = features_adj[num_columns]
    stats = df.describe().T.round(1)  # Transpose for easier plotting
    st.subheader("Основные статистики")
    st.dataframe(stats)



# Main App
def main():
    #st.sidebar.title("Navigation")
    p0 = "Распределение признаков"
    p1 = "Совместное распределение признаков"
    p2 = "Корреляция признаков"
    p3 = "Зависимости целевой переменной и признаков"
    p4 = "Характеристики распределения числовых столбцов"
    choice = st.selectbox("Выбрать страницу", [p0, p1, p2, p3, p4])

    if choice == p0:
        home_page()
    elif choice == p1:
        page1()
    elif choice == p2:
        page2()
    elif choice == p3:
        page3()
    elif choice == p4:
        page4()

if __name__ == "__main__":
    main()


























