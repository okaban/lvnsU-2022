# -*- coding: utf-8 -*-

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px

def set_iris_data():
    df = px.data.iris()
    return df

def show_data_selection_bar(df):
    st.sidebar.title('Selector')
    species = df['species'].unique()
    selected_species = st.sidebar.multiselect(
        'Species', options=species, default=species)
    min_value = df['sepal_length'].min().item()
    max_value = df['sepal_length'].max().item()
    sepal_len_min, sepal_len_max = st.sidebar.slider(
        'Sepal Length', 
        min_value=min_value, max_value=max_value,
        value=(min_value, max_value)
    )
    options = {}
    options['selected_species'] = selected_species
    options['sepal_len_min'] = sepal_len_min
    options['sepal_len_max'] = sepal_len_max
    return options
   
    
def show_dataframe(df):
    st.dataframe(df)


def show_scatterplot(df):
    # extract column names
    axis_list = df.columns.unique()
    # select X axis name
    selected_xaxis = st.selectbox(
        'X-axis', axis_list, 
    )
    # select Y axis name
    selected_yaxis = st.selectbox(
        'Y-axis', axis_list
    )
    # 
    fig = px.scatter(df, x=selected_xaxis, y=selected_yaxis, color="species")
    st.plotly_chart(fig, use_container_width=True)


def main():
    st.title('Iris Dataset Dashboard')
    df = set_iris_data()
    
    options = show_data_selection_bar(df)
    df_tmp = df[df['species'].isin(options['selected_species'])]
    df_selected = df_tmp[(df_tmp['sepal_length'] >= options['sepal_len_min']) & 
                         (df_tmp['sepal_length'] <= options['sepal_len_max'])]
    
    st.write("Before selection: %d rows, %d columns" % (df.shape[0], df.shape[1]))
    st.write("After selection: %d rows, %d columns" % (df_selected.shape[0], df_selected.shape[1]))
    st.subheader('Scatter Plot:')
    show_scatterplot(df_selected)
    st.subheader('Selected Data:')
    show_dataframe(df_selected)


if __name__ == '__main__':
    main()

