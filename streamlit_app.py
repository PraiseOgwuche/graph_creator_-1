import streamlit as st
from io import StringIO
import pandas as pd
from graphs import DataAnalyzer

st.title("Graph Creator")
st.write("This tool will help you to create various graphs 📉")
st.sidebar.header("Graph Parameters")
uploaded_file = st.sidebar.file_uploader("Upload dataframe", type='csv')

if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))

    # To read file as string:
    string_data = stringio.read()

    # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_csv(uploaded_file)
    st.header("Inputed Dataframe:")
    st.dataframe(dataframe)

    option = st.sidebar.selectbox(
        'Choose graph type to plot',
        ('Bar Graph', 'Group Bar Graph',
         'Multiple-Choice Question Bar Graph',
         'Self-Assessment Graph (requires specific data format)',
         'Pie Chart', 'Gauge Graph', 'Horizontal Bar Graph',
         'Bar Graph with Errors', 'Stacked Bar Graph',
         'Simple Bar Graph', 'Line Graph', 'Scatter Graph with Regression Line'))

    if option == 'Bar Graph':
        column = st.sidebar.selectbox('Select column to create graph for:', tuple(dataframe.columns))
        order = st.sidebar.multiselect('Select the order for the options:', tuple(dataframe[column].unique()))
        if st.sidebar.button('Plot Graph'):
            graph_creator = DataAnalyzer(dataframe)
            st.header('Resulting Graph')
            st.plotly_chart(graph_creator.create_bar_graph(column, order=order, one_color=True))


