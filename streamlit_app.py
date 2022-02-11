import streamlit as st
from io import StringIO
import pandas as pd
from graphs import DataAnalyzer, order
import SessionState


class GraphParams:
    def __init__(self, width, height, font_size, font, x_title, y_title, title, num_of_words_per_line):
        self.width = width
        self.height = height
        self.font_size = font_size
        self.font = font
        self.x_title = x_title
        self.y_title = y_title
        self.title = title
        self.num_of_words_per_line = num_of_words_per_line


def graph_params(width, height):
    with st.expander("Graph Parameters"):
        width = st.number_input('Width', min_value=500, max_value=1500, value=width)
        height = st.number_input('Height', min_value=300, max_value=1500, value=height)
        font_size = st.number_input('Font Size', min_value=10, max_value=60, value=18)
        font = st.selectbox('Font', ['Hevletica', 'Hevletica Neue', 'Arial'], index=1)
        x_title = st.text_input('X-axis title:')
        y_title = st.text_input('Y-axis title:')
        title = st.checkbox('Add title')
        num_of_words_per_line = st.number_input('Max words per line:', min_value=1, max_value=6, value=2)
    return GraphParams(width, height, font_size, font, x_title, y_title, title, num_of_words_per_line)


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
    session_state = SessionState.get(name='', options='')
    graph_creator = DataAnalyzer(dataframe)
    if option == 'Bar Graph':
        column = st.sidebar.selectbox('Select column to create graph for:', tuple(dataframe.columns))
        with st.sidebar:
            save = st.checkbox('Save the order')
            if not save:
                order = st.text_input('Select the order for the options:',
                                      value=','.join(sorted(list(dataframe[column].unique())[1:])))
                session_state.options = ','.join(sorted(list(dataframe[column].unique())[1:]))
            else:
                order = st.text_input('Select the order for the options:',
                                      value=session_state.options)
            gp = graph_params(900, 600)
        submitted = st.sidebar.button('Plot Graph')
        if submitted:
            st.header('Resulting Graph')
            graph_for_plot = graph_creator.create_bar_graph(column, width=gp.width, height=gp.height,
                                                            font_size=gp.font_size, font=gp.font,
                                                            order=order, one_color=True,
                                                            x_title=gp.x_title, y_title=gp.y_title,
                                                            title=gp.title, w=gp.num_of_words_per_line - 1)
            graph_for_download = graph_creator.create_bar_graph(column, width=gp.width * 2, height=gp.height * 2,
                                                                font_size=gp.font_size * 2, font=gp.font,
                                                                order=order, one_color=True,
                                                                x_title=gp.x_title, y_title=gp.y_title,
                                                                title=gp.title, w=gp.num_of_words_per_line - 1)
            st.plotly_chart(graph_for_plot)
            st.download_button('Download Plot', graph_for_download.to_image(scale=8), 'image.png')
    elif option == 'Group Bar Graph':
        columns = st.sidebar.multiselect('Select columns to create graph for:', tuple(dataframe.columns))
        with st.sidebar:
            save = st.checkbox('Save the order')
            if columns:
                if not save:
                    options = []
                    for col in columns:
                        options.extend(list(dataframe[columns[0]].unique())[1:])
                    options = sorted(list(set(options)))
                    order = st.text_input('Select the order for the options:',
                                          value=','.join(options))
                    session_state.options = ','.join(options)
                else:
                    order = st.text_input('Select the order for the options:',
                                          value=session_state.options)
            gp = graph_params(1200, 700)
        submitted = st.sidebar.button('Plot Graph')
        if submitted:
            st.header('Resulting Graph')
            graph_for_plot = graph_creator.create_bar_graph_group(columns, width=gp.width, height=gp.height,
                                                                  font_size=gp.font_size, font=gp.font,
                                                                  order=order, x_title=gp.x_title, y_title=gp.y_title,
                                                                  title=gp.title, w=gp.num_of_words_per_line - 1)
            graph_for_download = graph_creator.create_bar_graph_group(columns, width=gp.width * 2, height=gp.height * 2,
                                                                      font_size=gp.font_size * 2, font=gp.font,
                                                                      order=order, x_title=gp.x_title,
                                                                      y_title=gp.y_title,
                                                                      title=gp.title, w=gp.num_of_words_per_line - 1)
            st.plotly_chart(graph_for_plot)
            st.download_button('Download Plot', graph_for_download.to_image(scale=8), 'image.png')
