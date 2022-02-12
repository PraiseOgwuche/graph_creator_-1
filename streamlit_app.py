import streamlit as st
from io import StringIO
import pandas as pd
from graphs import DataAnalyzer, order
import SessionState
from default_orders import check_if_order_is_known


class GraphParams:
    def __init__(self, width, height, font_size, font, x_title, y_title, title, title_text, num_of_words_per_line,
                 legend_position):
        self.width = width
        self.height = height
        self.font_size = font_size
        self.font = font
        self.x_title = x_title
        self.y_title = y_title
        self.title = title
        self.title_text = title_text
        self.num_of_words_per_line = num_of_words_per_line
        self.legend_position = legend_position


def graph_params(width, height, text_size, add_legend_pos, title, w):
    with st.expander("Graph Parameters"):
        width = st.number_input('Width', min_value=500, max_value=2500, value=width)
        height = st.number_input('Height', min_value=300, max_value=1500, value=height)
        font_size = st.number_input('Font Size', min_value=10, max_value=60, value=text_size)
        font = st.selectbox('Font', ['Hevletica', 'Hevletica Neue', 'Arial'], index=1)
        x_title = st.text_input('X-axis title:')
        y_title = st.text_input('Y-axis title:')
        title_box = st.checkbox('Add title')
        if title_box:
            title_text = st.text_input(label='Title text:', value=title)
        else:
            title_text = None
        if w:
            num_of_words_per_line = st.number_input('Max words per line:', min_value=1, max_value=6, value=2)
        else:
            num_of_words_per_line = None
        if add_legend_pos:
            legend_position_type = st.radio('Select legend positioning:', ['Easy', 'Advanced'])
            if legend_position_type == 'Easy':
                col1, col2 = st.columns(2)
                legend_position = (col1.selectbox('x-axis:', ['left', 'center', 'right'], index=2),
                                   col2.selectbox('y-axis:', ['bottom', 'middle', 'top'], index=2))
            elif legend_position_type == 'Advanced':
                col1, col2 = st.columns(2)
                legend_position = (st.number_input('x-value:', max_value=2., min_value=-2., value=1., step=0.05),
                                   st.number_input('y-value:', max_value=2., min_value=-2., value=1., step=0.05),
                                   col1.selectbox('x-alignment:', ['right', 'center', 'left'], index=2),
                                   col2.selectbox('y-alignment:', ['bottom', 'middle', 'top'], index=2),
                                   st.radio('Legend Orientation:', ['horizontal', 'vertical']))
        else:
            legend_position = None
    return GraphParams(width, height, font_size, font, x_title, y_title, title_box, title_text, num_of_words_per_line,
                       legend_position)


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
                unique_vals = [x for x in list(dataframe[column].unique())[1:] if str(x) != 'nan']
                ord = check_if_order_is_known(unique_vals)
                if ord is None:
                    ord = sorted(unique_vals)
                order = st.text_area('Select the order for the options:',
                                     value=',\n'.join(ord), height=150)
                session_state.options = ',\n'.join(ord)
            else:
                order = st.text_area('Select the order for the options:',
                                     value=session_state.options, height=150)
            gp = graph_params(900, 600, 18, False, dataframe.loc[0, column], True)
        if column:
            st.header('Resulting Graph')
            graph_for_plot = graph_creator.create_bar_graph(column, width=gp.width, height=gp.height,
                                                            font_size=gp.font_size, font=gp.font,
                                                            order=order, one_color=True,
                                                            x_title=gp.x_title, y_title=gp.y_title,
                                                            title=gp.title, title_text=gp.title_text,
                                                            w=gp.num_of_words_per_line - 1)
            graph_for_download = graph_creator.create_bar_graph(column, width=gp.width * 2, height=gp.height * 2,
                                                                font_size=gp.font_size * 2, font='Arial',
                                                                order=order, one_color=True,
                                                                x_title=gp.x_title, y_title=gp.y_title,
                                                                title=gp.title, title_text=gp.title_text,
                                                                w=gp.num_of_words_per_line - 1)
            st.plotly_chart(graph_for_plot)
            scale = 6 if gp.width * 2 > 3000 else 8
            st.download_button('Download Plot', graph_for_download.to_image(scale=scale), 'image.png')
    elif option == 'Group Bar Graph':
        columns = st.sidebar.multiselect('Select columns to create graph for:', tuple(dataframe.columns))
        with st.sidebar:
            save = st.checkbox('Save the order')
            if columns:
                if not save:
                    options = []
                    for col in columns:
                        options.extend(list(dataframe[col].unique())[1:])
                    options = sorted([x for x in list(set(options)) if str(x) != 'nan'])
                    ord = check_if_order_is_known(options)
                    if ord is not None:
                        options = ord
                    order = st.text_area('Select the order for the options:',
                                         value=',\n'.join(options), height=150)
                    session_state.options = ',\n'.join(options)
                else:
                    order = st.text_area('Select the order for the options:',
                                         value=session_state.options, height=150)
            gp = graph_params(1200, 700, 18, True, '', True)
        if columns:
            st.header('Resulting Graph')
            graph_for_plot = graph_creator.create_bar_graph_group(columns, width=gp.width, height=gp.height,
                                                                  font_size=gp.font_size, font=gp.font,
                                                                  order=order, x_title=gp.x_title, y_title=gp.y_title,
                                                                  title=gp.title, title_text=gp.title_text,
                                                                  w=gp.num_of_words_per_line - 1,
                                                                  legend_position=gp.legend_position)
            graph_for_download = graph_creator.create_bar_graph_group(columns, width=gp.width * 2, height=gp.height * 2,
                                                                      font_size=gp.font_size * 2, font='Arial',
                                                                      order=order, x_title=gp.x_title,
                                                                      y_title=gp.y_title,
                                                                      title=gp.title, title_text=gp.title_text,
                                                                      w=gp.num_of_words_per_line - 1,
                                                                      legend_position=gp.legend_position)
            st.plotly_chart(graph_for_plot)
            scale = 6 if gp.width * 2 > 3000 else 8
            st.download_button('Download Plot', graph_for_download.to_image(scale=scale), 'image.png')
    elif option == 'Multiple-Choice Question Bar Graph':
        column = st.sidebar.selectbox('Select column to create graph for:', tuple(dataframe.columns))
        with st.sidebar:
            save = st.checkbox('Save the order')
            if not save:
                unique_vals = list(graph_creator.get_categories_from_columns(column, ',(\S)')['index'])
                ord = check_if_order_is_known(unique_vals)
                if ord is None:
                    ord = sorted(unique_vals)
                order = st.text_area('Select the order for the options:',
                                     value=',\n'.join(ord),
                                     height=250)
                session_state.options = ',\n'.join(ord)
            else:
                order = st.text_area('Select the order for the options:',
                                     value=session_state.options, height=250)
            gp = graph_params(900, 600, 18, False, dataframe.loc[0, column], True)
        if column:
            st.header('Resulting Graph')
            graph_for_plot = graph_creator.create_chart_for_categories(column, width=gp.width, height=gp.height,
                                                                       font_size=gp.font_size, font=gp.font,
                                                                       order=order, one_color=True,
                                                                       x_title=gp.x_title, y_title=gp.y_title,
                                                                       title=gp.title, title_text=gp.title_text,
                                                                       w=gp.num_of_words_per_line - 1)
            graph_for_download = graph_creator.create_chart_for_categories(column, width=gp.width * 2,
                                                                           height=gp.height * 2,
                                                                           font_size=gp.font_size * 2, font='Arial',
                                                                           order=order, one_color=True,
                                                                           x_title=gp.x_title, y_title=gp.y_title,
                                                                           title=gp.title, title_text=gp.title_text,
                                                                           w=gp.num_of_words_per_line - 1)
            st.plotly_chart(graph_for_plot)
            scale = 6 if gp.width * 2 > 3000 else 8
            st.download_button('Download Plot', graph_for_download.to_image(scale=scale), 'image.png')

    elif option == 'Pie Chart':
        column = st.sidebar.selectbox('Select column to create graph for:', tuple(dataframe.columns))
        with st.sidebar:
            save = st.checkbox('Save the order')
            if not save:
                unique_vals = [x for x in list(dataframe[column].unique())[1:] if str(x) != 'nan']
                ord = check_if_order_is_known(unique_vals)
                if ord is None:
                    ord = sorted(unique_vals)
                order = st.text_area('Select the order for the options:',
                                     value=',\n'.join(ord), height=150)
                session_state.options = ',\n'.join(ord)
            else:
                order = st.text_area('Select the order for the options:',
                                     value=session_state.options, height=150)
            what_show = st.selectbox('What to show in pie chart?', ['Percent', 'Percent and Label'], index=0)
            gp = graph_params(900, 600, 25, True, dataframe.loc[0, column], False)
        if column:
            st.header('Resulting Graph')
            graph_for_plot = graph_creator.create_pie_chart(column, width=gp.width, height=gp.height,
                                                            font_size=gp.font_size, font=gp.font,
                                                            x_title=gp.x_title, y_title=gp.y_title,
                                                            title=gp.title, title_text=gp.title_text,
                                                            what_show=what_show, legend_position=gp.legend_position)
            graph_for_download = graph_creator.create_pie_chart(column, width=gp.width * 2, height=gp.height * 2,
                                                                font_size=gp.font_size * 2, font='Arial',
                                                                x_title=gp.x_title, y_title=gp.y_title,
                                                                title=gp.title, title_text=gp.title_text,
                                                                what_show=what_show, legend_position=gp.legend_position)
            st.plotly_chart(graph_for_plot)
            scale = 6 if gp.width * 2 > 3000 else 8
            st.download_button('Download Plot', graph_for_download.to_image(scale=scale), 'image.png')
