import streamlit as st
from io import StringIO
import pandas as pd
from graphs import DataAnalyzer, order
import SessionState
from default_orders import check_if_order_is_known


class GraphParams:
    def __init__(self, width, height, font_size, font, x_title, y_title, title, title_text, max_symbols,
                 legend_position, transparent, inside_outside):
        self.width = width
        self.height = height
        self.font_size = font_size
        self.font = font
        self.x_title = x_title
        self.y_title = y_title
        self.title = title
        self.title_text = title_text
        self.max_symbols = max_symbols
        self.legend_position = legend_position
        self.transparent = transparent
        self.inside_outside = inside_outside


def graph_params(width, height, text_size, add_legend_pos, title, w, def_text=None,
                 show_inside_outside=False):
    with st.expander("Graph Parameters"):
        if def_text is not None:
            st.write(def_text)
        width = st.number_input('Width', min_value=500, max_value=5000, value=width)
        height = st.number_input('Height', min_value=300, max_value=5000, value=height)
        font_size = st.number_input('Font Size', min_value=10, max_value=60, value=text_size)
        font = st.selectbox('Font', ['Helvetica', 'Helvetica Neue', 'Arial'], index=1)
        if show_inside_outside:
            inside_outside = st.selectbox('Text position relative to the bars', ['inside', 'outside'], index=1)
        else:
            inside_outside = None
        transparent = st.checkbox('Transparent Background Graph', value=True)
        x_title = st.text_input('X-axis title:')
        y_title = st.text_input('Y-axis title:')
        title_box = st.checkbox('Add title')
        if title_box:
            title_text = st.text_input(label='Title text:', value=title)
        else:
            title_text = None
        if w:
            max_symbols = st.number_input('Max symbols per line:', min_value=10, max_value=100, value=20)
        else:
            max_symbols = None
        if add_legend_pos:
            legend_position_type = st.radio('Select legend positioning:', ['Easy', 'Advanced'])
            if legend_position_type == 'Easy':
                col1, col2 = st.columns(2)
                legend_position = (col1.selectbox('x-axis:', ['left', 'center', 'right'], index=2),
                                   col2.selectbox('y-axis:', ['bottom', 'middle', 'top'], index=2))
            elif legend_position_type == 'Advanced':
                col1, col2 = st.columns(2)
                legend_position = (st.number_input('x-value:', max_value=2., min_value=-2., value=0.5, step=0.05),
                                   st.number_input('y-value:', max_value=2., min_value=-2., value=-0.2, step=0.05),
                                   col1.selectbox('x-alignment:', ['right', 'center', 'left'], index=1),
                                   col2.selectbox('y-alignment:', ['bottom', 'middle', 'top'], index=2),
                                   st.radio('Legend Orientation:', ['horizontal', 'vertical']))
        else:
            legend_position = None
    return GraphParams(width, height, font_size, font, x_title, y_title, title_box, title_text, max_symbols,
                       legend_position, transparent, inside_outside)


st.title("Graph Creator")
st.write("This tool will help you to create various graphs 📉")
st.sidebar.header("Graph Parameters")
uploaded_file = st.sidebar.file_uploader("Upload dataframe", type='csv')
multilevel_columns = st.sidebar.checkbox("Dataframe contains multilevel columns:", value=False)

if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))

    # To read file as string:
    string_data = stringio.read()

    # Can be used wherever a "file-like" object is accepted:
    if multilevel_columns:
        dataframe = pd.read_csv(uploaded_file, header=[0, 1])
        dataframe.columns.set_levels(dataframe.columns.levels[0].str.strip(), level=0, inplace=True)
        dataframe.columns.set_levels(dataframe.columns.levels[1].str.strip(), level=1, inplace=True)
    else:
        dataframe = pd.read_csv(uploaded_file)
        dataframe.columns = dataframe.columns.str.strip()
    st.header("Inputed Dataframe:")
    st.dataframe(dataframe)

    option = st.sidebar.selectbox(
        'Choose graph type to plot',
        ('Bar Graph for Categorical Data', 'Bar Graph for Numeric Data', 'Group Bar Graph',
         'Multiple-Choice Question Bar Graph', 'Pie Chart', 'Gauge Graph', 'Horizontal Bar Graph',
         'Horizontal Bar Chart for NPS scores',
         'Self-Assessment Graph (requires specific data format)', 'Line Graph',
         'Stacked Bar Graph', 'Scatter Graph with Regression Line', 'Histogram'))
    session_state = SessionState.get(name='', options='')
    graph_creator = DataAnalyzer(dataframe)
    if option == 'Bar Graph for Categorical Data':
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
            percents = st.checkbox('Show percents on graph (if not checked, absolute values will be shown)',
                                   value=True)
            gp = graph_params(1500, 780, 27, False, dataframe.loc[0, column], True,
                              'The default options for this graph is: \n'
                              'rectangular - 1550x820 with 29 font, \n'
                              'square - 1200x900 with 27 font')
        if column:
            st.header('Resulting Graph')
            graph_for_plot = graph_creator.create_bar_graph(column, width=gp.width, height=gp.height,
                                                            font_size=gp.font_size, font=gp.font,
                                                            order=order, one_color=True,
                                                            x_title=gp.x_title, y_title=gp.y_title,
                                                            title=gp.title, title_text=gp.title_text,
                                                            max_symb=gp.max_symbols, transparent=gp.transparent,
                                                            percents=percents)
            st.plotly_chart(graph_for_plot)
            scale = 4 if gp.width * 2 > 3000 else 5 if gp.width > 2300 else 6
    elif option == 'Group Bar Graph':
        if multilevel_columns:
            course_column = st.sidebar.selectbox('Select course column to create graph for:',
                                                 tuple(dataframe.columns.levels[0]))
            columns = st.sidebar.selectbox('Select column to create graph for:', tuple(dataframe.columns.levels[0]))
        else:
            columns = st.sidebar.multiselect('Select columns to create graph for:', tuple(dataframe.columns))
            course_column = None
        if not multilevel_columns:
            remove_1_part = st.sidebar.checkbox('Remove first part of the question for options')
            if remove_1_part:
                remove = True
            else:
                remove = False
        else:
            remove = False
        with st.sidebar:
            save = st.checkbox('Save the order')
            if columns:
                if not save:
                    if not multilevel_columns:
                        options = []
                        for col in columns:
                            options.extend(list(dataframe[col].unique())[1:])
                        options = sorted([x for x in list(set(options)) if str(x) != 'nan'])
                    else:
                        options = sorted([col.strip() for col in dataframe[columns].columns])
                    ord = check_if_order_is_known(options)
                    if ord is not None:
                        options = ord
                    order = st.text_area('Select the order for the options:',
                                         value=',\n'.join(options), height=150)
                    session_state.options = ',\n'.join(options)
                else:
                    order = st.text_area('Select the order for the options:',
                                         value=session_state.options, height=150)

            gp = graph_params(1500, 700, 21, True, '', True)
        if columns:
            st.header('Resulting Graph')

            graph_for_plot = graph_creator.create_bar_graph_group(columns, width=gp.width, height=gp.height,
                                                                  font_size=gp.font_size, font=gp.font,
                                                                  order=order, x_title=gp.x_title, y_title=gp.y_title,
                                                                  title=gp.title, title_text=gp.title_text,
                                                                  max_symb=gp.max_symbols,
                                                                  legend_position=gp.legend_position,
                                                                  transparent=gp.transparent, remove=remove,
                                                                  multilevel_columns=multilevel_columns,
                                                                  course_col=course_column)
            st.plotly_chart(graph_for_plot)
            scale = 5 if gp.width * 2 > 3000 else 6 if gp.width > 2300 else 7
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
            gp = graph_params(900, 600, 20, False, dataframe.loc[0, column], True)
        if column:
            st.header('Resulting Graph')
            graph_for_plot = graph_creator.create_chart_for_categories(column, width=gp.width, height=gp.height,
                                                                       font_size=gp.font_size, font=gp.font,
                                                                       order=order, one_color=True,
                                                                       x_title=gp.x_title, y_title=gp.y_title,
                                                                       title=gp.title, title_text=gp.title_text,
                                                                       max_symb=gp.max_symbols,
                                                                       transparent=gp.transparent)
            st.plotly_chart(graph_for_plot)
            scale = 5 if gp.width * 2 > 3000 else 6 if gp.width > 2300 else 7

    elif option == 'Pie Chart':
        with st.sidebar:
            what_show = st.selectbox('What to show in pie chart?', ['Percent', 'Percent and Label'], index=0)

        from_column = st.sidebar.checkbox('Calculate from column instead of using predetermined values', value=True)
        if from_column:
            label_column, numbers_column = None, None
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
            with st.sidebar:
                gp = graph_params(900, 600, 25, True, dataframe.loc[0, column], False)

        else:
            column = None
            label_column = st.sidebar.selectbox('Select label column to create graph for:', tuple(dataframe.columns))
            numbers_column = st.sidebar.selectbox('Select numbers column to create graph for:',
                                                  tuple(dataframe.columns))
            with st.sidebar:
                save = st.checkbox('Save the order')
                if not save:
                    unique_vals = list(graph_creator.get_categories_from_columns(label_column, ',(\S)')['index'])
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
            with st.sidebar:
                gp = graph_params(900, 600, 25, True, dataframe.loc[0, label_column], False)

        if column or label_column:
            st.header('Resulting Graph')
            if from_column:
                graph_for_plot = graph_creator.create_pie_chart(column=column, width=gp.width, height=gp.height,
                                                                font_size=gp.font_size, font=gp.font,
                                                                x_title=gp.x_title, y_title=gp.y_title,
                                                                title=gp.title, title_text=gp.title_text,
                                                                what_show=what_show, legend_position=gp.legend_position,
                                                                transparent=gp.transparent, order=order)
            else:
                graph_for_plot = graph_creator.create_pie_chart(label_column=label_column,
                                                                numbers_column=numbers_column,
                                                                width=gp.width, height=gp.height,
                                                                font_size=gp.font_size, font=gp.font,
                                                                x_title=gp.x_title, y_title=gp.y_title,
                                                                title=gp.title, title_text=gp.title_text,
                                                                what_show=what_show, legend_position=gp.legend_position,
                                                                transparent=gp.transparent, order=order)
            st.plotly_chart(graph_for_plot)
            scale = 5 if gp.width * 2 > 3000 else 6 if gp.width > 2300 else 7

    elif option == 'Gauge Graph':
        column = st.sidebar.selectbox('Select column to create graph for:', tuple(dataframe.columns))
        with st.sidebar:
            with st.expander("Graph Parameters"):
                width = st.number_input('Width', min_value=500, max_value=5000, value=700)
                height = st.number_input('Height', min_value=300, max_value=5000, value=500)
                font_size = st.number_input('Font Size', min_value=10, max_value=60, value=20)
                font = st.selectbox('Font', ['Helvetica', 'Helvetica Neue', 'Arial'], index=1)
                transparent = st.checkbox('Transparent Background Graph')

        if column:
            st.header('Resulting Graph')
            graph_for_plot = graph_creator.create_gauge_graph(column, width=width, height=height,
                                                              font_size=font_size, font=font, transparent=transparent)
            st.plotly_chart(graph_for_plot)
            scale = 5 if width * 2 > 3000 else 6 if width > 2300 else 7

    elif option == 'Horizontal Bar Graph':
        column = st.sidebar.selectbox('Select column to create graph for:', tuple(dataframe.columns))
        ord = ['Detractor', 'Passive', 'Promoter']
        order = st.sidebar.text_area('Select the order for the options:',
                                     value=',\n'.join(ord), height=150)
        session_state.options = ',\n'.join(ord)
        with st.sidebar:
            with st.expander("Graph Parameters"):
                width = st.number_input('Width', min_value=500, max_value=3000, value=900)
                height = st.number_input('Height', min_value=300, max_value=3000, value=500)
                font_size = st.number_input('Font Size', min_value=10, max_value=80, value=40)
                font = st.selectbox('Font', ['Helvetica', 'Helvetica Neue', 'Arial'], index=1)
                transparent = st.checkbox('Transparent Background Graph')

        if column:
            st.header('Resulting Graph')
            graph_for_plot = graph_creator.create_horizontal_bar_graph(column, width=width, height=height,
                                                                       font_size=font_size, font=font,
                                                                       transparent=transparent, order=order)
            st.plotly_chart(graph_for_plot)
            scale = 5 if width * 2 > 3000 else 6 if width > 2300 else 7

    elif option == 'Bar Graph for Numeric Data':
        with st.sidebar:
            column = st.selectbox('Select label column to create graph for:', tuple(dataframe.columns))
            data_column = st.selectbox('Select data column to create graph for:', tuple(dataframe.columns))
            round_nums = st.number_input('Rounding of Inputs', min_value=1, max_value=10, step=1, value=2)
            save = st.checkbox('Save the order')
            if not save:
                unique_vals = [x for x in list(dataframe[column].unique()) if str(x) != 'nan']
                ord = check_if_order_is_known(unique_vals)
                if ord is None:
                    ord = sorted(unique_vals)
                order = st.text_area('Select the order for the options:',
                                     value=',\n'.join(ord), height=150)
                session_state.options = ',\n'.join(ord)
            else:
                order = st.text_area('Select the order for the options:',
                                     value=session_state.options, height=150)
            show_average = st.checkbox('Show average line on the graph')
            if show_average:
                average_line_x = st.selectbox('Select bar to locate the average line label', order.split(",\n"),
                                              index=len(order.split(",\n")) - 1)
            else:
                average_line_x = 0
            percents = st.checkbox('Show percents on graph (if not checked, absolute values will be shown)',
                                   value=True)
            gp = graph_params(1500, 780, 27, False, dataframe.loc[0, column], True,
                              'The default options for this graph is: \n'
                              'rectangular - 1550x820 with 29 font, \n'
                              'square - 1200x900 with 27 font', show_inside_outside=True)
        if column:
            st.header('Resulting Graph')
            graph_for_plot = graph_creator.create_simple_bar(course_col=column, column=data_column,
                                                             width=gp.width, height=gp.height,
                                                             font_size=gp.font_size, font=gp.font,
                                                             order=order, one_color=True,
                                                             x_title=gp.x_title, y_title=gp.y_title,
                                                             title=gp.title, title_text=gp.title_text,
                                                             max_symb=gp.max_symbols, transparent=gp.transparent,
                                                             percents=percents, show_average=show_average,
                                                             avg_line_title='',
                                                             inside_outside_pos=gp.inside_outside,
                                                             round_nums=round_nums,
                                                             average_line_x=average_line_x
                                                             )
            st.plotly_chart(graph_for_plot)
            scale = 4 if gp.width * 2 > 3000 else 5 if gp.width > 2300 else 6
    elif option == 'Horizontal Bar Chart for NPS scores':
        with st.sidebar:
            column = st.selectbox('Select label column to create graph for:', tuple(dataframe.columns))
            data_column = st.selectbox('Select data column to create graph for:', tuple(dataframe.columns))
            round_nums = st.number_input('Rounding of Inputs', min_value=1, max_value=10, step=1, value=2)
            percents = st.checkbox('Show percents on graph (if not checked, absolute values will be shown)',
                                   value=True)
            gp = graph_params(1500, 780, 27, False, dataframe.loc[0, column], True,
                              'The default options for this graph is: \n'
                              'rectangular - 1550x820 with 29 font, \n'
                              'square - 1200x900 with 27 font')
        if column:
            st.header('Resulting Graph')
            graph_for_plot = graph_creator.plot_horizontal_bar_for_nps(course_col=column, column=data_column,
                                                                       width=gp.width, height=gp.height,
                                                                       font_size=gp.font_size, font=gp.font,
                                                                       x_title=gp.x_title, y_title=gp.y_title,
                                                                       title=gp.title, title_text=gp.title_text,
                                                                       max_symb=gp.max_symbols,
                                                                       transparent=gp.transparent,
                                                                       percents=percents,
                                                                       round_nums=round_nums)
            st.plotly_chart(graph_for_plot)
            scale = 4 if gp.width * 2 > 3000 else 5 if gp.width > 2300 else 6

    elif option == 'Self-Assessment Graph (requires specific data format)':
        with st.sidebar:
            time_column = st.selectbox('Select pre-post-column column to create graph for:', tuple(dataframe.columns))
            round_nums = st.number_input('Rounding of Inputs', min_value=1, max_value=10, step=1, value=1)
            gp = graph_params(1500, 780, 20, False, 'Learning Outcomes Self-Assessment Chart', True,
                              'The default options for this graph is: \n'
                              'rectangular - 1550x820 with 29 font, \n'
                              'square - 1200x900 with 27 font')
            coordinate_of_legend_y = st.text_input('Coordinate of legend\'s y', value='-0.3')
        if time_column:
            st.header('Resulting Graph')
            graph_for_plot = graph_creator.plot_self_assessment(time_col=time_column,
                                                                width=gp.width, height=gp.height,
                                                                font_size=gp.font_size, font=gp.font,
                                                                x_title=gp.x_title, y_title=gp.y_title,
                                                                title=gp.title, title_text=gp.title_text,
                                                                max_symb=gp.max_symbols,
                                                                transparent=gp.transparent,
                                                                round_nums=round_nums,
                                                                legend_y_coord=coordinate_of_legend_y)
            st.plotly_chart(graph_for_plot)
            scale = 4 if gp.width * 2 > 3000 else 5 if gp.width > 2300 else 6

    elif option == 'Line Graph':
        with st.sidebar:
            time_column = st.selectbox('Select pre-post-column column to create graph for:', tuple(dataframe.columns))
            gp = graph_params(900, 550, 20, False, 'Learning Outcomes Scores Timeline', True,
                              'The default options for this graph is: \n'
                              'rectangular - 1550x820 with 29 font, \n'
                              'square - 1200x900 with 27 font')
        if time_column:
            st.header('Resulting Graph')
            graph_for_plot = graph_creator.plot_line(time_col=time_column,
                                                     width=gp.width, height=gp.height,
                                                     font_size=gp.font_size, font=gp.font,
                                                     x_title=gp.x_title, y_title=gp.y_title,
                                                     title=gp.title, title_text=gp.title_text,
                                                     transparent=gp.transparent)
            st.plotly_chart(graph_for_plot)
            scale = 4 if gp.width * 2 > 3000 else 5 if gp.width > 2300 else 6

    elif option == 'Stacked Bar Graph':
        column = st.sidebar.selectbox('Select column to create graph for:', tuple(dataframe.columns))
        first_column = st.sidebar.selectbox('Select value 1 column :', tuple(dataframe.columns))
        second_column = st.sidebar.selectbox('Select value 2 column:', tuple(dataframe.columns))

        with st.sidebar:
            percents = st.checkbox('Show percents on graph (if not checked, absolute values will be shown)',
                                   value=True)
            include_total = st.checkbox('Include total (if it is in the dataset as the last row)')
            gp = graph_params(1500, 780, 27, False, dataframe.loc[0, column], True,
                              'The default options for this graph is: \n'
                              'rectangular - 1550x820 with 29 font, \n'
                              'square - 1200x900 with 27 font')
        if column:
            st.header('Resulting Graph')
            graph_for_plot = graph_creator.stacked_bar_plot(column, first_column, second_column,
                                                            width=gp.width, height=gp.height,
                                                            font_size=gp.font_size, font=gp.font,
                                                            x_title=gp.x_title, y_title=gp.y_title,
                                                            title=gp.title, title_text=gp.title_text,
                                                            transparent=gp.transparent,
                                                            percents=percents, include_total=include_total,
                                                            max_symb=gp.max_symbols)
            st.plotly_chart(graph_for_plot)
            scale = 4 if gp.width * 2 > 3000 else 5 if gp.width > 2300 else 6

    elif option == 'Scatter Graph with Regression Line':
        first_column = st.sidebar.selectbox('Select value 1 column :', tuple(dataframe.columns))
        second_column = st.sidebar.selectbox('Select value 2 column:', tuple(dataframe.columns))

        with st.sidebar:
            marker_size = st.number_input('Marker size:', 1, 40, 10, 1)
            marker_border_width = st.number_input('Marker border width:', 1, 20, 2, 1)
            gp = graph_params(1500, 780, 27, False, dataframe.loc[0, first_column], True,
                              'The default options for this graph is: \n'
                              'rectangular - 1550x820 with 29 font, \n'
                              'square - 1200x900 with 27 font')
        if first_column and second_column:
            st.header('Resulting Graph')
            graph_for_plot = graph_creator.plot_scatter_with_regression(first_column, second_column,
                                                                        width=gp.width, height=gp.height,
                                                                        font_size=gp.font_size, font=gp.font,
                                                                        x_title=gp.x_title, y_title=gp.y_title,
                                                                        title=gp.title, title_text=gp.title_text,
                                                                        transparent=gp.transparent,
                                                                        marker_size=marker_size,
                                                                        marker_line_width=marker_border_width)
            st.plotly_chart(graph_for_plot)
            scale = 4 if gp.width * 2 > 3000 else 5 if gp.width > 2300 else 6
    elif option == 'Histogram':
        column = st.sidebar.selectbox('Select column to create graph for:', tuple(dataframe.columns))
        with st.sidebar:
            gp = graph_params(1200, 600, 27, False, dataframe.loc[0, column], False,
                                  'The default options for this graph is: \n'
                                  'rectangular - 1550x820 with 29 font, \n'
                                  'square - 1200x900 with 27 font')
        if column:
            st.header('Resulting Graph')
            graph_for_plot = graph_creator.plot_histogram(column, width=gp.width, height=gp.height,
                                                                font_size=gp.font_size, font=gp.font,
                                                                x_title=gp.x_title, y_title=gp.y_title,
                                                                title=gp.title, title_text=gp.title_text,
                                                                transparent=gp.transparent)
            st.plotly_chart(graph_for_plot)
            scale = 4 if gp.width * 2 > 3000 else 5 if gp.width > 2300 else 6
