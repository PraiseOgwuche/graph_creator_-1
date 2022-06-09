import pandas as pd
import plotly.graph_objects as go
from typing import Optional, List
import re
import numpy as np
from copy import deepcopy
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table
from math import isclose, sqrt

pd.options.mode.chained_assignment = None

order = []


class DataAnalyzer:
    large_rockwell_template = dict(
        layout=go.Layout(title_font=dict(family="Hevletica", size=24))
    )
    color_palette = list(reversed(['rgb(132,29,22)', 'rgb(179,39,39)', 'rgb(222,46,37)', 'rgb(244,81,28)',
                                   'rgb(215,116,102)', 'rgb(252,156,124)',
                                   'rgb(243,210,143)',
                                   'rgb(60,54,50)', 'rgb(99,99,99)', 'rgb(153,153,153)', 'rgb(211,211,211)']))
    color_palette2 = ['rgb(151,144,139)', 'rgb(60,54,50)', 'rgb(243,210,143)',
                      'rgb(222,46,37)', 'rgb(132,29,22)', 'rgb(252,156,124)', 'rgb(153,153,153)', 'rgb(211,211,211)']
    color_pallete3 = ['rgb(153,153,153)', 'rgb(222,46,37)']

    def __init__(self, data: pd.DataFrame):
        self.df = data

    @staticmethod
    def read_data(data: str) -> pd.DataFrame:
        df = pd.read_csv(data)
        return df

    def show_data(self) -> pd.DataFrame:
        return self.df

    def capitalize_list(self, l):
        return [i.capitalize() for i in l]

    def error_gen(self, actual: float, rounded: float):
        divisor = sqrt(1.0 if actual < 1.0 else actual)
        return abs(rounded - actual) ** 2 / divisor

    def round_to_100(self, percents: List[float]):
        if not isclose(sum(percents), 100):
            raise ValueError
        n = len(percents)
        rounded = [int(x) for x in percents]
        up_count = 100 - sum(rounded)
        errors = [(self.error_gen(percents[i], rounded[i] + 1) -
                   self.error_gen(percents[i], rounded[i]), i) for i in range(n)]
        rank = sorted(errors)
        for i in range(up_count):
            rounded[rank[i][1]] += 1
        return rounded

    def create_bar_graph(self, column: str, title: Optional[bool] = False, title_text: Optional[str] = None,
                         order: Optional[str] = None,
                         x_title: Optional[str] = None, y_title: Optional[str] = None,
                         one_color: bool = True, width: int = 900, height: int = 550,
                         font_size: int = 20, font: str = 'Hevletica Neue', max_symb: int = 20,
                         transparent: bool = False, percents: bool = True):
        if percents:
            df_temp = pd.DataFrame(self.df.loc[1:, column].value_counts(normalize=True))
            df_temp[column] = np.array(self.round_to_100(np.array(df_temp[column] * 100))) / 100
        else:
            df_temp = pd.DataFrame(self.df.loc[1:, column].value_counts())
        order = order.split(',\n')
        if order:
            not_in_df = [index for index in order if index not in set(list(
                df_temp.index))]
            for i in not_in_df:
                df_temp.loc[i, :] = [np.nan] * len(df_temp.columns)
            df_temp = df_temp.loc[order, ]
        df_temp = df_temp.fillna(0).reset_index()
        x = list(df_temp['index'])
        x = [split_string(string, max_symb) for string in x]
        return self.plot_bar(x, list(df_temp[column]), width, height, font_size, font,
                             title=title_text if title else None,
                             x_title=x_title, y_title=y_title, one_color=one_color,
                             transparent=transparent, percents=percents)

    def create_bar_graph_group(self, columns: List[str], title: Optional[bool] = False,
                               title_text: Optional[str] = None, order: str = None,
                               x_title: Optional[str] = None, y_title: Optional[str] = None, max_symb: int = 20,
                               names: Optional[List[str]] = None,
                               width: int = 900, height: int = 550,
                               font_size: int = 20, font: str = 'Hevletica Neue',
                               legend_position: List[str] = ('top', 'left'),
                               transparent: bool = False, remove: bool = False,
                               multilevel_columns: bool = False, course_col: Optional[str] = None):
        order = order.split(',\n')
        if len(order) <= 2:
            palette = ['rgb(170,170,170)', 'rgb(222,46,37)']
        elif len(order) <= 5:
            palette = self.color_palette2
        else:
            palette = self.color_palette
        if not multilevel_columns:
            list_vals = [self.df.loc[0, column] for column in columns]
            for ind, val in enumerate(list_vals):
                if remove:
                    title_text, list_vals[ind] = re.split(' - ', list_vals[ind])
                list_vals[ind] = split_string(list_vals[ind], max_symb)
            fig = go.Figure()
            dict_nums = {}
            for index, response in enumerate(order):
                list_num = []
                for column in columns:
                    df_temp = pd.DataFrame(self.df.loc[1:, column].value_counts(
                        normalize=True))
                    if not response in df_temp.index:
                        list_num.append(0)
                    else:
                        list_num.append(df_temp.loc[response, column])
                dict_nums[response] = (index, list_num)
            for val in range(len(list_vals)):
                percentages = []
                for key in dict_nums.keys():
                    percentages.append(dict_nums[key][1][val])
                percentages = np.array(self.round_to_100(np.array(percentages) * 100)) / 100
                for ind, key in enumerate(list(dict_nums.keys())):
                    dict_nums[key][1][val] = percentages[ind]
            for index, response in enumerate(order):
                fig.add_trace(go.Bar(x=list_vals,
                                     y=dict_nums[response][1],
                                     name=names[index] if names else response,
                                     marker_color=palette[dict_nums[response][0]],
                                     texttemplate='%{y}', textposition='outside',
                                     textfont_size=font_size
                                     ))
        else:
            dict_nums = {col: list(self.df[columns][col]) for col in order}
            fig = go.Figure()
            col = self.df[course_col].columns[0]
            x = list(self.df[course_col][col])
            x = [split_string(string, max_symb) for string in x]
            for index, response in enumerate(order):
                    fig.add_trace(go.Bar(x=x,
                                         y=dict_nums[response],
                                         name=names[index] if names else response,
                                         marker_color=palette[index],
                                         texttemplate='%{y}', textposition='outside',
                                         textfont_size=font_size
                                         ))
        if len(legend_position) == 2:
            y_legend = 1 if legend_position[1] == 'top' else 0.5 if legend_position[1] == 'middle' else -0.3
            x_legend = 1 if legend_position[0] == 'right' else 0.5 if legend_position[0] == 'center' else -0.15
            orientation = 'h' if legend_position[0] == 'center' else 'v'
            x_anchor = 'left'
            y_anchor = 'top'

        else:
            y_legend = legend_position[1]
            x_legend = legend_position[0]
            orientation = 'v' if legend_position[4] == 'vertical' else 'h'
            x_anchor = legend_position[2]
            y_anchor = legend_position[3]
        fig.update_layout(
            font_family=font,
            font_size=font_size,
            title=title_text if title else '',
            title_font_size=font_size * 1.5,
            xaxis_tickfont_size=font_size,
            xaxis=dict(
                title=x_title if x_title else '',
                titlefont_size=font_size,
                tickfont_size=font_size
            ),
            yaxis=dict(
                title=y_title if y_title else '',
                titlefont_size=font_size,
                tickfont_size=font_size,
                tickformat="1%"
            ),
            bargap=0.15,
            template=self.large_rockwell_template,
            legend=dict(font_size=font_size,
                        font_family=font,
                        orientation=orientation,
                        y=y_legend,
                        x=x_legend,
                        xanchor=x_anchor,
                        yanchor=y_anchor),
            width=width, height=height
        )
        if transparent:
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)')
        else:
            fig.update_layout(plot_bgcolor='rgb(255,255,255)')
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black')
        fig.update_yaxes(showgrid=False, gridwidth=1, gridcolor='lightgrey', automargin=True)
        fig.update_xaxes(tickangle=0, automargin=True)
        return fig

    def get_categories_from_columns(self, column: str, sep: str,
                                    order: Optional[List[str]] = None) -> pd.DataFrame:
        temp_df = self.df.copy()
        temp_df.loc[1:, column] = [re.split(sep, str(i)) for i in temp_df.loc[1:, column]]
        df_res = pd.DataFrame(columns=['count'])
        responses_num = len(temp_df.loc[1:, column])
        for index_row, tag_list in enumerate(temp_df.loc[1:, column]):
            for index_tag, tag in enumerate(tag_list):
                if len(tag) == 1:
                    temp_df.loc[index_row + 1, column][index_tag + 1] = tag + \
                                                                        temp_df.loc[index_row + 1, column][
                                                                            index_tag + 1]
                    continue
                tag = tag.strip()
                if tag[-1] == '.':
                    tag = tag[:-1]
                if tag in list(df_res.index):
                    df_res.loc[tag, 'count'] += 1
                else:
                    df_res.loc[tag, 'count'] = 1
        if order:
            for string in order:
                if not string in df_res.index:
                    df_res.loc[string, 'count'] = 0
        df_res = df_res.reset_index()
        df_res = df_res[df_res['index'] != 'nan']
        df_res['count'] = [i / responses_num for i in df_res['count']]
        df_res['count'] = [round(i, 2) for i in df_res['count']]
        df_res['index'] = pd.Categorical(df_res['index'], order)
        return df_res.sort_values('index')

    def create_chart_for_categories(self, column: str, title: Optional[bool] = False,
                                    title_text: Optional[str] = None, order: Optional[str] = None,
                                    x_title: Optional[str] = None, y_title: Optional[str] = None,
                                    one_color: bool = False, sep: str = ',(\S)', max_symb: int = 20,
                                    width: int = 900, height: int = 550,
                                    font_size: int = 20, font: str = 'Hevletica Neue',
                                    transparent: bool = False):
        order = order.split(',\n')
        df_res = self.get_categories_from_columns(column, sep, order)
        df_res['index'] = [split_string(string, max_symb) for string in df_res['index']]

        return self.plot_bar(df_res['index'], df_res['count'], width, height, font_size, font,
                             title=title_text if title else None,
                             x_title=x_title, y_title=y_title, one_color=one_color,
                             transparent=transparent)

    def plot_self_assessment(self, time_col: str, title: Optional[bool] = False,
                             title_text: Optional[str] = None,
                             x_title: Optional[str] = None, y_title: Optional[str] = None,
                             width: int = 900, height: int = 550,
                             font_size: int = 20, font: str = 'Hevletica Neue', max_symb: int = 20,
                             transparent: bool = False,
                             round_nums: int = 2, legend_y_coord: float = -0.3):
        fig = go.Figure()
        df = self.df
        df = df.set_index(time_col)
        palette = self.color_pallete3
        x = list(df.columns)
        x = self.capitalize_list(x)
        x = [split_string(string, max_symb) for string in x]
        round_nums = int(round_nums)
        for index, response in enumerate(['Pre-semester',
                                          'Post-semester']):
            y = [round(i, round_nums) for i in df.loc[response, :]]
            fig.add_trace(go.Bar(x=x,
                                 y=y,
                                 name=response,
                                 marker_color=palette[-index],
                                 text=y, textposition='outside',
                                 textfont_size=font_size
                                 ))
        fig.update_layout(
            font_family=font,
            font_size=font_size,
            title=title_text if title else '',
            title_font_size=font_size * 1.5,
            xaxis_tickfont_size=font_size,
            xaxis=dict(
                title=x_title if x_title else '',
                titlefont_size=font_size,
                tickfont_size=font_size
            ),
            yaxis=dict(
                title=y_title if y_title else '',
                titlefont_size=font_size,
                tickfont_size=font_size,
                tickformat="1"
            ),
            bargap=0.3,
            template=self.large_rockwell_template,
            legend=dict(font_size=font_size,
                        font_family=font,
                        orientation='h',
                        y=float(legend_y_coord),
                        x=0.5,
                        xanchor='center',
                        yanchor='top'),
            width=width, height=height
        )
        if transparent:
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)')
        else:
            fig.update_layout(plot_bgcolor='rgb(255,255,255)')
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black')
        fig.update_yaxes(showgrid=False, gridwidth=1, gridcolor='lightgrey', automargin=True)
        fig.update_xaxes(tickangle=0, automargin=True)
        return fig

    def plot_bar(self, x: list, y: list, width: int, height: int, font_size: int,
                 font: str, title: Optional[str] = None,
                 x_title: Optional[str] = None,
                 y_title: Optional[str] = None,
                 one_color: bool = False,
                 transparent: bool = False,
                 percents: bool = True,
                 textposition: str = 'outside',
                 showlegend: bool = False,
                 error_y: Optional[list] = None,
                 insidetextanchor: str = 'end'):
        fig = go.Figure()
        error_y = dict(type='data', array=error_y)
        if one_color:
            fig.add_trace(go.Bar(x=x,
                                 y=y,
                                 marker_color='rgb(224,44,36)',
                                 error_y=error_y,
                                 texttemplate='%{y}' if percents else '%{y}',
                                 textfont_size=font_size, textposition=textposition,
                                 insidetextanchor=insidetextanchor
                                 ))
        else:
            fig.add_trace(go.Bar(x=x,
                                 y=y,
                                 marker_color=self.color_palette[:len(x)],
                                 error_y=error_y, insidetextanchor=insidetextanchor,
                                 texttemplate='%{y}' if percents else '%{y}', textposition=textposition,
                                 ))

        fig.update_layout(

            title=title,
            title_font_size=font_size * 1.5,
            font_family=font,
            font_size=font_size,
            xaxis=dict(
                type='category',
                title=x_title if x_title else '',
                titlefont_size=font_size,
                tickfont_size=font_size
            ),
            yaxis=dict(
                title=y_title if y_title else '',
                titlefont_size=font_size,
                tickfont_size=font_size,
                tickformat='1%' if percents else '1'
            ),
            bargap=0.15,  # gap between bars of adjacent location coordinates.
            template=self.large_rockwell_template,
            width=width,
            height=height,
            showlegend=showlegend
        )
        if transparent:
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)')
        else:
            fig.update_layout(plot_bgcolor='rgb(255,255,255)')

        fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black')
        fig.update_yaxes(showgrid=False, gridwidth=1, gridcolor='lightgrey', automargin=True)
        fig.update_xaxes(tickangle=0, automargin=True)
        return fig

    def create_pie_chart(self, width: int, height: int, font_size: int,
                         font: str, title: Optional[str] = None, title_text: Optional[str] = None,
                         x_title: Optional[str] = None,
                         y_title: Optional[str] = None,
                         what_show: Optional[str] = None, legend_position: List[str] = ('top', 'left'),
                         transparent: bool = False, column: Optional[str] = None,
                         label_column: Optional[str] = None, numbers_column: Optional[str] = None,
                         order: Optional[str] = None
                         ):
        order = order.split(',\n')
        order = {key: i for i, key in enumerate(order)}
        if column:
            dictionary = dict(self.df.loc[1:, column].dropna().value_counts(normalize=True))
            labels = list(dictionary.keys())
            vals = np.array(self.round_to_100(np.array(list(dictionary.values())) * 100)) / 100
        else:
            labels = list(self.df[label_column])
            nums = np.array(list(self.df[numbers_column])) / sum(np.array(list(self.df[numbers_column])))
            vals = np.array(self.round_to_100(np.array(nums) * 100)) / 100
        labels, vals = zip(*sorted(zip(labels, vals), key=lambda d: order[d[0]]))
        text_temp = '%{percent:1.0%}' if what_show == 'Percent' else 'label+percent'
        if len(labels) <= 2:
            palette = ['rgb(222,46,37)', 'rgb(170,170,170)']
        elif len(labels) <= 5:
            palette = self.color_palette2
        else:
            palette = self.color_palette
        if what_show == 'Percent':
            fig = go.Figure(data=[go.Pie(labels=labels, values=vals,
                                         marker_colors=palette[:len(labels)],
                                         texttemplate=text_temp, sort=False)])
        else:
            fig = go.Figure(data=[go.Pie(labels=labels, values=vals,
                                         marker_colors=palette[:len(labels)],
                                         textinfo=text_temp, sort=False)])
        if len(legend_position) == 2:
            y_legend = 1 if legend_position[1] == 'top' else 0.5 if legend_position[1] == 'middle' else -0.3
            x_legend = 1 if legend_position[0] == 'right' else 0.5 if legend_position[0] == 'center' else -0.15
            orientation = 'h' if legend_position[0] == 'center' else 'v'
            x_anchor = 'left'
            y_anchor = 'top'
        else:
            y_legend = legend_position[1]
            x_legend = legend_position[0]
            orientation = 'v' if legend_position[4] == 'vertical' else 'h'
            x_anchor = legend_position[2]
            y_anchor = legend_position[3]
        fig.update_layout(
            title=title_text if title else '',
            title_font_size=font_size * 1.5,
            font_family=font,
            font_size=font_size,
            xaxis=dict(
                title=x_title if x_title else '',
                titlefont_size=font_size,
                tickfont_size=font_size,
            ),
            yaxis=dict(
                title=y_title if y_title else '',
                titlefont_size=font_size,
                tickfont_size=font_size,
                tickformat='1%'
            ),
            bargap=0.15,  # gap between bars of adjacent location coordinates.
            template=self.large_rockwell_template,
            width=width,
            height=height,
            legend=dict(font_size=font_size,
                        font_family=font,
                        orientation=orientation,
                        y=y_legend,
                        x=x_legend,
                        xanchor=x_anchor,
                        yanchor=y_anchor),
        )
        if transparent:
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)')
        else:
            fig.update_layout(plot_bgcolor='rgb(255,255,255)')
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black')
        fig.update_yaxes(showgrid=False, gridwidth=1, gridcolor='lightgrey', automargin=True)
        fig.update_xaxes(tickangle=0, automargin=True)
        return fig

    def create_gauge_graph(self, column: str, width: int, height: int,
                           font_size: int, font: str, transparent: bool):
        promoters = (self.df.loc[1:, column] == 'Promoter').sum() / (len(self.df) - 1)
        detractors = (self.df.loc[1:, column] == 'Detractor').sum() / (len(self.df) - 1)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(100 * (promoters - detractors), 1),
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': [-100, 100]},
                   'bar': {'color': 'rgb(224,44,36)', 'thickness': 1}}))

        fig.update_layout(font_family=font, font_size=font_size, width=width,
                          height=height)
        if transparent:
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)')
        else:
            fig.update_layout(plot_bgcolor='rgb(255,255,255)')

        return fig

    def create_horizontal_bar_graph(self, column: str, order: Optional[str] = None,
                                    width: int = 900, height: int = 500,
                                    transparent: bool = False,
                                    font_size: int = 20, font: str = 'Hevletica Neue'):
        order = order.split(',\n')
        df_temp = pd.DataFrame(self.df.loc[1:, column].value_counts(
            normalize=True))
        df_temp[column] = self.round_to_100(np.array(df_temp[column] * 100))
        if order:
            not_in_df = [index for index in order if index not in set(list(
                df_temp.index))]
            for i in not_in_df:
                df_temp.loc[i, :] = [np.nan] * len(df_temp.columns)
            df_temp = df_temp.loc[order,]
        df_temp = df_temp.fillna(0).reset_index()
        df_temp = df_temp.sort_values(by='index', ascending=True)
        fig = go.Figure()
        annotations = []
        for row, color in zip(range(len(df_temp)), ['rgb(60,54,50)', 'rgb(222,46,37)', 'rgb(132,29,22)']):
            fig.add_trace(go.Bar(
                y=[''],
                x=[df_temp.loc[row, column]],
                name=df_temp.loc[row, 'index'],
                orientation='h',
                marker=dict(color=color)
            ))
        # labeling the first percentage of each bar (x_axis)
        if df_temp.loc[0, column] > 5:
            annotations.append(dict(xref='x', yref='y',
                                    x=df_temp.loc[0, column] / 2, y=0,
                                    text=' ' + str(
                                        int(df_temp.loc[0, column])) + '%' + '<br> <span style="font-size: 25px;">' +
                                         df_temp.loc[0, 'index'] + '</span>',
                                    font=dict(family=font, size=font_size,
                                              color='rgb(255, 255, 255)'),
                                    showarrow=False))
        space = df_temp.loc[0, column]
        for i in range(1, len(df_temp[column])):
            # labeling the rest of percentages for each bar (x_axis)
            if df_temp.loc[i, column] > 5:
                annotations.append(dict(xref='x', yref='y',
                                        x=space + (df_temp.loc[i, column] / 2), y=0,
                                        text=' ' + str(
                                            int(df_temp.loc[i, column])) + '%' + '<br> <span style="font-size: 25px;">' +
                                             df_temp.loc[i, 'index'] + '</span>',
                                        font=dict(family=font, size=font_size,
                                                  color='rgb(255, 255, 255)'),
                                        showarrow=False, align="center"))
            space += df_temp.loc[i, column]
        fig.update_layout(
            xaxis=dict(
                showgrid=False,
                showline=False,
                showticklabels=False,
                zeroline=False
            ),
            yaxis=dict(
                showgrid=False,
                showline=False,
                showticklabels=False,
                zeroline=False,
            ),
            barmode='stack',
            plot_bgcolor='rgb(255, 255, 255)',
            showlegend=True,
            annotations=annotations,
            width=width,
            height=height
        )
        fig.update_layout(font_family=font,
                          legend=dict(
                              orientation="h",
                              yanchor="bottom",
                              y=-0.05,
                              xanchor="center",
                              x=0.48,
                              font=dict(size=font_size / 2, color="black"),
                              traceorder='normal'
                          ))
        if transparent:
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)')
        else:
            fig.update_layout(plot_bgcolor='rgb(255,255,255)')

        return fig

    def create_simple_bar(self, avg_line_title: str, average_line_x: str,
                          course_col: str, column: str, y_range: Optional[list] = None,
                          title: Optional[bool] = False, title_text: Optional[str] = None,
                          order: Optional[str] = None,
                          x_title: Optional[str] = None, y_title: Optional[str] = None,
                          one_color: bool = True, width: int = 900, height: int = 550,
                          font_size: int = 20, font: str = 'Hevletica Neue', max_symb: int = 20,
                          transparent: bool = False, percents: bool = True,
                          inside_outside_pos: str = 'outside', show_average: bool = False,
                          round_nums: int = 2, err_column: Optional[str] = None,
                          ):
        df = deepcopy(self.df)
        overall = sum(self.df.loc[:, column]) / len(self.df.loc[:, column])
        order = order.split(',\n')
        df = df.set_index(course_col)
        if order:
            not_in_df = [index for index in order if index not in set(list(
                df.index))]
            for i in not_in_df:
                df.loc[i, :] = [np.nan] * len(df.columns)
            df = df.loc[order,]
        df = df.fillna(0).reset_index()
        x = list(df[course_col]).copy()
        x_copy = x.copy()
        x = [split_string(string, max_symb) for string in x]
        v = df[column]
        if err_column is not None:
            insidetextanchor = 'middle'
            inside_outside_pos = 'inside'
        else:
            insidetextanchor = 'end'
        fig = self.plot_bar(x, [round(i, int(round_nums)) for i in v], width, height, font_size, font,
                            title=title_text if title else None,
                            x_title=x_title, y_title=y_title, one_color=one_color,
                            error_y=df[err_column] if err_column else None,
                            transparent=transparent, percents=percents, textposition=inside_outside_pos,
                            showlegend=False, insidetextanchor=insidetextanchor)
        if y_range is not None:
            fig.update_yaxes(range=y_range)
        if show_average:
            fig.add_trace(go.Scatter(x=x, y=[round(overall, int(round_nums))] * len(x),
                                     marker_color=self.color_palette[-1],
                                     name=avg_line_title))
            addition = '%' if percents else ''
            num = round(overall, int(round_nums)) * 100 if percents else round(overall, int(round_nums))
            num = round(num, int(round_nums))
            if num % 1 == 0:
                num = int(num)
            num = str(num)
            ind = x_copy.index(average_line_x)
            fig.add_annotation(x=x[ind], y=overall + overall * 0.05,
                               text='Average = ' + num + addition,
                               showarrow=False,
                               yshift=10)
        return fig

    def plot_line(self, time_col, title: Optional[bool] = False,
                  title_text: Optional[str] = None, y_range: Optional[list] = None,
                  x_title: Optional[str] = None, y_title: Optional[str] = None,
                  width: int = 900, height: int = 550,
                  font_size: int = 20, font: str = 'Hevletica Neue',
                  transparent: bool = False):

        fig = go.Figure()
        cols = list(self.df.columns)
        cols.remove(time_col)

        if len(cols) > 5:
            colors = self.color_palette
        else:
            colors = self.color_palette2


        for ind, col in enumerate(cols):
            df_new = self.df.dropna(subset=col)
            fig.add_trace(go.Scatter(y=df_new[col], x=pd.to_datetime(df_new[time_col]),
                                     mode='lines+text',
                                     name=col,
                                     line=dict(color=colors[ind + 1], width=4)))
        fig.update_layout(
                    font_family=font,
                    font_size=font_size,
                    title=title_text if title else '',
                    title_font_size=font_size * 1.5,
                    xaxis_tickfont_size=font_size,
                    xaxis=dict(
                        title=x_title if x_title else '',
                        titlefont_size=font_size,
                        tickfont_size=font_size,
                        tickformat='%b, %d'
                    ),
                    yaxis=dict(
                        title=y_title if y_title else '',
                        titlefont_size=font_size,
                        tickfont_size=font_size,
                        tickformat="1",
                        title_standoff=width * 0.02
                    ),
                    template=self.large_rockwell_template,
                    legend=dict(font_size=font_size,
                                font_family=font),
                    width=width, height=height
                )
        if y_range is not None:
            fig.update_yaxes(range=y_range)

        if transparent:
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)')
        else:
            fig.update_layout(plot_bgcolor='rgb(255,255,255)')

        fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black')
        fig.update_yaxes(showgrid=False, gridwidth=1, gridcolor='lightgrey', automargin=True)
        fig.update_xaxes(tickangle=0, automargin=True)
        return fig

    def plot_horizontal_bar_for_nps(self,
                                    course_col: str, column: str, title: Optional[bool] = False,
                                    title_text: Optional[str] = None,
                                    x_title: Optional[str] = None, y_title: Optional[str] = None,
                                    width: int = 900, height: int = 550,
                                    font_size: int = 20, font: str = 'Hevletica Neue', max_symb: int = 20,
                                    transparent: bool = False, percents: bool = True,
                                    round_nums: int = 2):
        df = deepcopy(self.df)
        df = df.set_index(course_col)
        df = df.fillna(0).reset_index()
        x = list(df[course_col]).copy()
        x = [split_string(string, max_symb) for string in x]
        v = self.df[column]
        fig = go.Figure()
        fig.add_trace(go.Bar(y=x, x=[round(i, int(round_nums)) for i in v],
                             marker_color='rgb(224,44,36)',
                             texttemplate='%{x}' if percents else '%{x}%',
                             textfont_size=font_size, orientation='h',
                             textposition='outside'
                             ))
        fig.update_xaxes(range=[-120, 120])
        fig.update_layout(
            title=title_text if title else '',
            title_font_size=font_size * 1.5,
            font_family=font,
            font_size=font_size,
            xaxis=dict(
                title=x_title if x_title else '',
                titlefont_size=font_size,
                tickfont_size=font_size
            ),
            yaxis=dict(
                title=y_title if y_title else '',
                titlefont_size=font_size,
                tickfont_size=font_size,
                tickformat='1%' if percents else '1'
            ),
            bargap=0.15,  # gap between bars of adjacent location coordinates.
            template=self.large_rockwell_template,
            width=width,
            height=height,
            barmode='relative'
        )
        if transparent:
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)')
        else:
            fig.update_layout(plot_bgcolor='rgb(255,255,255)')

        fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
        fig.update_yaxes(showline=False, linewidth=1, linecolor='black')
        fig.update_yaxes(showgrid=False, gridwidth=1, gridcolor='lightgrey', automargin=True)
        fig.update_xaxes(tickangle=0, automargin=True)
        return fig

    def stacked_bar_plot(self, column: str, first_column: str, second_column: str,
                         title: Optional[bool] = False, title_text: Optional[str] = None,
                         x_title: Optional[str] = None, y_title: Optional[str] = None,
                         width: int = 900, height: int = 550,
                         font_size: int = 20, font: str = 'Hevletica Neue',
                         transparent: bool = False, percents: bool = True, include_total: bool = False,
                         max_symb: int = 20):
        fig = go.Figure()
        if include_total:
            df = self.df
        else:
            df = self.df.iloc[:-1, :]
        x = list(df[column]).copy()
        x = self.capitalize_list(x)
        x = [split_string(string, max_symb) for string in x]
        fig.add_trace(go.Bar(
            name=first_column,
            x=x, y=[round(i, 2) for i in df[first_column]],
            marker_color=self.color_palette[-2],
            texttemplate='%{y}', textposition='outside', textfont_size=font_size
        ))
        fig.add_trace(go.Bar(
            name=second_column,
            x=x, y=[round(i, 2) for i in df[second_column]],
            marker_color='rgb(232,148,60)',
            texttemplate='%{y}', textposition='outside', textfont_size=font_size
        ))
        fig.update_layout(

            title=title_text if title else '',
            title_font_size=font_size * 1.5,
            font_family=font,
            font_size=font_size,
            xaxis=dict(
                title=x_title if x_title else '',
                titlefont_size=font_size,
                tickfont_size=font_size
            ),
            yaxis=dict(
                title=y_title if y_title else '',
                titlefont_size=font_size,
                tickfont_size=font_size,
                tickformat='1%' if percents else '1'
            ),
            bargap=0.15,  # gap between bars of adjacent location coordinates.
            template=self.large_rockwell_template,
            width=width,
            height=height,
            showlegend=True,
            legend=dict(font_size=font_size,
                        font_family=font,
                        orientation='h',
                        y=-0.35,
                        x=0.5,
                        xanchor='center',
                        yanchor='middle'),
        )
        if transparent:
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)')
        else:
            fig.update_layout(plot_bgcolor='rgb(255,255,255)')

        fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black')
        fig.update_yaxes(showgrid=False, gridwidth=1, gridcolor='lightgrey', automargin=True)
        fig.update_xaxes(tickangle=0, automargin=True)
        fig.update_layout(barmode='stack')
        return fig

    def plot_histogram(self, column: str,
                       title: Optional[bool] = False, title_text: Optional[str] = None,
                       x_title: Optional[str] = None, y_title: Optional[str] = None,
                       width: int = 900, height: int = 550,
                       font_size: int = 20, font: str = 'Hevletica Neue',
                       transparent: bool = False):
        fig = go.Figure(data=[go.Histogram(x=self.df[column], marker_color='rgb(222,46,37)')])
        fig.update_layout(
                title=title_text if title else '',
                title_font_size=font_size * 1.5,
                font_family=font,
                font_size=font_size,
                xaxis=dict(
                    title=x_title if x_title else '',
                    titlefont_size=font_size,
                    tickfont_size=font_size
                ),
                yaxis=dict(
                    title=y_title if y_title else '',
                    titlefont_size=font_size,
                    tickfont_size=font_size,
                    tickformat='1'
                ),
                template=self.large_rockwell_template,
                width=width,
                height=height
            )
        if transparent:
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)')
        else:
            fig.update_layout(plot_bgcolor='rgb(255,255,255)')

        fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black')
        fig.update_yaxes(showgrid=False, gridwidth=1, gridcolor='lightgrey', automargin=True)
        fig.update_xaxes(tickangle=0, automargin=True)
        return fig

    def plot_scatter_with_regression(self, first_column: str, second_column: str,
                                     title: Optional[bool] = False, title_text: Optional[str] = None,
                                     x_title: Optional[str] = None, y_title: Optional[str] = None,
                                     width: int = 900, height: int = 550,
                                     font_size: int = 20, font: str = 'Hevletica Neue',
                                     transparent: bool = False, marker_size: int = 10, marker_line_width: int = 2):
        df = self.df
        y = np.array([float(i) for i in df[first_column]])
        x = np.array([float(i) for i in df[second_column]])
        X = sm.add_constant(x)
        res = sm.OLS(y, X).fit()

        st, data, ss2 = summary_table(res, alpha=0.05)
        preds = pd.DataFrame.from_records(data, columns=[s.replace('\n', ' ') for s in ss2])
        preds['displ'] = x
        preds = preds.sort_values(by='displ')

        fig = go.Figure()
        p1 = go.Scatter(**{
            'mode': 'markers', 'marker_line_width': marker_line_width, 'marker_size': marker_size,
            'marker_color': 'rgb(222,46,37)',
            'x': x,
            'y': y,
            'name': 'Points'
        })
        p2 = go.Scatter({
            'mode': 'lines',
            'x': preds['displ'],
            'y': preds['Predicted Value'],
            'name': 'Regression',
            'line': {
                'color': 'rgb(215,116,102)'
            }
        })
        #Add a lower bound for the confidence interval, white
        p3 = go.Scatter({
            'mode': 'lines',
            'x': preds['displ'],
            'y': preds['Mean ci 95% low'],
            'name': 'Lower 95% CI',
            'showlegend': False,
            'line': {
                'color': 'white'
            }
        })
        # Upper bound for the confidence band, transparent but with fill
        p4 = go.Scatter( {
            'type': 'scatter',
            'mode': 'lines',
            'x': preds['displ'],
            'y': preds['Mean ci 95% upp'],
            'name': '95% CI',
            'fill': 'tonexty',
            'line': {
                'color': 'white'
            },
            'fillcolor': 'rgba(215,116,102, 0.3)'
        })
        fig.add_trace(p1)
        fig.add_trace(p2)
        fig.add_trace(p3)
        fig.add_trace(p4)
        fig.update_layout(
            font_family=font,
            title=title_text if title else '',
            xaxis=dict(
                title=x_title,
                titlefont_size=font_size,
                tickfont_size=font_size,
                tickformat='1%',
            ),
            yaxis=dict(
                title=y_title,
                titlefont_size=font_size,
                tickfont_size=font_size,
                tickformat='1%',
                title_standoff=width*0.01
            ),
            template=dict(
                layout=go.Layout(title_font=dict(family=font, size=font_size * 1.5))
            ),
            width=width,
            height=height,
            legend=dict(font=dict(family=font, size=font_size)),
            margin=dict(
                l=150,
                r=50,
                b=100,
                t=100,
                pad=4
            ),
        )

        if transparent:
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)')
        else:
            fig.update_layout(plot_bgcolor='rgb(255,255,255)')
        return fig


def split_string(string, max_symb):
    new_str_list = string.split(" ")
    whole_str = ""
    new_str = ""
    end = False
    ind = 0
    if len(new_str_list) == 1 and len(new_str_list[0]) > max_symb:
        return string
    if max([len(w) for w in new_str_list]) > max_symb:
        raise ValueError('number of symbols is too low. Increase it.')
    while not end:
        if len(new_str) + len(new_str_list[ind]) <= max_symb:
            if new_str == "":
                new_str += new_str_list[ind]
            else:
                new_str = new_str + " " + new_str_list[ind]
            ind += 1
            if ind == len(new_str_list):
                whole_str += new_str + '<br>'
                end = True
        else:
            whole_str += new_str + '<br>'
            new_str = ""
    return whole_str
