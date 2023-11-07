import pandas as pd
import numpy as np
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
from exploration import create_scatter_plot
import plotly.express as px
import math


file = 'RedWine.csv'
df = pd.read_csv(file)
pd.options.display.float_format = '{:.2f}'.format

column_names = df.columns
labels = []

def create_marks(min, max, rng):
    
    match rng:
        case rng if rng >= 100:
            step = math.ceil(rng / 100) * 10
            if min < 20:
                min = 0
                max = (step * 10) + min + 1
            else:
                min = math.floor(min / 10) * 100
                max = (step * 10) + min + 1
            return {i: str(i) for i in range(min, max, step)}
        case rng if rng >= 10 and rng < 100:
            step = math.ceil(rng / 10)
            if step < 5:
                min = 0
                max = (step * 10) + 1
            else:
                min = 5
                max = (step * 10) + min + 1
            return {i: str(i) for i in range(min, max, step)}
        case rng if rng < 10:
            # because float object cannot be interpreted as integer and steps need to be divided by the same
            m = 1000000
            step = rng * m / 10
            min = min * m - step
            max = max * m + step + 1
            return {(str(i/m)): str(i/m) for i in range(min, max , step)}


for name in column_names:
    label = {'label': f'{name}', 'value': f'{name}'}
    labels.append(label)

print(labels)

app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H1("Wine Quality Dashboard", 
                        style = {
                        'textAlign': 'center', 
                        'color': '#503D36',
                        'font-size': 40}),
    html.Br(),
    html.Div(children=[
        html.P("X axis: ",
                        style = {
                        'textAlign': 'center', 
                        'color': '#503D36',
                        'font-size': 20,
                        'font-weight': 'bold'}),
        html.Div(
            dcc.Dropdown(   id = 'top_drop_horizontal', 
                            options = labels,
                            value = labels[0]['label'], 
                            searchable = True)),
        html.P("Y axis: ",
                        style = {
                        'textAlign': 'center', 
                        'color': '#503D36',
                        'font-size': 20,
                        'font-weight': 'bold'}),
        html.Div(
            dcc.Dropdown(   id = 'top_drop_vertical', 
                            options = labels,
                            value = labels[1]['label'], 
                            searchable = True)
                )
            ]
        ),
    html.Br(),
    html.Div(
        dcc.Graph(      id = 'top_graph')),
    html.Br(),
    html.Div(children=[
        html.P("X axis: ",
                        style = {
                        'textAlign': 'center', 
                        'color': '#503D36',
                        'font-size': 20,
                        'font-weight': 'bold'}),
        html.Div(
        [dcc.Dropdown(  id = 'bottom_drop_horizontal', 
                        options = labels, 
                        searchable = True,
                        value = labels[0]['label']),
        dcc.RangeSlider(id = 'bottom_slider_horizontal',
                        min = 0,
                        max = 1,
                        marks = 1,
                        value = [0, 1])])
            ]
        ),
    html.Div(children=[
        html.P("Y axis: ",
                        style = {
                        'textAlign': 'center', 
                        'color': '#503D36',
                        'font-size': 20,
                        'font-weight': 'bold'}),
        html.Div(
        [dcc.Dropdown(  id = 'bottom_drop_vertical', 
                        options = labels,
                        searchable = True,
                        value = labels[2]['label']),
        dcc.RangeSlider(id = 'bottom_slider_vertical',
                        min = 0,
                        max = 1,
                        marks = 1,
                        value = [0, 1])])
            ]
        ),
    html.Br(),
    html.Div(
        dcc.Graph(      id = 'bottom_graph')
        )
    ],
    style={'backgroundColor': '#ff8566'}
)

# Top graph

@app.callback(
        Output(component_id = 'top_graph', component_property = 'figure'),
        [Input(component_id = 'top_drop_horizontal', component_property = 'value'),
         Input(component_id = 'top_drop_vertical', component_property = 'value')],
)
def get_scatter_chart(top_x, top_y):

    _df = df
    fig = px.scatter(_df, x = top_x, y = top_y)
    return fig


@app.callback(
    [Output(component_id = 'top_drop_horizontal', component_property = 'options'),
    Output(component_id = 'top_drop_vertical', component_property = 'options')],
    [Input(component_id = 'top_drop_horizontal', component_property = 'value'),
    Input(component_id = 'top_drop_vertical', component_property = 'value')]
)
def update_dropdown_options(selected_horizontal, selected_vertical):

    available_options = df.columns.tolist()
    disabled_horizontal = [{'label': option, 'value': option, 'disabled': True} if option == selected_vertical 
                            else {'label': option, 'value': option}
                            for option in available_options]
    disabled_vertical = [{'label': option, 'value': option, 'disabled': True} if option == selected_horizontal 
                            else {'label': option, 'value': option}
                            for option in available_options]

    return disabled_horizontal, disabled_vertical


#### Bottom graph ####


@app.callback(
        Output(component_id = 'bottom_graph', component_property = 'figure'),
        [Input(component_id = 'bottom_drop_horizontal', component_property = 'value'),
         Input(component_id = 'bottom_drop_vertical', component_property = 'value'),
         Input(component_id = 'bottom_slider_horizontal', component_property = 'value'),
         Input(component_id = 'bottom_slider_vertical', component_property = 'value')]
)
def get_scatter_chart_sliders(bottom_x, bottom_y, x_range, y_range):
    _df = df
    x_rng = _df[(_df[bottom_x] >= x_range[0]) & (_df[bottom_x] <= x_range[1])]
    y_rng = _df[(_df[bottom_y] >= y_range[0]) & (_df[bottom_y] <= y_range[1])]
    _df = pd.DataFrame.merge(x_rng, y_rng)
    fig = px.scatter(_df, x = bottom_x, y = bottom_y)
    return fig


@app.callback(
    [Output(component_id = 'bottom_drop_horizontal', component_property = 'options'),
    Output(component_id = 'bottom_drop_vertical', component_property = 'options')],
    [Input(component_id = 'bottom_drop_horizontal', component_property = 'value'),
    Input(component_id = 'bottom_drop_vertical', component_property = 'value')]
)
def update_dropdown_options(selected_horizontal, selected_vertical):

    available_options = df.columns.tolist()
    disabled_horizontal = [{'label': option, 'value': option, 'disabled': True} if option == selected_vertical 
                            else {'label': option, 'value': option}
                            for option in available_options]
    disabled_vertical = [{'label': option, 'value': option, 'disabled': True} if option == selected_horizontal 
                            else {'label': option, 'value': option}
                            for option in available_options]

    return disabled_horizontal, disabled_vertical


@app.callback(
        [Output(component_id = 'bottom_slider_horizontal', component_property = 'min'),
         Output(component_id = 'bottom_slider_horizontal', component_property = 'max'),
         Output(component_id = 'bottom_slider_horizontal', component_property = 'value'),
         Output(component_id = 'bottom_slider_horizontal', component_property = 'marks')],
         Input(component_id = 'bottom_drop_horizontal', component_property = 'value')
)
def update_slider_horizontal(bottom_drop_horizontal):
    min_marks = df[bottom_drop_horizontal].min()
    max_marks = df[bottom_drop_horizontal].max()
    min_value = math.floor(min_marks)
    max_value = math.ceil(max_marks)
    value_range = [min_value, max_value]
    rng = max_value - min_value
    marks = create_marks(min=min_marks, max=max_marks, rng=rng)
    return min_value, max_value, value_range, marks


@app.callback(
        [Output(component_id = 'bottom_slider_vertical', component_property = 'min'),
         Output(component_id = 'bottom_slider_vertical', component_property = 'max'),
         Output(component_id = 'bottom_slider_vertical', component_property = 'value'),
         Output(component_id = 'bottom_slider_vertical', component_property = 'marks')],
         Input(component_id = 'bottom_drop_vertical', component_property = 'value')
)
def update_slider_vertical(bottom_drop_vertical):

    min_value = math.floor(df[bottom_drop_vertical].min())
    max_value = math.ceil(df[bottom_drop_vertical].max())
    value_range = [min_value, max_value]
    rng = max_value - min_value
    marks = create_marks(min=min_value, max=max_value, rng=rng)
    print(marks)
    return min_value, max_value, value_range, marks


if __name__ == '__main__':
    app.run_server(port=8080)