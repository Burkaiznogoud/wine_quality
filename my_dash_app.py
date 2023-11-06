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
    html.Div(
        dcc.Dropdown(   id = 'top_drop_horizontal', 
                        options = labels, 
                        placeholder = 'Choose factor',
                        value = labels[0]['label'], 
                        searchable = True)),
    html.Div(
        dcc.Dropdown(   id = 'top_drop_vertical', 
                        options = labels, 
                        placeholder = 'Choose factor',
                        value = labels[1]['label'], 
                        searchable = True)),
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
                        placeholder = 'Choose factor', 
                        searchable = True,
                        value = labels[0]['label']),
        dcc.RangeSlider(id = 'bottom_slider_horizontal',
                        min = 0,
                        max = 1,
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
                        placeholder = 'Choose factor', 
                        searchable = True,
                        value = labels[2]['label']),
        dcc.RangeSlider(id = 'bottom_slider_vertical',
                        min = 0,
                        max = 1,
                        value = [0, 1])])
            ]
        ),
    html.Br(),
    html.Div(
        dcc.Graph(      id = 'bottom_graph')
        )
])

# Top graph

@app.callback(
        Output(component_id = 'top_graph', component_property = 'figure'),
        [Input(component_id = 'top_drop_horizontal', component_property = 'value'),
         Input(component_id = 'top_drop_vertical', component_property = 'value')],
)
def get_scatter_chart(top_x, top_y):
    df_entered = df
    fig = px.scatter(df_entered, x = top_x, y = top_y)
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
         Input(component_id = 'bottom_drop_vertical', component_property = 'value')]
)
def get_scatter_chart_sliders(bottom_x, bottom_y):
    df_entered = df
    fig = px.scatter(df_entered, x = bottom_x, y = bottom_y)
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
         Output(component_id = 'bottom_slider_horizontal', component_property = 'value')],
         Input(component_id = 'bottom_drop_horizontal', component_property = 'value')
)
def update_slider_horizontal(bottom_drop_horizontal):
    df_entered = df
    min_value = math.floor(df[bottom_drop_horizontal].min())
    max_value = math.ceil(df[bottom_drop_horizontal].max())
    value_range = [min_value, max_value]
    return min_value, max_value, value_range

@app.callback(
        [Output(component_id = 'bottom_slider_vertical', component_property = 'min'),
         Output(component_id = 'bottom_slider_vertical', component_property = 'max'),
         Output(component_id = 'bottom_slider_vertical', component_property = 'value')],
         Input(component_id = 'bottom_drop_vertical', component_property = 'value')
)
def update_slider_vertical(bottom_drop_vertical):
    df_entered = df
    min_value = math.floor(df[bottom_drop_vertical].min())
    max_value = math.ceil(df[bottom_drop_vertical].max())
    value_range = [min_value, max_value]
    return min_value, max_value, value_range


if __name__ == '__main__':
    app.run_server(port=8080)