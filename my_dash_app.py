import pandas as pd
import numpy as np
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
from exploration import create_scatter_plot
import plotly.express as px

file = 'RedWine.csv'
df = pd.read_csv(file)

column_names = df.columns
labels = []

for name in column_names:
    label = {'label': f'{name}', 'value': f'{name}'}
    labels.append(label)

print(labels)

app = dash.Dash(__name__)

# Implement header, two dropdowns and slider for current values of X factor.
app.layout = html.Div(children = [html.H1("Wine Quality Dashboard", 
                                          style = {'textAlign': 'center', 
                                                   'color': '#503D36',
                                                   'font-size': 40}),
                                  html.Br(),
                                  html.Div(
                                      dcc.Dropdown(id = 'drop_x', options = labels, placeholder = 'X axis', searchable = True, value=labels[0]['value'])),
                                  html.Div(
                                      dcc.Dropdown(id = 'drop_y', options = labels, placeholder = 'Y axis', searchable = True, value=labels[1]['value'])),
                                  html.Br(),
                                  html.Div(
                                      dcc.Graph(id = 'top_graph')),
                                  html.Br(),
                                  html.Div(
                                      dcc.RangeSlider(id = 'slider_horizontal')),
                                  html.Br(),
                                  html.Div(
                                      dcc.Graph(id = 'bottom_graph'))
                                ]
                        )
@app.callback(
        Output(component_id = 'top_graph', component_property = 'figure'),
        [Input(component_id = 'drop_x', component_property = 'value'),
         Input(component_id = 'drop_y', component_property = 'value')],
)
def get_scatter_chart(entered_value_x, entered_value_y):
    df_entered = df
    fig = px.scatter(df_entered, x = entered_value_x, y = entered_value_y)
    return fig


if __name__ == '__main__':
    app.run_server(port=8080)