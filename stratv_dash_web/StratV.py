
#from __future__ import print_function
#from google.auth.transport.requests import Request
#from google_auth_oauthlib.flow import InstalledAppFlow
#from httplib2 import Http
#from apiclient.discovery import build
##from oauth2client import file, client, tools
#from oauth2client.service_account import ServiceAccountCredentials
#import gspread
import base64
import datetime
import io
#from base64 import *
#from urllib.parse import quote as urlquote
from flask import Flask, send_from_directory
from flask import Flask, render_template, request, redirect, url_for
import itertools
import dash
import dash_auth
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import chart_studio.plotly as py
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from dash.dependencies import Output, Input
from dash.exceptions import PreventUpdate
from datetime import datetime
import os
import dash_table
from pprint import pprint
import csv
import math
from decimal import*
import numpy as np
from numpy import*
import random
import matplotlib
from scipy.optimize import*
import matplotlib.pyplot as plt
from pandastable.core import Table
from pandastable.data import TableModel
from datetime import datetime
import os
import sys
#from googleapiclient import discovery
from scipy.optimize import *

import pyqtgraph
from PyQt5 import QtWidgets, QtCore
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
# import sys  # We need sys so that we can pass argv to QApplication
# import os
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

server = Flask(__name__)
app = dash.Dash(server=server)
# scope = ['https://spreadsheets.google.com/feeds',
#          'https://www.googleapis.com/auth/drive']
# creds = None
# creds = ServiceAccountCredentials.from_json_keyfile_name(
#     'transactionmanagerdash-3eedffe0ea0a.json', scope)
# client = gspread.authorize(creds)
# service = discovery.build('sheets', 'v4', credentials=creds)
# spreadsheet_id = '1g-zsD21zRQpp7goFkxtmn-48lZxZXVJf23CeZzWmWU0'
# sheet = service.spreadsheets()
# Sheet = client.open("DataBase").sheet1
# data = pd.DataFrame(Sheet.get_all_records()).reset_index()
# SCOPES = ['https://www.googleapis.com/auth/documents.readonly']
# DOCUMENT_ID = '1TN45bAT3dF5h2VPDp1En0Q4sye99o6B9tNbGseBqqTY'


# def get_google_sheet(spreadsheet_id, range_name):
#     """ Retrieve sheet data using OAuth credentials and Google Python API. """
#     service = build('sheets', 'v4', http=creds.authorize(Http()))
#     gsheet = sheet.values().get(spreadsheetId=spreadsheet_id, range=range_name).execute()
#     return gsheet


# def gsheet2df(gsheet):
#     """ Converts Google sheet data to a Pandas DataFrame.
#     Note: This script assumes that your data contains a header file on the first row!

#     Also note that the Google API returns 'none' from empty cells - in order for the code
#     below to work, you'll need to make sure your sheet doesn't contain empty cells,
#     or update the code to account for such instances.

#     """
#     header = gsheet.get('values', [])[0]
#     values = gsheet.get('values', [])[1:]
#     if not values:
#         print('No data found.')
#     else:
#         all_data = []
#         for col_id, col_name in enumerate(header):
#             column_data = []
#             for row in values:
#                 column_data.append(row[col_id])
#             ds = pd.Series(data=column_data, name=col_name)
#             all_data.append(ds)
#         df = pd.concat(all_data, axis=1)
#         return df


# RANGE_NAME = 'DataBase'
# gsheet = get_google_sheet(spreadsheet_id, RANGE_NAME)
# df = gsheet2df(gsheet)
# df["Date"] = pd.to_datetime(data["Date"])
# analysis_data = df.set_index('Date')
# last_date = df['Date'].iloc[-1].strftime("%B %d, %Y")

# now = datetime.now()
# date_time = now.strftime("%B %d, %Y")
# date_and_time = now.strftime("%B %d, %Y | %H:%M:%S")
external_stylesheets = [
    {
        "href": "https://fonts.googleapis.com/css2?"
        "family=Lato:wght@400;700&display=swap",
        "rel": "stylesheet",
    },
]
# VALID_USERNAME_PASSWORD_PAIRS = {
#     'gas': 'pilotxlab'
# }

app = dash.Dash(__name__,
                external_stylesheets=external_stylesheets,
                meta_tags=[
                    {"name": "viewport", "content": "width=device-width, initial-scale=1"}
                ],
                )
server = app.server

# auth = dash_auth.BasicAuth(
#     app,
#     VALID_USERNAME_PASSWORD_PAIRS
# )

bed_data = pd.read_csv('Permeability_Porosity_distribution_data.csv')
RPERM_data = pd.read_csv('Oil_Water_Relative_Permeability_data.csv')
# ARRANGING THE DATA IN ORDER OF DECREASING PERMEABILITY.
bed_data_sort = bed_data.sort_values(by='PERMEABILITY', ascending=False)
PORO = np.array(bed_data_sort['POROSITY'])
permeability_array = np.array(bed_data_sort['PERMEABILITY'])
h = np.array(bed_data_sort['THICKNESS'])
SW = np.array(RPERM_data['SW'])
KRW = np.array(RPERM_data['KRW'])
KRO = np.array(RPERM_data['KRO'])
#KRW_1_SOR = np.interp(1-SOR, SW, KRW)
#KRO_SWI = np.interp(SWI, SW, KRO)
# EXTRACTING THE SORTED LAYER COLUMN
Layer_column = bed_data_sort['LAYER'].to_numpy()
Layer_table =  pd.DataFrame(Layer_column, columns = ['Layers'])
#==========================================================================================================================
#This code calculates the permeability ratio, ki/kn
List_of_permeability_ratio = []
for permeability_index in range(len(permeability_array)):
    List_of_permeability_ratio_subset = [][:-permeability_index]
    for index,permeability in enumerate(permeability_array):
        if permeability_index <= index:
            permaebility_ratio = permeability/permeability_array[permeability_index]
            List_of_permeability_ratio_subset.append(permaebility_ratio)
    List_of_permeability_ratio.append(List_of_permeability_ratio_subset)

List_of_permeability_ratio_DataTable = pd.DataFrame(List_of_permeability_ratio).transpose()

Average_porosity = '%.2f' % np.mean(bed_data_sort.POROSITY)

app.title = "StratV"
app.layout = html.Div(id="tm", children=[
    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='WATERFLOOD METHODS', value='tab-1'),
    ]),
    html.Div(id='tabs-content')
])


@app.callback(
    Output('tabs-content', 'children'),
    [
        Input('tabs', 'value'),
    ],
)
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.Div(id='container1', children=[
                html.Aside(id="inputs",
                        children=[
                            html.H1('INPUTS'),
                            html.H5('(Field Units)'),
                            dcc.Input(id="number-of-points",
                                    placeholder='Number of Points',
                                    type='number',
                                    value=10
                                    ),
                            dcc.Input(id="length-of-bed",
                                    placeholder='Length of Bed',
                                    type='number',
                                    value=2896
                                    ),
                            dcc.Input(id="bed-width",
                                    placeholder='Width of Bed',
                                    type='number',
                                    value=2000
                                    ),
                            dcc.Input(id="porosity",
                                    placeholder='Average Porosity',
                                    type='number',
                                    value=0.25
                                    ),
                            dcc.Input(id="VISO",
                                    placeholder='Oil Viscosity',
                                    type='number',
                                    value=3.6
                                    ),
                            dcc.Input(id="VISW",
                                    placeholder='Water Viscosity',
                                    type='number',
                                    value=0.95
                                    ),
                            dcc.Input(id="OFVF",
                                    placeholder='Oil Formation Volume Factor',
                                    type='number',
                                    value=1.11
                                    ),
                            dcc.Input(id="WFVF",
                                    placeholder='Water Formation Volume Factor',
                                    type='number',
                                    value=1.01
                                    ),
                            dcc.Input(id="SWI",
                                    placeholder='Initial Water Saturation',
                                    type='number',
                                    value=0.2
                                    ),
                            dcc.Input(id="SGI",
                                    placeholder='Initial Gas Saturation',
                                    type='number',
                                    value=0.16
                                    ),
                            dcc.Input(id="SOI",
                                    placeholder='Initial Oil Saturation',
                                    type='number',
                                    value=0.64
                                    ),
                            dcc.Input(id="SOR",
                                    placeholder='Residual Oil Saturation',
                                    type='number',
                                    value=0.35
                                    ),
                            dcc.Input(id="CIR",
                                    placeholder='Constant Injection Rate',
                                    type='number',
                                    value=1800
                                    ),
                            dcc.Input(id="injection-pressure",
                                    placeholder='Injection Pressure',
                                    type='number',
                                    value=700
                                    ),
                            dcc.Input(id="RGSUZ",
                                    placeholder='Unswept Zone Residual Gas Saturation',
                                    type='number',
                                    value=0.06
                                    ),
                            dcc.Input(id="RGSSZ",
                                    placeholder='Swept Zone Residual Gas Saturation',
                                    type='number',
                                    value=0.02
                                    )
                            # html.Button('View Fractional Flow Plot', id='fractionalflow-button', n_clicks=0),
                            # html.Div(id='button-click', children='')

                        ]
                        )
                    ]),
                    html.Div(id='container2', children=[
                        html.Aside(id='method', children=[
                            html.H1('WATERFLOOD METHODS'),
                            dcc.RadioItems(id="method-options",
                                options=[
                                    {'label': 'Dykstra-Parson','value':'D'},
                                    {'label': 'Reznik et al.', 'value': 'Rz'},
                                    {'label': 'Roberts', 'value': 'Ro'}
                                ],
                                value=''
                            ),
                            html.Div(id='method-output',
                                children=''    
                            ),
                            html.H5(children="Chart type"),
                            dcc.Dropdown(id="chart-type",className="inputs",
                                options=[
                                    {'label': 'line',
                                        'value': 'line'},
                                    {'label': 'bar',
                                        'value': 'bar'},
                                    {'label': 'scatter',
                                        'value': 'scatter'},
                                    {'label': 'pie',
                                        'value': 'pie'},
                                    {'label': 'sunburst',
                                        'value': 'sunburst'},
                                    {'label': 'sankey',
                                        'value': 'sankey'},
                                    {'label': 'pointcloud',
                                        'value': 'pointcloud'},
                                    {'label': 'treemap',
                                        'value': 'treemap'},
                                    {'label': 'table',
                                        'value': 'table'},
                                    {'label': 'scattergl',
                                        'value': 'scattergl'}
                                    
                                ],
                                value='',
                                placeholder="Chart type",
                                clearable=True
                            ),
                            html.Div(id='x-axis',
                                children=''    
                            ),
                        ]

                        )
                    ]),
                     html.H2('General Calculations'),
                     html.Div(id='general-output',
                         children=[
                             html.Div(className='general-calculation',
                                 children=[
                                     html.Div(className='P',children='Average Porosity'),
                                     html.Div(className='V',id='average-porosity',children=str(Average_porosity)+' cp')
                                 ]
                             ),
                             html.Div(className='general-calculation',
                                 children=[
                                     html.Div(className='P',children='Relative Permeability at 1-SOR'),
                                     html.Div(className='V',id='relative-perm-at-1-SOR',children=0.00)
                                 ]
                             ),
                             html.Div(className='general-calculation',
                                 children=[
                                     html.Div(className='P',children='Relative Permeability at SWI'),
                                     html.Div(className='V',id='relative-perm-at-SWI',children=0.00)
                                 ]
                             ),
                             html.Div(className='general-calculation',
                                 children=[
                                     html.Div(className='P',children='Gross Rock Volume'),
                                     html.Div(className='V',id='gross-rock-volume',children=0.00)
                                 ]
                             ),
                             html.Div(className='general-calculation',
                                 children=[
                                     html.Div(className='P',children='Displacement Efficiency'),
                                     html.Div(className='V',id='displacement-efficiency',children=0.00)
                                 ]
                             ),
                             html.Div(className='general-calculation',
                                 children=[
                                     html.Div(className='P',children='Mobility Ratio'),
                                     html.Div(className='V',id='mobility-ratio',children=0.00)
                                 ]
                             ),
                             html.Div(className='general-calculation',
                                 children=[
                                     html.Div(className='P',children='Average Sweep Efficiency at Breakthrough'),
                                     html.Div(className='V',id='average-sweep-efficiency-at-breakthrough',children=0.00)
                                 ]
                             ),
                             html.Div(className='general-calculation',
                                 children=[
                                     html.Div(className='P',children='Area of Reservoir Bed'),
                                     html.Div(className='V',id='area-of-reservoir',children=0.00)
                                 ]
                             ),
                             html.Div(className='general-calculation',
                                 children=[
                                     html.Div(className='P',children='Area Sweep Efficiency'),
                                     html.Div(className='V',id='area-sweep-efficiency',children=0.00)
                                 ]
                             ),
                         ]
                     ),
                    html.Div(className='table',children=[
                        html.H2('Dykstra-Parsons Tabular Result'),
                        html.Div(id='table-display',
                            children=''
                        )
                    ]
                    ),
                    html.Div(className='table',children=[
                        html.H2('Reznik et al. Tabular Result'),
                        html.Div(id='Reznik-discrete-point-division',children=0.00),
                        html.Div(id='Reznik-table-display',
                            children=''
                        )
                    ]
                    ),
                     html.Div(className='table',children=[
                        html.H2('Roberts Tabular Result'),
                        html.Div(id='Robert-table-display',
                            children=''
                        )
                    ]
                    ),
                    html.Div(id='graph',
                        children=[
                        dcc.Graph(className="chart",
                            id="Dykstra-Parsons-Chart", config={"displayModeBar": True},
                            style={'border': 'solid rgb(19, 18, 18)'}
                        ),
                        dcc.Graph(className="chart",
                            id="Reznik-Chart", config={"displayModeBar": True},
                            style={'border': 'solid rgb(19, 18, 18)', 'backgroundColor':'black'}
                        ),
                        dcc.Graph(className="chart",
                            id="Robert-Chart", config={"displayModeBar": True},
                            style={'border': 'solid rgb(19, 18, 18)'}
                        )
                        ]
                    )
            ]
        )
      



@app.callback(
    Output("relative-perm-at-1-SOR", "children"),
    [
        Input("SOR", "value")
    ]
)
def relative_perm_1_SOR(SOR):
    KRW_1_SOR = '%.3f' % np.interp(1 - float(SOR), SW, KRW)
    return '{}'.format(KRW_1_SOR)

@app.callback(
    Output("relative-perm-at-SWI", "children"),
    [
        Input("SWI", "value")
    ]
)
def relative_perm_SWI(SWI):
    KRO_SWI = '%.3f' % np.interp(SWI, SW, KRO)
    return '{}'.format(KRO_SWI)


@app.callback(
    Output("mobility-ratio", "children"),
    [
        Input("SOR", "value"),
        Input("SWI", "value"),
        Input("VISO", "value"),
        Input("VISW", "value"),
    ]
)
def mobility_ratio(SOR,SWI,VISO,VISW):
    KRW_1_SOR = np.interp(1-SOR, SW, KRW)
    KRO_SWI = np.interp(SWI, SW, KRO)
    Mobility_Ratio = '%.3f' % (KRW_1_SOR * VISO / (KRO_SWI * VISW))
    return '{}'.format(Mobility_Ratio)

@app.callback(
    Output("average-sweep-efficiency-at-breakthrough", "children"),
    [
        Input("SOR", "value"),
        Input("SWI", "value"),
        Input("VISO", "value"),
        Input("VISW", "value"),
    ]
)
def areal_sweep_efficiency_at_breakthrough(SOR,SWI,VISO,VISW):
    KRW_1_SOR = np.interp(1-SOR, SW, KRW)
    KRO_SWI = np.interp(SWI, SW, KRO)
    Mobility_Ratio = KRW_1_SOR*VISO/(KRO_SWI*VISW)
    Areal_sweep_efficiency_at_breakthrough =  '%.3f' % (0.54602036+(0.03170817/Mobility_Ratio)+(0.30222997/math.exp(Mobility_Ratio)-0.0050969*Mobility_Ratio))
    return '{}'.format(Areal_sweep_efficiency_at_breakthrough)

@app.callback(
    Output("area-of-reservoir", "children"),
    [
        Input("length-of-bed", "value"),
        Input("bed-width", "value"),
    ]
)
def area_acres(Length_of_bed_ft,width_of_bed_ft):
    Area_acres =  '%.3f' % (Length_of_bed_ft*width_of_bed_ft/43560)
    return '{} acres'.format(Area_acres)

@app.callback(
    Output("gross-rock-volume", "children"),
    [
        Input("length-of-bed", "value"),
        Input("bed-width", "value"),
    ]
)
def gross_rock_volume(Length_of_bed_ft,width_of_bed_ft):
    Area_acres = Length_of_bed_ft*width_of_bed_ft/43560
    Gross_rock_volume_acre_ft =  '%.3f' % (Area_acres*bed_data_sort.THICKNESS.sum())
    return '{} acres-ft'.format(Gross_rock_volume_acre_ft)
@app.callback(
    Output("displacement-efficiency", "children"),
    [
        Input("SOR", "value"),
        Input("SWI", "value"),
        Input("SGI", "value"),
    ]
)
def displacement_efficiency(SOR,SWI,SGI):
    Displacement_efficiency =  '%.3f' % ((1-SWI-SGI-SOR)/(1-SWI-SGI))
    return '{}'.format(Displacement_efficiency)

@app.callback(
    Output("area-sweep-efficiency", "children"),
    [
        Input("SOR", "value"),
        Input("SWI", "value"),
        Input("SGI", "value"),
        Input("VISO", "value"),
        Input("VISW", "value"),
    ]
)
def areal_sweep_efficiency(SOR,SWI,SGI,VISO,VISW):
    KRW_1_SOR = np.interp(1-SOR, SW, KRW)
    KRO_SWI = np.interp(SWI, SW, KRO)
    Mobility_Ratio = KRW_1_SOR*VISO/(KRO_SWI*VISW)
    Areal_sweep_efficiency_at_breakthrough = 0.54602036+(0.03170817/Mobility_Ratio)+(0.30222997/math.exp(Mobility_Ratio)-0.0050969*Mobility_Ratio)
    Displacement_efficiency = (1-SWI-SGI-SOR)/(1-SWI-SGI)
    Areal_sweep_efficiency =  '%.3f' % (Areal_sweep_efficiency_at_breakthrough+0.2749*np.log((1/Displacement_efficiency)))
    return '{}'.format(Areal_sweep_efficiency)


@app.callback(
    [Output("method-output", "children"),Output("x-axis", "children")],
    [
        Input("method-options", "value")
    ]
)




def choose_method(method):
    suppress_callback_exceptions=True
    #suppress_callback_exceptions=True
    if method=='D':
        method_table_list=html.Div(className='checklist_container',children=[dcc.Checklist(id='Dykstra-checklist',
                options=[
                    {'label': 'Bed Layer', 'value': "Layer_table"},
                    {'label': 'Oil Mobility', 'value': "Oil_Mobility_table"},
                    {'label': 'Water Mobility', 'value': "Water_Mobility_table"},
                    {'label': 'Water Flowrate per Bed', 'value': "Water_Flowrate_per_bed_table"},
                    {'label': 'Vertical Coverage', 'value': "coverage_table"},
                    {'label': 'Water Oil Ratio', 'value': "WOR_table"},
                    {'label': 'Cumulative Oil Recovery', 'value': "Cumulative_oil_recovery"},
                    {'label': 'Water Volume to Fillup Gas Space', 'value': "Water_volume_to_fillup_gas_space_table"},
                    {'label': 'Producing Water Oil Ratio', 'value': "Producing_water_oil_ratio"},
                    {'label': 'Cumulative Water Produced', 'value': "Cumulative_water_produced"},
                    {'label': 'Cumulative Water Injectied', 'value': "Cumulative_water_injected_table"},
                    {'label': 'Time (days)', 'value': "Time_days_table"},
                    {'label': 'Time (years)', 'value': "Time_years_table"},
                    {'label': 'Water Flowrate', 'value': "Water_Flowrate_table"},
                    {'label': 'Oil Flowrate', 'value': "Oil_Flowrate_table"},
                    {'label': 'Oil Flowrate per Bed', 'value': "Oil_Flowrate_per_bed_table"},
                    {'label':'Flood Front Location', 'value':"Front_Location_list_DataTable"}

                ],
                value=['Layer_table']
            )])
        x_axis_options=html.Div(className='checklist_container',children=[html.H5(children="x-axis"),dcc.Dropdown(id='Dykstra-xaxis_dropdown',
            options=[
                {'label': 'Bed Layer', 'value': "Layer_table"},
                {'label': 'Oil Mobility', 'value': "Oil_Mobility_table"},
                {'label': 'Water Mobility', 'value': "Water_Mobility_table"},
                {'label': 'Water Flowrate per Bed', 'value': "Water_Flowrate_per_bed_table"},
                {'label': 'Vertical Coverage', 'value': "coverage_table"},
                {'label': 'Water Oil Ratio', 'value': "WOR_table"},
                {'label': 'Cumulative Oil Recovery', 'value': "Cumulative_oil_recovery"},
                {'label': 'Water Volume to Fillup Gas Space', 'value': "Water_volume_to_fillup_gas_space_table"},
                {'label': 'Producing Water Oil Ratio', 'value': "Producing_water_oil_ratio"},
                {'label': 'Cumulative Water Produced', 'value': "Cumulative_water_produced"},
                {'label': 'Cumulative Water Injectied', 'value': "Cumulative_water_injected_table"},
                {'label': 'Time (days)', 'value': "Time_days_table"},
                {'label': 'Time (years)', 'value': "Time_years_table"},
                {'label': 'Water Flowrate', 'value': "Water_Flowrate_table"},
                {'label': 'Oil Flowrate', 'value': "Oil_Flowrate_table"}
            ],
            value='Layer_table',
            placeholder="choose x-axis",
            clearable=True
        )])
    if method == 'Rz':
        method_table_list=html.Div(className='checklist_container',children=[dcc.Checklist(id='Reznik-checklist',
            options=[
                {'label': 'Bed Layer', 'value': "Layer_table1"},
                {'label': 'Breakthrough Time', 'value': "breakthrough_time_table"},
                {'label': 'Flood Front of Last Bed at Breakthrough', 'value': "Flood_front_position_of_bed_n_j"},
                {'label': 'Ultimate Recoverable Oil', 'value': "Ultimate_recoverable_oil_per_bed_table"},
                {'label': 'Flood Front of Beds at Breakthrough', 'value': "Front_position_of_other_beds_at_breakthrough_table"},
                {'label': 'Flood Front of Last Bed', 'value': "flood_front_of_last_bed_table"},
                {'label': 'Real Time', 'value': "Real_time_CIP_table"},
                {'label': 'Dynamic Bed', 'value': "dynamic_bed_table"},
                {'label': 'Total Water Flowrate Before Breakthrough of Dynamic Bed', 'value': "sum_water_flowrate_before_breakthrough_of_dynamic_bed_table"},
                {'label': 'Total Oil Flowrate Before Breakthrough of Dynamic Bed', 'value': "sum_oil_flowrate_before_breakthrough_of_dynamic_bed_table"},
                {'label': 'Instantaneaous Produced Water Oil Ratio', 'value': "Instantaneous_producing_Water_Oil_Ratio_before_breakthrough_of_dynamic_bed_table"},
                {'label': 'Instantaneaous Produced Water Cut', 'value': "Instantaneous_producing_Water_cut_table"},
                {'label': 'Cumulative Oil Recovery', 'value': "cumulative_oil_recovered_at_time_t_table"},
                {'label': 'Cumulative Oil Recovery From All Beds', 'value': "Cumulative_oil_recovered_from_all_beds_table"},
                {'label': 'Vertical Coverage', 'value': "Vertical_coverage_at_time_t_table"},
                {'label': 'Cumulative Water Oil Ratio CIR', 'value': "Cumumlative_water_oil_ratio_for_CIR_table"},
                {'label': 'Cumulative Water Oil Ratio CIP', 'value': "Cumumlative_water_oil_ratio_for_CIP_table"},
                {'label': 'Flood Front Location of Beds', 'value': "Flood_front_location_of_other_beds_table"},
                {'label': 'Flood Front Location of Beds Beyond Breakthrough', 'value': "Flood_front_location_of_other_beds_beyond_breakthrough_table"},
                {'label': 'Property Time', 'value': "Property_time_table"},
                {'label': 'Average Mobility', 'value': "average_mobility_at_time_t_table"},
                {'label': 'Superficial Velocity', 'value': "Superficial_filter_velocity_table"},
                {'label': 'Actual Linear Velocity', 'value': "actual_linear_velocity_table"},
                {'label': 'Instantaneous Volumetric Water Flowrate', 'value': "instantaneous_volumetric_flowrate_of_water_table"},
                {'label': 'Instantaneous Volumetric Oil Flowrate', 'value': "instantaneous_volumetric_flowrate_of_oil_table"},
                {'label':'Cumulative Water Injected', 'value':"cumulative_water_injected_table"}

            ],
            value=['Layer_table1']
        )])
        x_axis_options=html.Div(className='checklist_container',children=[html.H5(children="x-axis"),dcc.Dropdown(id='Reznik-xaxis-dropdown',
            options=[
                {'label': 'Bed Layer', 'value': "Layer_table1"},
                {'label': 'Breakthrough Time', 'value': "breakthrough_time_table"},
                {'label': 'Flood Front of Last Bed at Breakthrough', 'value': "Flood_front_position_of_bed_n_j"},
                {'label': 'Ultimate Recoverable Oil', 'value': "Ultimate_recoverable_oil_per_bed_table"},
                #{'label': 'Flood Front of Beds at Breakthrough', 'value': "Front_position_of_other_beds_at_breakthrough_table"},
                {'label': 'Flood Front of Last Bed', 'value': "flood_front_of_last_bed_table"},
                {'label': 'Real Time', 'value': "Real_time_CIP_table"},
                {'label': 'Dynamic Bed', 'value': "dynamic_bed_table"},
                {'label': 'Total Water Flowrate Before Breakthrough of Dynamic Bed', 'value': "sum_water_flowrate_before_breakthrough_of_dynamic_bed_table"},
                {'label': 'Total Oil Flowrate Before Breakthrough of Dynamic Bed', 'value': "sum_oil_flowrate_before_breakthrough_of_dynamic_bed_table"},
                {'label': 'Instantaneaous Produced Water Oil Ratio', 'value': "Instantaneous_producing_Water_Oil_Ratio_before_breakthrough_of_dynamic_bed_table"},
                {'label': 'Instantaneaous Produced Water Cut', 'value': "Instantaneous_producing_Water_cut_table"},
                {'label': 'Cumulative Oil Recovery', 'value': "cumulative_oil_recovered_at_time_t_table"},
                {'label': 'Cumulative Oil Recovery From All Beds', 'value': "Cumulative_oil_recovered_from_all_beds_table"},
                {'label': 'Vertical Coverage', 'value': "Vertical_coverage_at_time_t_table"},
                {'label': 'Cumulative Water Oil Ratio CIR', 'value': "Cumumlative_water_oil_ratio_for_CIR_table"},
                {'label': 'Cumulative Water Oil Ratio CIP', 'value': "Cumumlative_water_oil_ratio_for_CIP_table"}
                # {'label': 'Flood Front Location of Beds', 'value': "Flood_front_location_of_other_beds_table"},
                # {'label': 'Flood Front Location of Beds Beyond Breakthrough', 'value': "Flood_front_location_of_other_beds_beyond_breakthrough_table"},
                # {'label': 'Property Time', 'value': "Property_time_table"},
                # {'label': 'Average Mobility', 'value': "average_mobility_at_time_t_table"},
                # {'label': 'Superficial Velocity', 'value': "Superficial_filter_velocity_table"},
                # {'label': 'Actual Linear Velocity', 'value': "actual_linear_velocity_table"},
                # {'label': 'Instantaneous Volumetric Water Flowrate', 'value': "instantaneous_volumetric_flowrate_of_water_table"},
                # {'label': 'Instantaneous Volumetric Oil Flowrate', 'value': "instantaneous_volumetric_flowrate_of_oil_table"},
                # {'label':'Cumulative Water Injected', 'value':"cumulative_water_injected_table"}

            ],
            value='Layer_table1',
            placeholder="X-axis",
            clearable=True
        )])
    if method=='Ro':
        method_table_list = html.Div(className='checklist_container',children=[dcc.Checklist(id='Robert-checklist',
                options=[
                    {'label': 'Fractional Flow Table', 'value': "Fractional_flow_table"},
                    {'label': 'Capacity', 'value': "Capacity"},
                    {'label': 'Fraction of Total Capacity', 'value': "Fraction_of_total_Capacity"},
                    {'label': 'Injection Rate Per Layer', 'value': "Injection_Rate_Per_Layer"},
                    {'label': 'Cumulative Water Injection Per Layer', 'value': "Cummulative_Water_Injection_Per_Layer_list"},
                    {'label': 'Oil Production Before Breakthrough', 'value': "Oil_Production_Before_Breakthrough"},
                    {'label': 'Oil Production Per Layer After Breakkthrough', 'value': "Oil_Production_Per_Layer_After_Breakthrough_list"},
                    {'label': 'Water Production Per Layer After Breakkthrough', 'value': "Water_Production_Per_Layer_After_Breakthrough_list"},
                    {'label': 'Recovery at Breakthrough per Layer', 'value': "Recovery_At_Breakthrough_Per_Layer_list"},
                    {'label': 'Cumulative Water Injection per Layer at Breakthrough', 'value': "Cummulative_Water_Injection_Per_Layer_At_Breakthrough_list"},
                    {'label': 'Time to Breakthrough for Each Layer', 'value': "Time_To_Breakthrough_For_Each_Layer_list"},
                    {'label': 'Oil Recovery to Each Point', 'value': "Oil_Recovery_To_Each_Point_List"},
                    {'label': 'Time to Each Point', 'value': "Time_To_Each_Point_list"},
                ],
                value=['Fractional_flow_table']
            )])
        x_axis_options=html.Div(className='checklist_container',children=[html.H5(children="x-axis"),dcc.Dropdown(id='Robert-xaxis-dropdown',
            options=[
                {'label': 'Capacity', 'value': "Capacity"},
                {'label': 'Fraction of Total Capacity', 'value': "Fraction_of_total_Capacity"},
                {'label': 'Injection Rate Per Layer', 'value': "Injection_Rate_Per_Layer"},
                {'label': 'Cumulative Water Injection Per Layer', 'value': "Cummulative_Water_Injection_Per_Layer_list"},
                {'label': 'Oil Production Before Breakthrough', 'value': "Oil_Production_Before_Breakthrough"},
                {'label': 'Oil Production Per Layer After Breakkthrough', 'value': "Oil_Production_Per_Layer_After_Breakthrough_list"},
                {'label': 'Water Production Per Layer After Breakkthrough', 'value': "Water_Production_Per_Layer_After_Breakthrough_list"},
                {'label': 'Recovery at Breakthrough per Layer', 'value': "Recovery_At_Breakthrough_Per_Layer_list,"},
                {'label': 'Cumulative Water Injection per Layer at Breakthrough', 'value': "Cummulative_Water_Injection_Per_Layer_At_Breakthrough_list"},
                {'label': 'Time to Breakthrough for Each Layer', 'value': "Time_To_Breakthrough_For_Each_Layer_list"}
            ],
            value='',
            placeholder="X-axis",
            clearable=True
        )])
    return method_table_list, x_axis_options

@app.callback(
    [Output("table-display", "children"),Output("Dykstra-Parsons-Chart", "figure")],
    [
        Input("SOR", "value"),
        Input("SWI", "value"),
        Input("SGI", "value"),
        Input("SOI", "value"),
        Input("VISO", "value"),
        Input("VISW", "value"),
        Input("OFVF", "value"),
        Input("WFVF", "value"),
        Input("injection-pressure", "value"),
        Input("length-of-bed", "value"),
        Input("bed-width", "value"),
        Input("CIR", "value"),
        Input("RGSUZ", "value"),
        Input("RGSSZ", "value"),
        Input("Dykstra-checklist", "value"),
        Input("chart-type", "value"),
        Input("Dykstra-xaxis_dropdown", "value"),
    ]
)
def D_tables(SOR, SWI, SGI, SOI, VISO, VISW, OFVF, WFVF, Inj_Pressure_differential, Length_of_bed_ft, width_of_bed_ft,
                Constant_injection_rate, Residual_gas_saturation_unswept_area, Residual_gas_saturation_swept_area,
                dykstra_parson_checklist,chart_type,Dykstra_xaxis_dropdown):
    suppress_callback_exceptions=True
    global Layer_table
    global Oil_Mobility_table
    global Water_Mobility_table
    global Water_Flowrate_per_bed_table
    global coverage_table
    global WOR_table
    global Cumulative_oil_recovery
    global Water_volume_to_fillup_gas_space_table
    global Producing_water_oil_ratio
    global Cumulative_water_produced
    global Cumulative_water_injected_table
    global Time_days_table
    global Time_years_table
    global Water_Flowrate_table
    global Oil_Flowrate_table
    global Oil_Flowrate_per_bed_table
    global Front_Location_list_DataTable
    
    import numpy as np
    Residual_gas_saturation = Residual_gas_saturation_unswept_area+Residual_gas_saturation_swept_area
    Saturation_gradient = 1-SOR-SWI

    KRW_1_SOR = np.interp(1-SOR, SW, KRW)
    KRO_SWI = np.interp(SWI, SW, KRO)
    Oil_Mobility =  permeability_array*KRO_SWI/VISO
    Oil_Mobility_table = pd.DataFrame(Oil_Mobility, columns = ['Oil Mobility'])

    Water_Mobility =  permeability_array*KRW_1_SOR/VISW
    Water_Mobility_table = pd.DataFrame(Water_Mobility, columns = ['Water Mobility'])

    Front_Location_list = []
    Mobility_Ratio = KRW_1_SOR*VISO/(KRO_SWI*VISW)
    Areal_sweep_efficiency_at_breakthrough = 0.54602036+(0.03170817/Mobility_Ratio)+(0.30222997/math.exp(Mobility_Ratio)-0.0050969*Mobility_Ratio)
    Displacement_efficiency = (1-SWI-SGI-SOR)/(1-SWI-SGI)
    Areal_sweep_efficiency =  Areal_sweep_efficiency_at_breakthrough+0.2749*np.log((1/Displacement_efficiency))
    
    #Average_porosity = np.mean(bed_data_sort.POROSITY)
    Average_porosity = np.mean(bed_data_sort.POROSITY)
    Area_acres = Length_of_bed_ft*width_of_bed_ft/43560
    Gross_rock_volume_acre_ft =  Area_acres*bed_data_sort.THICKNESS.sum()
    
    for permeability_index1 in range(len(permeability_array)):
        Front_Location = (Mobility_Ratio - np.sqrt(Mobility_Ratio**2+List_of_permeability_ratio_DataTable[permeability_index1]*(1-Mobility_Ratio**2)))/(Mobility_Ratio-1)
        Front_Location_list.append(Front_Location)
    #This code generates table of flood front location as the layers breakthrough
    Front_Location_list_DataTable = pd.DataFrame(Front_Location_list).transpose()
    #==========================================================================================================================
    # CALCULATING THE OIL FLOW RATE IN EACH BED AS EACH BED BREAKS THROUGH
    Water_Flowrate_per_bed = (0.0011267*width_of_bed_ft*bed_data_sort['THICKNESS']*Inj_Pressure_differential/Length_of_bed_ft)*Water_Mobility

    Water_Flowrate_per_bed_table = pd.DataFrame(Water_Flowrate_per_bed).rename(columns={'THICKNESS':'Water Flowrate Per Bed (Barrels/D)'})
    #=============================================================================================================================
    Water_Flowrate_list = []
    for n in range(len(permeability_array)):
        Water_Flowrate = Water_Flowrate_per_bed_table.iloc[0:n].sum()
        Water_Flowrate_list.append(Water_Flowrate)
    Water_Flowrate_table = pd.DataFrame(Water_Flowrate_list).rename(columns={'Water Flowrate Per Bed (Barrels/D)':'Water Production Rate (Barrels/D)'})
    #===========================================================================================================================
    # CALCULATING THE OIL FLOW RATE IN EACH BED AS EACH BED BREAKS THROUGH
    Oil_Flowrate_per_bed_list = []
    for bed in Front_Location_list_DataTable.columns:
        Oil_Flowrate_per_bed = (0.0011267*width_of_bed_ft*bed_data_sort['THICKNESS']*Inj_Pressure_differential/Length_of_bed_ft)*Water_Mobility/((1-Mobility_Ratio)*Front_Location_list_DataTable[bed]+Mobility_Ratio)
        Oil_Flowrate_per_bed_list.append(Oil_Flowrate_per_bed)    
    Oil_Flowrate_per_bed_table = pd.DataFrame(Oil_Flowrate_per_bed_list).transpose()
    #==========================================================================================================================
    Oil_Flowrate_list = []
    for n in range(len(permeability_array)):
        Oil_Flowrate = Oil_Flowrate_per_bed_table[n].sum()
        Oil_Flowrate_list.append(Oil_Flowrate)
    Oil_Flowrate_table = pd.DataFrame(Oil_Flowrate_list).rename(columns={0:'Oil Production Rate'})
    #===========================================================================================================================
    # CALCULATING THE VERTICAL COVERAGE
    coverage_list = []
    Total_Number_of_layers = len(permeability_array)-1
    for number_layer_breakthrough in range(len(permeability_array)):
        coverage_individual = (number_layer_breakthrough+((Total_Number_of_layers-number_layer_breakthrough)*Mobility_Ratio/(Mobility_Ratio-1))-(1/(Mobility_Ratio-1))*np.sqrt(Mobility_Ratio**2+List_of_permeability_ratio_DataTable[number_layer_breakthrough][1:]*(1-Mobility_Ratio**2)).sum())/Total_Number_of_layers
        coverage_list.append(coverage_individual)
    #Table of vertical coverage of the reservoir when a given layer just broke through.
    coverage_table = pd.DataFrame(coverage_list, columns=['Vertical Coverage (Fraction)'])
    #============================================================================================================================
    WOR_denominator_ratio_list = []
    for denominator_index in range(len(permeability_array)):
        WOR_denominator_ratio = permeability_array[denominator_index]/np.sqrt(Mobility_Ratio**2+List_of_permeability_ratio_DataTable[denominator_index]*(1-Mobility_Ratio**2))
        WOR_denominator_ratio_list.append(WOR_denominator_ratio)
    WOR_denominator_ratio_table = pd.DataFrame(WOR_denominator_ratio_list)
        #WOR_denominator_ratio_table
    WOR_list = []
    for n in range(len(permeability_array)):
        # CALCULATING THE WATER OIL RATIO, WORn and generate table
        sum_of_permeability = bed_data_sort.PERMEABILITY.iloc[0:n].sum()
        #for number_layer_breakthrough in range(len(permeability_array)):
        WOR = sum_of_permeability/(WOR_denominator_ratio_table[n].sum())
        WOR_list.append(WOR)
    WOR_table = pd.DataFrame(WOR_list).rename(columns={0:'Water-Oil Ratio'})#,columns=['WATER-OIL RATIO'])
    #==========================================================================================================================
    #CALCULATING THE CUMULATIVE OIL RECOVERY AS EACH BED BREAKSTHROUGH.
    Cumulative_oil_recovery = (7758*Areal_sweep_efficiency_at_breakthrough*Gross_rock_volume_acre_ft*Average_porosity*(SOI-SOR)*coverage_table).rename(columns={'Vertical Coverage (Fraction)':'Cumulative Oil Recovery (Barrels)'})
    #============================================================================================================================
    #CALCULATING THE VOLUME OF WATER REQUIRED TO FILL-UP THE GAS SPACE.
    Water_volume_to_fillup_gas_space = 7758*Area_acres*bed_data_sort.THICKNESS*bed_data_sort.POROSITY*(SGI-Residual_gas_saturation)
    Water_volume_to_fillup_gas_space_table=pd.DataFrame(Water_volume_to_fillup_gas_space, columns = ['Water Volume For Gas Space Fill-Up'])
    #============================================================================================================================
    #CALCULATING THE PRODUCING WATER-OIL RATIO
    Producing_water_oil_ratio = (WOR_table*OFVF).rename(columns={'Water-Oil Ratio':'Producing Water-Oil Ratio'})
    Producing_water_oil_ratio
    #===========================================================================================================================
    # Note that the integration for the calculation of the cumulative oil produced starts from 0
    # Hence, a new row will have to be inserted at the first row with element 0
    # this is done for both the producing water-oil ratio and the cumulative oil produced.

    # for the cumulative oil recovery
    Cumulative_oil_recovery.loc[-1] = [0]  # adding a row
    #Cumulative_oil_recovery.index = Cumulative_oil_recovery.index + 1  # shifting index
    Cumulative_oil_recovery_Starting_from_0 = Cumulative_oil_recovery.sort_index()  # sorting by index

    # for the producing water-oil ratio
    Producing_water_oil_ratio.loc[-1] = [0]  # adding a row
    #Producing_water_oil_ratio.index = Producing_water_oil_ratio.index + 1  # shifting index
    Producing_water_oil_ratio_Starting_from_0 = Producing_water_oil_ratio.sort_index()  # sorting by index

    # CALCULATING THE CUMULATIVE WATER PRODUCTION
    # To determine the cumulative water production, the produced water oil ratio is ingreated against the cumulative oil recovery.
    # The integration uses a cumulative trapezoidal row by row integration.
    # import numpy and scipy.integrate.cumtrapz 
    import numpy as np 
    from scipy import integrate
    # Preparing the Integration variables y, x.
    # the to.numpy() method converts from dataframe to numpy array which appears in the form of list of lists in the array. 
    #The concatenate function helps to bring the list of lists together.
    x = np.concatenate(Cumulative_oil_recovery_Starting_from_0.to_numpy(),axis=0)
    y = np.concatenate(Producing_water_oil_ratio_Starting_from_0.to_numpy(),axis=0) 
    # using scipy.integrate.cumtrapz() method 
    Cumulative_water_produced = pd.DataFrame(integrate.cumtrapz(y, x), columns = ['Cumulative Water Produced'])
    #==============================================================================================================================
    # CALCULATING THE CUMULATIVE WATER INJECTED, Wi
    Cumulative_water_injected = (Cumulative_water_produced['Cumulative Water Produced'] + OFVF*Cumulative_oil_recovery['Cumulative Oil Recovery (Barrels)'] + Water_volume_to_fillup_gas_space_table['Water Volume For Gas Space Fill-Up']).drop([-1])
    Cumulative_water_injected_table = pd.DataFrame(Cumulative_water_injected,columns = ['Cumulative Water Injected (Barrels)'])
    #===================================================================================================================================
    # CALCULATING THE TIME REQUIRED FOR INJECTION TO REACH A GIVEN RECOVERY.
    Time_days = Cumulative_water_injected_table['Cumulative Water Injected (Barrels)']/Constant_injection_rate
    Time_days_table = pd.DataFrame(Time_days).rename(columns ={'Cumulative Water Injected (Barrels)': 'Time (Days)'}, inplace = False)
    #print(Time_days_table)
    Time_years = Time_days_table/365
    Time_years_table = Time_years.rename(columns ={'Time (Days)': 'Time (Years)'}, inplace = False)
    #=======================================================================================================================================
    # TABLE OF ALL OBTAINED VALUES.
    Dykstra_dataframe_list = [Layer_table, Oil_Mobility_table, Water_Mobility_table, Water_Flowrate_per_bed_table,
    coverage_table, WOR_table, Cumulative_oil_recovery, Water_volume_to_fillup_gas_space_table, Producing_water_oil_ratio,
    Cumulative_water_produced, Cumulative_water_injected_table, Time_days_table, Time_years_table, Water_Flowrate_table,
     Oil_Flowrate_table, Oil_Flowrate_per_bed_table, Front_Location_list_DataTable]

    
    #This code coverts strings to variable names or removes quotation marks from a list of strings
    translation = {39: None}
    dykstra_parson_checklists = str(dykstra_parson_checklist).translate(translation)

    #This code converts variable name to strings
    def variablename(var):
        return [tpl[0] for tpl in filter(lambda x:var is x[1], globals().items())]
        
    #Compare checklist with a list of dataframe variables and append the result to an empty dataframe

    l_d = pd.DataFrame()
    # print(dykstra_parson_checklist)
    for j in dykstra_parson_checklist:
        for d in Dykstra_dataframe_list:
            #print(d)
            if variablename(d)[0] == j:
                l_d = l_d.append(d, ignore_index=True)

    translation = {39: None}
    Dykstra_xaxis_dropdowns = str(Dykstra_xaxis_dropdown).translate(translation)
    def variablename(var):
        return [tpl[0] for tpl in filter(lambda x: var is x[1], globals().items())]
    l_dxaxis = pd.DataFrame()
    for q in Dykstra_dataframe_list:
        #print(q)
        if variablename(q)[0] == Dykstra_xaxis_dropdown:
            l_dxaxis=l_dxaxis.append(q,ignore_index=True)
    
    # Time = []
    # for i in Time_days_table.values.tolist():
    #     for k in i:
    #         Time.append(k)

    ld = []
    for j in l_d.values.tolist():
        for m in j:
            ld.append(m)

    ld_xaxis = []
    for j in l_dxaxis.values.tolist():
        for m in j:
            ld_xaxis.append(m)
    #Replace column names with column index

    #The code integrates marker status
    if chart_type == 'line':
        marker = None
    else:
        marker='markers'
    Dykstra_Parsons_Chart = {
        "data": [
            {
                "x": ld_xaxis,
                "y": ld,
                "type": str(chart_type),
                "mode": marker,
                
            },
        ],
        "layout": {
            "title": {"text": 'Dykstra-Parsons: '+str(dykstra_parson_checklists) +' vs ' + str(Dykstra_xaxis_dropdowns), "x": 0.05, "xanchor": "left"},
            "xaxis": {"title":str(Dykstra_xaxis_dropdowns),"fixedrange": True},
            "yaxis": {"title":str(dykstra_parson_checklists),"fixedrange": True},
            "colorway": ["#E12D39"],
        },
    }
    return dash_table.DataTable(
                            columns=[{"name": str(i), "id": str(i)} for i in l_d.columns],
                            data=l_d.to_dict('records'),
                            editable=True,
                            style_header={'backgroundColor': 'rgb(30, 30, 30)'},
                            style_cell={
                                'backgroundColor': '#54c5f9',
                                'color': 'white'
                            }
                            ), Dykstra_Parsons_Chart

@app.callback(
    [Output("Reznik-table-display", "children"),
    Output("Reznik-discrete-point-division", "children"),
    Output("Reznik-Chart", "figure")],
    [
        Input("number-of-points", "value"),
        Input("SOR", "value"),
        Input("SWI", "value"),
        Input("SGI", "value"),
        Input("SOI", "value"),
        Input("VISO", "value"),
        Input("VISW", "value"),
        Input("OFVF", "value"),
        Input("WFVF", "value"),
        Input("injection-pressure", "value"),
        Input("length-of-bed", "value"),
        Input("bed-width", "value"),
        Input("CIR", "value"),
        Input("RGSUZ", "value"),
        Input("RGSSZ", "value"),
        Input("Reznik-checklist", "value"),
        Input("chart-type", "value"),
        Input("Reznik-xaxis-dropdown", "value"),
    ]
)

def Reznik(Number_of_points, SOR, SWI, SGI, SOI, VISO, VISW, OFVF, WFVF, Inj_Pressure_differential, Length_of_bed_ft,
    width_of_bed_ft, Constant_injection_rate, Residual_gas_saturation_unswept_area, Residual_gas_saturation_swept_area,
    Reznik_checklist,chart_type,Reznik_xaxis_dropdown):
    try:
        

        #RGSU = float(entries['Residual_gas_saturation_unswept_area'].get())
        #RGSS =  float(entries['Residual_gas_saturation_swept_area'].get())
        global Real_time_CIP_table
        global Layer_table1
        global breakthrough_time_table
        global Flood_front_position_of_bed_n_j
        global Ultimate_recoverable_oil_per_bed_table
        global Front_position_of_other_beds_at_breakthrough_table
        global flood_front_of_last_bed_table, Real_time_CIP_table
        global dynamic_bed_table
        global sum_water_flowrate_before_breakthrough_of_dynamic_bed_table
        global sum_oil_flowrate_before_breakthrough_of_dynamic_bed_table
        global Instantaneous_producing_Water_Oil_Ratio_before_breakthrough_of_dynamic_bed_table
        global Instantaneous_producing_Water_cut_table
        global cumulative_oil_recovered_at_time_t_table
        global Cumulative_oil_recovered_from_all_beds_table
        global Vertical_coverage_at_time_t_table
        global Cumumlative_water_oil_ratio_for_CIR_table
        global Cumumlative_water_oil_ratio_for_CIP_table
        global Flood_front_location_of_other_beds_table
        global Flood_front_location_of_other_beds_beyond_breakthrough_table
        global Property_time_table
        global average_mobility_at_time_t_table
        global Superficial_filter_velocity_table
        global actual_linear_velocity_table
        global instantaneous_volumetric_flowrate_of_water_table
        global instantaneous_volumetric_flowrate_of_oil_table
        global cumulative_water_injected_table

        import pandas as pd
        import math
        import numpy as np
        KRW_1_SOR = np.interp(1-SOR, SW, KRW)
        KRO_SWI = np.interp(SWI, SW, KRO)

        Water_Mobility = bed_data.PERMEABILITY*KRW_1_SOR/VISW
        Water_Mobility_table = pd.DataFrame(Water_Mobility).rename(columns={'PERMEABILITY': 'Water Mobility'})
        Saturation_gradient = 1-SOR-SWI
        #Water_Mobility_table
        #==========================================================================================================================
        #CALCULATING THE WATER MOBILITY

        Oil_Mobility = bed_data.PERMEABILITY*KRO_SWI/VISO
        Oil_Mobility_table= pd.DataFrame(Oil_Mobility).rename(columns={'PERMEABILITY': 'Oil Mobility'})
        #Oil_Mobility_table
        #=========================================================================================================================
        # CALCULATING THE MOBILITY RATIO, M.
        #import math
        Mobility_Ratio = Water_Mobility/Oil_Mobility
        Mobility_Ratio_table= pd.DataFrame(Mobility_Ratio).rename(columns={'PERMEABILITY': 'MOBILITY RATIO'})
        #Mobility_Ratio_table

        #==========================================================================================================================

        # ARRANGING THE DATA IN ORDER OF DECREASING PERMEABILITY.
        Bed_ordering_parameter=np.array(bed_data.POROSITY)*Saturation_gradient*(1+Mobility_Ratio)/Water_Mobility

        # ARRANGING THE DATA IN ORDER OF DECREASING PERMEABILITY.
        #Bed Ordering
        Bed_ordering_parameter = np.array(bed_data.POROSITY)*Saturation_gradient*(1+Mobility_Ratio)/Water_Mobility
        Bed_ordering_parameter_table = pd.DataFrame(Bed_ordering_parameter).rename(columns={'PERMEABILITY': 'BED ORDERING PARAMETER'})
        bed_data_combine = pd.concat([bed_data, Bed_ordering_parameter_table,Water_Mobility_table,Oil_Mobility_table,Mobility_Ratio_table], axis = 1)

        bed_data_sort = bed_data_combine.sort_values(by='BED ORDERING PARAMETER',ignore_index=True, ascending=True) 
        Average_porosity = '%.3f' % np.mean(bed_data_sort.POROSITY)
        #===========================================================================================================================
        # Extracting input variables from data table.
        import numpy as np
        Layers = np.array(bed_data_sort['LAYER'])
        Layer_table1 = pd.DataFrame(Layers, columns=['Layers'])
        Bed_ordering_parameter = np.array(bed_data_sort['BED ORDERING PARAMETER'])
        Bed_ordering_parameter_sort_table = pd.DataFrame(Bed_ordering_parameter)
        PORO = np.array(bed_data_sort['POROSITY'])
        Porosity_sort_table = pd.DataFrame(PORO)
        permeability_array = np.array(bed_data_sort['PERMEABILITY'])

        Water_mobility_array = np.array(bed_data_sort['Water Mobility'])
        Water_mobility_sort_table = pd.DataFrame(Water_mobility_array)

        Oil_mobility_array = np.array(bed_data_sort['Oil Mobility'])
        Oil_mobility_sort_table = pd.DataFrame(Oil_mobility_array)

        Permeability_sort_table = pd.DataFrame(permeability_array)
        bed_thickness = np.array(bed_data_sort['THICKNESS'])

        #==========================================================================================================================

        #Bed order parameter ratio of each bed to the last bed
        bed_order_ratio_list = []
        for j in range(len(Layers)):
                bed_order_ratio_to_lastbed = bed_data_sort['BED ORDERING PARAMETER'][j]/bed_data_sort['BED ORDERING PARAMETER'].iat[-1]
                bed_order_ratio_list.append(bed_order_ratio_to_lastbed)
        bed_order_ratio=pd.DataFrame(bed_order_ratio_list)

        #==========================================================================================================================

        #Bed order parameter ratio of each bed to the last bed
        bed_order_ratio_to_other_beds_list = []
        for j in range(len(Layers)):
                bed_order_ratio_to_otherbeds = bed_data_sort['BED ORDERING PARAMETER'].iat[-1]/bed_data_sort['BED ORDERING PARAMETER'][j]
                bed_order_ratio_to_other_beds_list.append(bed_order_ratio_to_otherbeds)
        bed_order_ratio_to_otherbeds=pd.DataFrame(bed_order_ratio_to_other_beds_list)

        #===========================================================================================================================
        #Flood front position of bed n when bed j has just broken through.

        last_mobility_ratio = bed_data_sort['MOBILITY RATIO'].iat[-1]

        Flood_front_position_of_bed_n_j = (-last_mobility_ratio+np.sqrt(last_mobility_ratio**2+(bed_order_ratio)*(1-last_mobility_ratio**2)))/(1-last_mobility_ratio)
        Flood_front_position_of_bed_n_j = pd.DataFrame(Flood_front_position_of_bed_n_j).rename(columns = {0:'Flood Front Position of the last bed at breakthrough of other beds'})

        #==========================================================================================================================
        #Flood front location of the last bed.
        #converting the flood front table to a list
        flood_front_of_last_bed = 0
        flood_front_of_last_bed_list = []
        if Mobility_Ratio_table.iloc[0,0] == 1:
            Flood_front_position_of_bed_n_j_list=bed_order_ratio.to_list()
            for index, position in list(enumerate(Flood_front_position_of_bed_n_j_list)):
                while flood_front_of_last_bed < Flood_front_position_of_bed_n_j_list[0]:
                   # if flood_front_of_last_bed < Flood_front_position_of_bed_n_j_list[0]:
                    flood_front_of_last_bed = flood_front_of_last_bed + Flood_front_position_of_bed_n_j_list[0]/Number_of_points
                    flood_front_of_last_bed_list.append(flood_front_of_last_bed)


                if(index > 0):
                    while flood_front_of_last_bed >=Flood_front_position_of_bed_n_j_list[index-1] and flood_front_of_last_bed <= Flood_front_position_of_bed_n_j_list[index]:
                        flood_front_of_last_bed = flood_front_of_last_bed + (Flood_front_position_of_bed_n_j_list[index]-Flood_front_position_of_bed_n_j_list[index-1])/Number_of_points
                        flood_front_of_last_bed_list.append(flood_front_of_last_bed)

        else:
            Flood_front_position_of_bed_n_j_list=Flood_front_position_of_bed_n_j['Flood Front Position of the last bed at breakthrough of other beds'].to_list()
            for index, position in list(enumerate(Flood_front_position_of_bed_n_j_list)):
                while flood_front_of_last_bed < Flood_front_position_of_bed_n_j_list[0]:
                   # if flood_front_of_last_bed < Flood_front_position_of_bed_n_j_list[0]:
                    flood_front_of_last_bed = flood_front_of_last_bed + Flood_front_position_of_bed_n_j_list[0]/Number_of_points
                    flood_front_of_last_bed_list.append(flood_front_of_last_bed)


                if(index > 0):
                    while flood_front_of_last_bed >=Flood_front_position_of_bed_n_j_list[index-1] and flood_front_of_last_bed <= Flood_front_position_of_bed_n_j_list[index]:
                        flood_front_of_last_bed = flood_front_of_last_bed + (Flood_front_position_of_bed_n_j_list[index]-Flood_front_position_of_bed_n_j_list[index-1])/Number_of_points
                        flood_front_of_last_bed_list.append(flood_front_of_last_bed)


        flood_front_of_last_bed_table = pd.DataFrame(flood_front_of_last_bed_list).rename(columns = {0:'Flood Front Position of the last bed at time t'})

        #===========================================================================================================================

        #Calculating Real or Process time for the CIP case
        porosity_of_last_bed = bed_data_sort['POROSITY'].iat[-1]
        water_mobility_of_last_bed = bed_data_sort['Water Mobility'].iat[-1]
        Real_time_CIP = 158.064*((Length_of_bed_ft**2/Inj_Pressure_differential)*porosity_of_last_bed*Saturation_gradient/water_mobility_of_last_bed)*(last_mobility_ratio*np.array(flood_front_of_last_bed_list) + 0.5*(1-last_mobility_ratio)*np.array(flood_front_of_last_bed_list)**2)
        Real_time_CIP_table = pd.DataFrame(Real_time_CIP).rename(columns = {0:'Real time for constant injection pressure'})
        #Real_time_CIP_table

        #==========================================================================================================================
        # Calculating breakthrough time of each bed.
        porosity_of_last_bed = bed_data_sort['POROSITY'].iat[-1]
        water_mobility_of_last_bed = bed_data_sort['Water Mobility'].iat[-1]
        breakthrough_time = 158.064*((Length_of_bed_ft**2/Inj_Pressure_differential)*porosity_of_last_bed*Saturation_gradient/water_mobility_of_last_bed)*(last_mobility_ratio*np.array(Flood_front_position_of_bed_n_j['Flood Front Position of the last bed at breakthrough of other beds'].to_list()) + 0.5*(1-last_mobility_ratio)*np.array(Flood_front_position_of_bed_n_j['Flood Front Position of the last bed at breakthrough of other beds'].to_list())**2)
        breakthrough_time_table = pd.DataFrame(breakthrough_time).rename(columns = {0:'Breakthrough time'})
        #breakthrough_time_table

        #==========================================================================================================================
        # Flood front position of other beds with resect to bed n
        Flood_front_location_of_other_beds_list = []
        for j in range(len(Layers)):
            aj = Mobility_Ratio[j]**2
            bed_order_of_last_bed = bed_data_sort['BED ORDERING PARAMETER'].iat[-1]
            bj = (bed_order_ratio_to_other_beds_list[j])*(2*last_mobility_ratio/(1+last_mobility_ratio))*(1-Mobility_Ratio[j]**2)
            cj = (bed_order_ratio_to_other_beds_list[j])*((1-last_mobility_ratio)/(1+last_mobility_ratio))*(1-Mobility_Ratio[j]**2)

            Flood_front_location_of_other_beds = (-Mobility_Ratio[j]+np.sqrt(aj+bj*np.array(flood_front_of_last_bed_list)+cj*np.array(flood_front_of_last_bed_list)**2))/(1-Mobility_Ratio[j])

            Flood_front_location_of_other_beds_list.append(Flood_front_location_of_other_beds)
            for i in range(len(Flood_front_location_of_other_beds_list[j])):
                if Flood_front_location_of_other_beds_list[j][i] > 1:
                     Flood_front_location_of_other_beds_list[j][i] = 1 
        Flood_front_location_of_other_beds_table = pd.DataFrame(Flood_front_location_of_other_beds_list).transpose()
        #Flood_front_location_of_other_beds_table
        #==========================================================================================================================
                
        
        #==========================================================================================================================
        '''# Front position of other beds at breakthrough.'''
        Front_position_of_other_beds_at_breakthrough_list = []
        
        if Mobility_Ratio_table.iloc[0,0] == 1:
            for j in range(len(Layers)):
                Front_position_of_other_beds_at_breakthrough= ((bed_order_ratio_to_other_beds_list[j])*((1-last_mobility_ratio)*(np.array(Flood_front_position_of_bed_n_j['Flood Front Position of the last bed at breakthrough of other beds'].to_list()))**2 +2*last_mobility_ratio*(np.array(Flood_front_position_of_bed_n_j['Flood Front Position of the last bed at breakthrough of other beds'].to_list()))))/(1+last_mobility_ratio)
                Front_position_of_other_beds_at_breakthrough_list.append(Front_position_of_other_beds_at_breakthrough)
        else:    
            for j in range(len(Layers)):
                aj = Mobility_Ratio[j]**2
                bed_order_of_last_bed = bed_data_sort['BED ORDERING PARAMETER'].iat[-1]
                bj = (bed_order_ratio_to_other_beds_list[j])*(2*last_mobility_ratio/(1+last_mobility_ratio))*(1-Mobility_Ratio[j]**2)
                cj = (bed_order_ratio_to_other_beds_list[j])*((1-last_mobility_ratio)/(1+last_mobility_ratio))*(1-Mobility_Ratio[j]**2)
        
                Front_position_of_other_beds_at_breakthrough = (-Mobility_Ratio[j]+np.sqrt(aj+bj*np.array(Flood_front_position_of_bed_n_j['Flood Front Position of the last bed at breakthrough of other beds'].to_list())+cj*np.array(Flood_front_position_of_bed_n_j['Flood Front Position of the last bed at breakthrough of other beds'].to_list())**2))/(1-Mobility_Ratio[j])
        
                Front_position_of_other_beds_at_breakthrough_list.append(Front_position_of_other_beds_at_breakthrough)
        
        Front_position_of_other_beds_at_breakthrough_table = pd.DataFrame(Front_position_of_other_beds_at_breakthrough_list)
        #============================================================================================
        
        # Flood front position of other beds with resect to bed n. This is to know how far each front has advanced beyond the bed
        from decimal import Decimal
        Flood_front_location_of_other_beds_beyond_breakthrough_list = []
        if Mobility_Ratio_table.iloc[0,0] == 1:
            for j in range(len(Layers)):
                Flood_front_location_of_other_beds_beyond_breakthrough= ((bed_order_ratio_to_other_beds_list[j])*((1-last_mobility_ratio)*(flood_front_of_last_bed_list)**2 +2*last_mobility_ratio*(flood_front_of_last_bed_list)))/(1+last_mobility_ratio)
                Flood_front_location_of_other_beds_beyond_breakthrough_list.append(Flood_front_location_of_other_beds_beyond_breakthrough)
        else:
            for j in range(len(Layers)):
                aj = Mobility_Ratio[j]**2
                bed_order_of_last_bed = bed_data_sort['BED ORDERING PARAMETER'].iat[-1]
                bj = (bed_order_ratio_to_other_beds_list[j])*(2*last_mobility_ratio/(1+last_mobility_ratio))*(1-Mobility_Ratio[j]**2)
                cj = (bed_order_ratio_to_other_beds_list[j])*((1-last_mobility_ratio)/(1+last_mobility_ratio))*(1-Mobility_Ratio[j]**2)
    
                Flood_front_location_of_other_beds_beyond_breakthrough = (-Mobility_Ratio[j]+np.sqrt(aj+bj*np.array(flood_front_of_last_bed_list)+cj*np.array(flood_front_of_last_bed_list)**2))/(1-Mobility_Ratio[j])
    
                Flood_front_location_of_other_beds_beyond_breakthrough_list.append(Flood_front_location_of_other_beds_beyond_breakthrough)
        Flood_front_location_of_other_beds_beyond_breakthrough_table = pd.DataFrame(Flood_front_location_of_other_beds_beyond_breakthrough_list).transpose().round(4)
        #Flood_front_location_of_other_beds_beyond_breakthrough_table

        #==================================================================================================================================
        Property_time_list = []
        for i in range(len(Layers)):
            Property_time = 158.064*((Length_of_bed_ft**2/Inj_Pressure_differential)*bed_data_sort['POROSITY'][i]*Saturation_gradient/bed_data_sort['Water Mobility'][i])*(Mobility_Ratio[i]*Flood_front_location_of_other_beds_beyond_breakthrough_table[i]+0.5*(1-Mobility_Ratio[i])*Flood_front_location_of_other_beds_beyond_breakthrough_table[i]**2)
            Property_time_list.append(Property_time)
        Property_time_table= pd.DataFrame(Property_time_list).T
        #Property_time_table

        #==========================================================================================================================
        #Average mobility of the fluids in each bed at time t
        average_mobility_at_time_t_list = []
        for i in range(len(Layers)):
            average_mobility_at_time_t = Water_mobility_array[i]/(Mobility_Ratio[i]+(1-Mobility_Ratio[i])*Flood_front_location_of_other_beds_beyond_breakthrough_table[i])
            average_mobility_at_time_t_list.append(average_mobility_at_time_t)
        average_mobility_at_time_t_table = pd.DataFrame(average_mobility_at_time_t_list).transpose()

        #==========================================================================================================================
        #Superficial filter velocity of Darcy's law at time t
        Superficial_filter_velocity_list = []
        for i in range(len(Layers)):
            Superficial_filter_velocity = (Inj_Pressure_differential/Length_of_bed_ft)*average_mobility_at_time_t_table[i]
            Superficial_filter_velocity_list.append(Superficial_filter_velocity)
        Superficial_filter_velocity_table = pd.DataFrame(Superficial_filter_velocity_list).transpose()

        #==========================================================================================================================
        #Real time actual linear velocity of the flood front.
        actual_linear_velocity_list = []
        for i in range(len(Layers)):
            actual_linear_velocity  = Superficial_filter_velocity_table[i]/(bed_data_sort['POROSITY'][i]*Saturation_gradient)
            actual_linear_velocity_list.append(actual_linear_velocity)
        actual_linear_velocity_table = pd.DataFrame(actual_linear_velocity_list).transpose()

        #==========================================================================================================================
        # Instantaneous volumetric flow rate of water into bed.
        instantaneous_volumetric_flowrate_of_water_list = []
        for i in range(len(Layers)):
            instantaneous_volumetric_flowrate_of_water = 0.0011267*width_of_bed_ft*bed_thickness[i]*Superficial_filter_velocity_table[i]
            instantaneous_volumetric_flowrate_of_water_list.append(instantaneous_volumetric_flowrate_of_water)
        instantaneous_volumetric_flowrate_of_water_table = pd.DataFrame(instantaneous_volumetric_flowrate_of_water_list).transpose()
        #instantaneous_volumetric_flowrate_of_water_table

        #==========================================================================================================================
        # Instantaneous volumetric flow rate of oil into bed.
        instantaneous_volumetric_flowrate_of_oil_list = []
        for i in range(len(Layers)):
            instantaneous_volumetric_flowrate_of_oil = 0.0011267*width_of_bed_ft*bed_thickness[i]*Superficial_filter_velocity_table[i]/((1-bed_data_sort['MOBILITY RATIO'][i])*Flood_front_location_of_other_beds_table[i]+bed_data_sort['MOBILITY RATIO'][i])
            instantaneous_volumetric_flowrate_of_oil_list.append(instantaneous_volumetric_flowrate_of_oil)
        instantaneous_volumetric_flowrate_of_oil_table = pd.DataFrame(instantaneous_volumetric_flowrate_of_oil_list).transpose()

        #==========================================================================================================================
        # Total flow rate for each bed.
        Constant_total_injection_rate_list = []
        for i in range(len(Layers)):
            Constant_total_injection_rate = np.sum(instantaneous_volumetric_flowrate_of_water_table[i])
            Constant_total_injection_rate_list.append(Constant_total_injection_rate)
        Constant_total_injection_rate_table = pd.DataFrame(Constant_total_injection_rate_list)
        Constant_total_injection_rate_for_all_beds = Constant_total_injection_rate_table.sum(axis=0).values[0]
        #Constant_total_injection_rate_for_all_beds
        #==========================================================================================================================
        # Get the count of ones in each column at a given time
        number_of_ones_list = {}
        for i in range(len(Real_time_CIP_table)+1):
            number_of_ones_list[i] = Flood_front_location_of_other_beds_table[0:i].isin([1]).sum().to_frame().T.iloc[0,:]
            #number_of_ones_list.append(number_of_ones)
        number_of_ones_table = pd.DataFrame.from_dict(number_of_ones_list).T

        # returns the column with lowest count of 1 at a given time period. this represents the dynamic bed.
        number_of_ones_table['Dynamic_bed'] = number_of_ones_table.idxmin(axis=1)
        pd.set_option("display.max_rows", None, "display.max_columns", None)

        #==========================================================================================================================
        dynamic_bed = number_of_ones_table['Dynamic_bed']
        dynamic_bed_table = pd.DataFrame(dynamic_bed).rename(columns = {'Dynamic_bed':'Dynamic Bed'})
        #dynamic_bed_table = pd.DataFrame(number_of_ones_table['Dynamic_bed'], columns = ['Dynamic bed'])
        water_flow_rate_and_Dynamic_bed=pd.concat([instantaneous_volumetric_flowrate_of_water_table,dynamic_bed], axis = 1)
        #print(dynamic_bed_table)
        #==========================================================================================================================
        #just before breakthrough of the dynamic bed
        sum_water_flowrate_before_breakthrough_of_dynamic_bed_list = []
        for i in range(len(Real_time_CIP_table)):
            for j in range(len(Layers)):
                if dynamic_bed[i] == j:
                    sum_water_flowrate_before_breakthrough_of_dynamic_bed = instantaneous_volumetric_flowrate_of_water_table.iloc[i,0:j].sum(axis = 0)
                    sum_water_flowrate_before_breakthrough_of_dynamic_bed_list.append(sum_water_flowrate_before_breakthrough_of_dynamic_bed)
        sum_water_flowrate_before_breakthrough_of_dynamic_bed_table = pd.DataFrame(sum_water_flowrate_before_breakthrough_of_dynamic_bed_list).rename(columns = {0:'Sum water flowrate before breakthrough of dynamic bed'})
        #sum_water_flowrate_before_breakthrough_of_dynamic_bed_table

        #==========================================================================================================================
        #just before breakthrough of the dynamic bed
        sum_oil_flowrate_before_breakthrough_of_dynamic_bed_list = []
        for i in range(len(Real_time_CIP_table)):
            for j in range(len(Layers)):
                if dynamic_bed[i] == j:
                    sum_oil_flowrate_before_breakthrough_of_dynamic_bed = instantaneous_volumetric_flowrate_of_oil_table.iloc[i,j:len(Layers)+1].sum(axis = 0)
                    sum_oil_flowrate_before_breakthrough_of_dynamic_bed_list.append(sum_oil_flowrate_before_breakthrough_of_dynamic_bed)
        sum_oil_flowrate_before_breakthrough_of_dynamic_bed_table = pd.DataFrame(sum_oil_flowrate_before_breakthrough_of_dynamic_bed_list).rename(columns = {0:'Sum oil flowrate before breakthrough of dynamic bed'})
        sum_oil_flowrate_before_breakthrough_of_dynamic_bed_table

        #==========================================================================================================================
        # Instantaneous producing WOR, defined at xj = l, for all j, at time t just before breakthrough of the dynamic bed
        Instantaneous_producing_Water_Oil_Ratio_before_breakthrough_of_dynamic_bed = (np.array(sum_water_flowrate_before_breakthrough_of_dynamic_bed_list)/np.array(sum_oil_flowrate_before_breakthrough_of_dynamic_bed_list))
        Instantaneous_producing_Water_Oil_Ratio_before_breakthrough_of_dynamic_bed_table = pd.DataFrame(Instantaneous_producing_Water_Oil_Ratio_before_breakthrough_of_dynamic_bed).rename(columns = {0:'Instantaneous producing Water Oil ratio before breakthrough of dynamic bed'})
        #Instantaneous_producing_Water_Oil_Ratio_before_breakthrough_of_dynamic_bed_table

        # Instantaneous producing Water cut, defined at xj = l, for all j, at time t just before breakthrough of the dynamic bed
        Instantaneous_producing_Water_cut = np.array(sum_water_flowrate_before_breakthrough_of_dynamic_bed_list)/(np.array(sum_oil_flowrate_before_breakthrough_of_dynamic_bed_list)+np.array(sum_water_flowrate_before_breakthrough_of_dynamic_bed_list))
        Instantaneous_producing_Water_cut_table = pd.DataFrame(Instantaneous_producing_Water_cut).rename(columns = {0:'Instantaneous producing Water cut'})
        #Instantaneous_producing_Water_cut_table

        #==========================================================================================================================
        # Ultimate recoverable oil per bed
        global Ultimate_recoverable_oil_per_bed_table
        Ultimate_recoverable_oil_per_bed_list = []
        for i in range(len(Layers)):
            Ultimate_recoverable_oil_per_bed = 0.1781*Length_of_bed_ft*width_of_bed_ft*bed_data_sort['THICKNESS'][i]*bed_data_sort['POROSITY'][i]*Saturation_gradient
            Ultimate_recoverable_oil_per_bed_list.append(Ultimate_recoverable_oil_per_bed)
        Ultimate_recoverable_oil_per_bed_table = pd.DataFrame(Ultimate_recoverable_oil_per_bed_list).rename(columns = {0:'Ultimate recoverable oil per bed'})
        #Ultimate_recoverable_oil_per_bed_table
        #==========================================================================================================================
        # Total recoverable oil in place for the entire system of n beds.

        Total_recoverable_oil_in_place = Ultimate_recoverable_oil_per_bed_table.sum(axis = 0).values[0]

        #Total_recoverable_oil_in_place

        #==========================================================================================================================
        # Product of flood front location and ultimate recovery at per bed.
        Product_of_flood_front_location_and_ultimate_recovery_list = []
        for j in range(len(Layers)):
            Product_of_flood_front_location_and_ultimate_recovery = Flood_front_location_of_other_beds_beyond_breakthrough_table[j]*Ultimate_recoverable_oil_per_bed_table.iloc[j,0]
            Product_of_flood_front_location_and_ultimate_recovery_list.append(Product_of_flood_front_location_and_ultimate_recovery)
        Product_of_flood_front_location_and_ultimate_recovery_table = pd.DataFrame(Product_of_flood_front_location_and_ultimate_recovery_list).T
        #Product_of_flood_front_location_and_ultimate_recovery_table

        #==========================================================================================================================
        # cumulative oil recovered from all beds at time t .
        # Term 1
        cumulative_oil_recovered_at_time_t_list = []
        for i in range(len(Real_time_CIP_table)):
            for j in range(len(Layers)):
                if dynamic_bed[i] == j:
                    cumulative_oil_recovered_at_time_t = np.array(Ultimate_recoverable_oil_per_bed_list)[0:j].sum(axis = 0)
                    cumulative_oil_recovered_at_time_t_list.append(cumulative_oil_recovered_at_time_t)
        cumulative_oil_recovered_at_time_t_table = pd.DataFrame(cumulative_oil_recovered_at_time_t_list)
        #cumulative_oil_recovered_at_time_t_table

        # Term 2
        cumulative_oil_recovered_and_flood_front_location_at_time_t_list = []
        for k in range(len(Real_time_CIP_table)):
            for l in range(len(Layers)):
                if dynamic_bed[k] == l:
                    cumulative_oil_recovered_and_flood_front_location_at_time_t = Product_of_flood_front_location_and_ultimate_recovery_table.iloc[k,l:len(Layers)+1].sum(axis = 0)
                    cumulative_oil_recovered_and_flood_front_location_at_time_t_list.append(cumulative_oil_recovered_and_flood_front_location_at_time_t)
        cumulative_oil_recovered_and_flood_front_location_at_time_t_table = pd.DataFrame(cumulative_oil_recovered_and_flood_front_location_at_time_t_list)
        #cumulative_oil_recovered_and_flood_front_location_at_time_t_table

        # Cumulative oil recovered from all beds at time t
        Cumulative_oil_recovered_from_all_beds = cumulative_oil_recovered_at_time_t_table + cumulative_oil_recovered_and_flood_front_location_at_time_t_table
        Cumulative_oil_recovered_from_all_beds_table = pd.DataFrame(Cumulative_oil_recovered_from_all_beds).rename(columns = {0:'Cumulative oil recovered from all beds at time t'})
        
        #Cumulative_oil_recovered_from_all_beds_table
        #=========================================================================================================================
        # Vertical coverage at time t
        Vertical_coverage_at_time_t = Cumulative_oil_recovered_from_all_beds/Total_recoverable_oil_in_place
        Vertical_coverage_at_time_t_table = pd.DataFrame(Vertical_coverage_at_time_t).rename(columns = {0:'Vertical coverage at time t'})
        #Vertical_coverage_at_time_t_table

        #=========================================================================================================================
        # Cumumlative water oil ratio for constant injecction rate case.
        Cumumlative_water_oil_ratio_for_CIR = ((Constant_total_injection_rate_for_all_beds*Real_time_CIP_table['Real time for constant injection pressure']) - Cumulative_oil_recovered_from_all_beds_table['Cumulative oil recovered from all beds at time t'])/Cumulative_oil_recovered_from_all_beds_table['Cumulative oil recovered from all beds at time t']
        Cumumlative_water_oil_ratio_for_CIR_table = pd.DataFrame(Cumumlative_water_oil_ratio_for_CIR).rename(columns = {0:'Cumumlative water oil ratio for constant injection rate'})
        #Cumumlative_water_oil_ratio_for_CIR_table
        #=========================================================================================================================
        # Cumumlative water oil ratio for constant injecction Pressure case.
        # First get the product of difference between the real time and the breakthrough time, the bed thickness and water mobility.
        product_1_list  = []
        for j in range(len(Layers)):
            product_1 = (Real_time_CIP_table['Real time for constant injection pressure'].to_numpy() - breakthrough_time_table['Breakthrough time'][j])*bed_data_sort['THICKNESS'][j]*instantaneous_volumetric_flowrate_of_water_table[j].to_numpy()
            product_1_list.append(product_1)
        product_1_table = pd.DataFrame(product_1_list).T
        #product_1_table

        Cumumlative_water_oil_ratio_for_CIP_list = []
        for i in range(len(Real_time_CIP_table)):
            for j in range(len(Layers)):
                if dynamic_bed[i] == j:
                    Cumumlative_water_oil_ratio_for_CIP = ((width_of_bed_ft*Inj_Pressure_differential/Length_of_bed_ft)*product_1_table.iloc[i, 0:j].sum(axis = 0))/Cumulative_oil_recovered_from_all_beds_table['Cumulative oil recovered from all beds at time t'][i]
                    Cumumlative_water_oil_ratio_for_CIP_list.append(Cumumlative_water_oil_ratio_for_CIP)
        Cumumlative_water_oil_ratio_for_CIP_table = pd.DataFrame(Cumumlative_water_oil_ratio_for_CIP_list).rename(columns = {0:'Cumumlative water oil ratio for constant injection pressure'})
        #Cumumlative_water_oil_ratio_for_CIP_table
        #=========================================================================================================================
        # The cumulative water injected into bed i to time t, is given for the Constant injection pressure case by;
        cumulative_water_injected_list = []
        #for i in range(len(Real_time_CIP_table)):
        for j in range(len(Layers)):
            cumulative_water_injected_1 = Flood_front_location_of_other_beds_beyond_breakthrough_table[j]*Ultimate_recoverable_oil_per_bed_table['Ultimate recoverable oil per bed'][j]
            cumulative_water_injected_2 = Ultimate_recoverable_oil_per_bed_table['Ultimate recoverable oil per bed'][j] + 1.1267e-3*(width_of_bed_ft*Inj_Pressure_differential/Length_of_bed_ft)*product_1_table[j]
            #cumulative_water_injected_list_1.append(cumulative_water_injected_1)
            for i in range(len(Real_time_CIP_table)):
                if Real_time_CIP_table['Real time for constant injection pressure'][i] <= breakthrough_time[j]:

                    cumulative_water_injected_list.append(cumulative_water_injected_1)
                    #cumulative_water_injected = Flood_front_location_of_other_beds_beyond_breakthrough_table.iloc[:,j]*Ultimate_recoverable_oil_per_bed_table[0][j]

                else:
                    cumulative_water_injected_list.append(cumulative_water_injected_2)
                break
               # cumulative_water_injected = Ultimate_recoverable_oil_per_bed_table[0][j] + (width_of_bed_ft*Inj_Pressure_differential/Length_of_bed_ft)*product_1_table[j]

        cumulative_water_injected_table = pd.DataFrame(cumulative_water_injected_list).T
        #cumulative_water_injected_table
        Reznik_dataframe_list = [Layer_table1,breakthrough_time_table,Flood_front_position_of_bed_n_j,Ultimate_recoverable_oil_per_bed_table,
                                Front_position_of_other_beds_at_breakthrough_table,
                                flood_front_of_last_bed_table,Real_time_CIP_table,dynamic_bed_table,sum_water_flowrate_before_breakthrough_of_dynamic_bed_table,
                                sum_oil_flowrate_before_breakthrough_of_dynamic_bed_table,Instantaneous_producing_Water_Oil_Ratio_before_breakthrough_of_dynamic_bed_table,
                                Instantaneous_producing_Water_cut_table,cumulative_oil_recovered_at_time_t_table,Cumulative_oil_recovered_from_all_beds_table,
                                Vertical_coverage_at_time_t_table,Cumumlative_water_oil_ratio_for_CIR_table,Cumumlative_water_oil_ratio_for_CIP_table,
                                Flood_front_location_of_other_beds_table,
                                Flood_front_location_of_other_beds_beyond_breakthrough_table,
                                Property_time_table,
                                average_mobility_at_time_t_table,
                                Superficial_filter_velocity_table,
                                actual_linear_velocity_table,
                                instantaneous_volumetric_flowrate_of_water_table,
                                instantaneous_volumetric_flowrate_of_oil_table,
                                cumulative_water_injected_table
                                ]
        translation = {39: None}
        Reznik_checklists = str(Reznik_checklist).translate(translation)
        def variablename(var):
            return [tpl[0] for tpl in filter(lambda x: var is x[1], globals().items())]
        l_r=pd.DataFrame()
        #print(Reznik_checklist)
        for j in Reznik_checklist:
            for d in Reznik_dataframe_list:
                if variablename(d)[0] == j:
                    l_r = l_r.append(d)
                    
        translation = {39: None}
        Reznik_xaxis_dropdowns = str(Reznik_xaxis_dropdown).translate(translation)
        def variablename(var):
            return [tpl[0] for tpl in filter(lambda x: var is x[1], globals().items())]
        l_rxaxis = pd.DataFrame()
        for q in Reznik_dataframe_list:
            if variablename(q)[0] == Reznik_xaxis_dropdown:
                l_rxaxis = l_rxaxis.append(q)
        
        lr = []
        for j in l_r.values.tolist():
            for m in j:
                lr.append(m)

        lrxaxis = []
        for j in l_rxaxis.values.tolist():
            for m in j:
                lrxaxis.append(m)
        #Replace column names with column index

        #The code integrates marker status
        if chart_type == 'line':
            marker = None
        else:
            marker='markers'
        Reznik_Chart = {
            "data": [
                {
                    "x": lrxaxis,
                    "y": lr,
                    "type": str(chart_type),
                    "mode": marker,
                },
            ],
            "layout": {
                "title": {"text": 'Reznik et al.: ' + str(Reznik_checklists) +' vs ' + str(Reznik_xaxis_dropdowns), "x": 0.05, "xanchor": "left"},
                "xaxis": {"title":str(Reznik_xaxis_dropdowns),"fixedrange": True},
                "yaxis": {"title":str(Reznik_checklists),"fixedrange": True},
                "colorway": ["#E12D39"],
            },
        }
        return dash_table.DataTable(
                                columns=[{"name": str(i), "id": str(i)} for i in l_r.columns],
                                data=l_r.to_dict('records'),
                                editable=True,
                                style_header={'backgroundColor': 'rgb(30, 30, 30)'},
                                style_cell={
                                    'backgroundColor': '#54c5f9',
                                    'color': 'white'
                                }
                                ), len(Real_time_CIP_table), Reznik_Chart

        #==========================================================================================================================
        # TABLE OF ALL OBTAINED VALUES.
        
    except ZeroDivisionError:
        #messagebox.showerror("Omission", "Enter a value of Number of points other than zero")
        return None

@app.callback(
    [Output("Robert-table-display", "children"),
    Output("Robert-Chart", "figure")
    ],
    [
        Input("SWI", "value"),
        Input("VISO", "value"),
        Input("VISW", "value"),
        Input("OFVF", "value"),
        Input("length-of-bed", "value"),
        Input("bed-width", "value"),
        Input("CIR", "value"),
        Input("Robert-checklist", "value"),
        Input("chart-type", "value"),
        Input("Robert-xaxis-dropdown", "value"),
    ]
)

def Robert(SWI, VISO, VISW, OFVF, Length_of_bed_ft, width_of_bed_ft,
            Constant_injection_rate, Robert_checklist, chart_type, Robert_xaxis_dropdown):
    global Fractional_flow_table
    global Capacity
    global Fraction_of_total_Capacity
    global Injection_Rate_Per_Layer
    global Cummulative_Water_Injection_Per_Layer_list
    global Oil_Production_Before_Breakthrough
    global Oil_Production_Per_Layer_After_Breakthrough_list
    global Water_Production_Per_Layer_After_Breakthrough_list
    global Recovery_At_Breakthrough_Per_Layer_list
    global Cummulative_Water_Injection_Per_Layer_At_Breakthrough_list
    global Time_To_Breakthrough_For_Each_Layer_list
    global Oil_Recovery_To_Each_Point_List
    global Time_To_Each_Point_list

    SW_table = pd.DataFrame(SW, columns = ['SW'])
    # Using the correlation between relative permeability ratio and water saturation
    #print('Correlation:\n Kro/Krw = aexp(-bSw)\n')

    # Calculating the coefficient b
    b = (np.log((KRO/KRW)[2])-np.log((KRO/KRW)[3]))/(SW[3]-SW[2])
    #print('b is:\n ', b)
    #========================================================================

    # Calculating the coefficient a
    a = (KRO/KRW)[2]*math.exp(b*SW[2])
    #print('a is:\n ', a)
    #========================================================================
    # Calculating the fractional flow
    def fw(SW):
        fw = 1/(1+a*(VISW/VISO)*np.exp(-b*SW))
        return(fw)
    #========================================================================
    ''' To calculate a suitable slope for the tangent to the fractional flow curve
    Drawn from the initial water saturation'''

    ''' STEP1: Generate a list of uniformly distributed random numbers from a water saturation
    # greater than the initial water saturation to 1'''
    xList = []
    for i in range(0, 10000):
        x = random.uniform(SWI+0.1, 1)
        xList.append(x) 
    xs = np.array(xList)

    '''STEP2: Calculate different slopes of tangents or lines intersecting the fractional
    flow curve using the array generated in step 1 as the water saturation.'''
    m = 1/((xs-SWI)*(1+(VISW/VISO)*a*np.exp(-b*xs)))

    '''STEP3: Calculate the maximum slope from different slopes generated in step 2.
    The value of this slope will be the slope of the tangent to the fractional flow
    curve.'''
    tangent_slope=max(m)
    #print('slope of the tangent line is:\n ',tangent_slope)
    #==========================================================================
    # Calculate the breakthrough saturation.
    Saturation_at_Breakthrough = SWI + 1/tangent_slope
    #print('saturation at breakthrough is:\n ', Saturation_at_Breakthrough)
    #===========================================================================
    # Calculating the saturation at the flood front

    def funct(SWF):
        swf = SWF[0]
        F = np.empty((1))
        F[0] = ((tangent_slope*(swf-SWI)*(1+(VISW/VISO)*a*math.exp(-b*swf)))-1)
        return F
    SWF_Guess = np.array([SWI+0.1])
    SWF = fsolve(funct, SWF_Guess)[0]
    #============================================================================
    # Fractional flow at the flood front
    Fwf = fw(SWF)
    #=============================================================================
    # Fractional flow
    Fw = fw(SW)
    Fw_table = pd.DataFrame(Fw, columns = ['Fractional Flow (Fw)'])
    #=============================================================================
    # Calculating the differential of the fractional flow equation
    def dFw_dSw(Sw):
        dfw_dSw = (VISW/VISO)*a*b*np.exp(-Sw*b)/(1+(VISW/VISO)*a*np.exp(-Sw*b))**2
        return dfw_dSw
    dfw_dSw_table = pd.DataFrame(dFw_dSw(SW), columns = ['dFw/dSw'])
    #============================================================================
    # Generating the data for the tangent plot
    tangent = (SW-SWI)*tangent_slope
    tangent_table = pd.DataFrame(tangent, columns = ['Tangent'])
    #==============================================================================
    '''Draw several tangents to the fractional flow curve at Sw values greater than the
    breakthrough saturation. Determine Sw and dFw/dSw and corresponding to these values.
    Plot fw’ versus Sw and construct a smooth curve through the points '''
    # Sw greater than SwBT
    Sw_greater_SwBT = arange(Saturation_at_Breakthrough+0.01,SW[len(SW)-1],0.01)
    dFw_dSw_greater_SwBT = dFw_dSw(Sw_greater_SwBT)
    #============================================================================
    Fractional_flow_table = pd.concat([SW_table, Fw_table, dfw_dSw_table, tangent_table], axis=1)
    #=============================================================================
            
    
    # class MainWindow(QtWidgets.QMainWindow):
    
    #     def __init__(self, *args, **kwargs):
    #         super(MainWindow, self).__init__(*args, **kwargs)
    
    #         self.graphWidget = pg.PlotWidget()
    #         self.setCentralWidget(self.graphWidget)
    
    #         #Add Background colour to white
    #         self.graphWidget.setBackground('w')
    #         # Add Title
    #         self.graphWidget.setTitle("Fractional Flow Curve", color="b", size="20pt")
    #         # Add Axis Labels
    #         styles = {"color": "#f00", "font-size": "18px"}
    #         self.graphWidget.setLabel("left", "Fractional Flow (Fw)", **styles)
    #         self.graphWidget.setLabel("right", "Differential of Fractional Flow (dFw/dSw)", **styles)
    #         self.graphWidget.setLabel("bottom", "Water Saturation (Sw)", **styles)
    #         #Add legend
    #         self.graphWidget.addLegend()
    #         #Add grid
    #         self.graphWidget.showGrid(x=True, y=True)
    #         #Set Range
    #         self.graphWidget.setXRange(0, 1, padding=0)
    #         self.plot(SW, fw(SW), "Fw", 'r')
    #         self.plot(SW, tangent, "Tangent", 'k')
    #         self.plot(SW, dfw_dSw, "dFw/dSw", 'b')

    
    #     def plot(self, x, y, plotname, color):
    #         pen = pg.mkPen(color=color)
    #         self.graphWidget.plot(x, y, name=plotname, pen=pen, symbolBrush=(color))
    
    # def main():
    #     app = QtWidgets.QApplication(sys.argv)
    #     main = MainWindow()
    #     main.show()
    #     main._exit(app.exec_())
    #     #QApplication.exec_()
    # if __name__ == '__main__':
    #     main()
            

    #=============================================================================
    #Calculating capacity 
    #Bed_data = pd.read_csv("C:/Users/ELAKUTO/jews feastival/Desktop/AUST MSC PET/Thesis/Permeability_Porosity_distribution_data.csv")
    permeability = bed_data['PERMEABILITY']
    thickness = bed_data['THICKNESS']
    porosity = bed_data['POROSITY']
    Capacity=permeability*thickness
    Fraction_of_total_Capacity= Capacity/sum(Capacity)
    #=============================================================================
    #Calculating Injection Rate per layer
    Injection_Rate_Per_Layer = Constant_injection_rate*Fraction_of_total_Capacity

    #=============================================================================
    #Calculating Water injection rate per layer
    Area = Length_of_bed_ft*width_of_bed_ft/43560

    Cummulative_Water_Injection_Per_Layer_list = []
    for j in range(len(thickness)):
        Cummulative_Water_Injection_Per_Layer = 7758*Area*thickness[j]*porosity[j]/dFw_dSw_greater_SwBT
        Cummulative_Water_Injection_Per_Layer_list.append(Cummulative_Water_Injection_Per_Layer)

    #=============================================================================
    #Oil Production Rate Before Breakthrough
    Oil_Production_Before_Breakthrough = Injection_Rate_Per_Layer/OFVF

    #Oil Production Rate After Breakthrough
    Oil_Production_Per_Layer_After_Breakthrough_list = []
    for j in range(len(thickness)):
        Oil_Production_Per_Layer_After_Breakthrough = Oil_Production_Before_Breakthrough[j]*(1-Fw)
        Oil_Production_Per_Layer_After_Breakthrough_list.append(Oil_Production_Per_Layer_After_Breakthrough)

    #Water Production 
    Water_Production_Per_Layer_After_Breakthrough_list = []
    for j in range(len(thickness)):
        Water_Production_Per_Layer_After_Breakthrough = Injection_Rate_Per_Layer[j]*Fw
        Water_Production_Per_Layer_After_Breakthrough_list.append(Water_Production_Per_Layer_After_Breakthrough)    

    #Calculate the recovery at breakthrough and the time to breakthrough for each layer
    Recovery_At_Breakthrough_Per_Layer_list = []
    for j in range(len(thickness)):    
        Recovery_At_Breakthrough_Per_Layer = 7758*Area*thickness[j]*porosity[j]*(Saturation_at_Breakthrough - SWI)/OFVF
        Recovery_At_Breakthrough_Per_Layer_list.append(Recovery_At_Breakthrough_Per_Layer)
    #print(Recovery_At_Breakthrough_Per_Layer_list)   
    #Time to Breakthrough for each layer
    #Water injection at Breakthrough
    Cummulative_Water_Injection_Per_Layer_At_Breakthrough_list = []
    for j in range(len(thickness)):
        Cummulative_Water_Injection_Per_Layer_At_Breakthrough = 7758*Area*thickness[j]*porosity[j]/dFw_dSw(Saturation_at_Breakthrough)
        Cummulative_Water_Injection_Per_Layer_At_Breakthrough_list.append(Cummulative_Water_Injection_Per_Layer_At_Breakthrough)   

    Time_To_Breakthrough_For_Each_Layer_list = []
    for j in range(len(thickness)):
        Time_To_Breakthrough_For_Each_Layer = Cummulative_Water_Injection_Per_Layer_At_Breakthrough/Injection_Rate_Per_Layer[j]
        Time_To_Breakthrough_For_Each_Layer_list.append(Time_To_Breakthrough_For_Each_Layer)    

    #Oil recovery and time to each point.
    Oil_Recovery_To_Each_Point_List = []
    for j in range(len(thickness)):
        Oil_Recovery_To_Each_Point =  7758*Area*thickness[j]*porosity[j]*(SW - SWI)/OFVF
        Oil_Recovery_To_Each_Point_List.append(Oil_Recovery_To_Each_Point)

    Time_To_Each_Point_list = []
    for j in range(len(thickness)):    
        Time_To_Each_Point = Cummulative_Water_Injection_Per_Layer/Injection_Rate_Per_Layer[j]
        Time_To_Each_Point_list.append(Time_To_Each_Point)

    Capacity = pd.DataFrame(Fraction_of_total_Capacity).rename(columns={0: 'Capacity'})
    Fraction_of_total_Capacity = pd.DataFrame(Fraction_of_total_Capacity).rename(columns={0: 'Fraction of Total Capacity'})
    Injection_Rate_Per_Layer=pd.DataFrame(Injection_Rate_Per_Layer).rename(columns = {0:'Injection Rate per Layer'})
    Oil_Production_Before_Breakthrough = pd.DataFrame(Injection_Rate_Per_Layer).rename(columns={0: 'Oil Production Before Breakthrough'})
    Recovery_At_Breakthrough_Per_Layer_list = pd.DataFrame(Recovery_At_Breakthrough_Per_Layer_list).rename(columns={0: 'Recovery At Breakthrough Per Layer'})
    Cummulative_Water_Injection_Per_Layer_At_Breakthrough_list=pd.DataFrame(Cummulative_Water_Injection_Per_Layer_At_Breakthrough_list).rename(columns = {0:'Cumulative Water Injection Per Layer At Breakthrough'})
    Time_To_Breakthrough_For_Each_Layer_list = pd.DataFrame(Time_To_Breakthrough_For_Each_Layer_list).rename(columns={0: 'Time To Breakthrough For Each Layer'})
    

    Robert_data_list = [Fractional_flow_table, Capacity, Fraction_of_total_Capacity, Injection_Rate_Per_Layer,
    Cummulative_Water_Injection_Per_Layer_list, Oil_Production_Before_Breakthrough, Oil_Production_Per_Layer_After_Breakthrough_list,
    Water_Production_Per_Layer_After_Breakthrough_list, Recovery_At_Breakthrough_Per_Layer_list,
    Cummulative_Water_Injection_Per_Layer_At_Breakthrough_list, Time_To_Breakthrough_For_Each_Layer_list,
    Oil_Recovery_To_Each_Point_List, Time_To_Each_Point_list]
    
    translation = {39: None}
    Robert_checklists = str(Robert_checklist).translate(translation)
    def variablename(var):
        return [tpl[0] for tpl in filter(lambda x:var is x[1], globals().items())]
    
    #Collecting DataFrame for Table
    l_b=pd.DataFrame()
    for j in Robert_checklist:
        for d in Robert_data_list:
            if variablename(d)[0] == j:
                l_b = l_b.append(d,ignore_index=True)
    #print(l_b)      
    translation = {39: None}
    Robert_xaxis_dropdowns = str(Robert_xaxis_dropdown).translate(translation)

    #Collecting Data for graph x-axis
    l_bxaxis = pd.DataFrame()
    for q in Robert_data_list:
        if variablename(q)[0] == Robert_xaxis_dropdowns:
            l_bxaxis=l_bxaxis.append(q,ignore_index=True)

    #Collecting Data for graph y-axis
    lb = []
    for j in l_b.values.tolist():
        for m in j:
            lb.append(m)
    #print(lb)
    lbxaxis = []
    for j in l_bxaxis.values.tolist():
        for m in j:
            lbxaxis.append(m)
    #print(lbxaxis)

    #The code integrates marker status
    if chart_type == 'line':
        marker = None
    else:
        marker='markers'
    Robert_Chart = {
        "data": [
            {
                "x": lbxaxis,
                "y": lb,
                "type": str(chart_type),
                "mode": marker,
            },
        ],
        "layout": {
            "title": {"text": 'Robert: ' + str(Robert_checklists) +' vs ' + str(Robert_xaxis_dropdowns), "x": 0.05, "xanchor": "left"},
            "xaxis": {"title":str(Robert_xaxis_dropdowns),"fixedrange": True},
            "yaxis": {"title":str(Robert_checklists),"fixedrange": True},
            "colorway": ["#E12D39"],
        },
    }
    return dash_table.DataTable(
                            columns=[{"name": str(i), "id": str(i)} for i in l_b.columns],
                            data=l_b.to_dict('records'),
                            editable=True,
                            style_header={'backgroundColor': 'rgb(30, 30, 30)'},
                            style_cell={
                                'backgroundColor': '#54c5f9',
                                'color': 'white'
                            }
                            ),Robert_Chart
                                


if __name__ == "__main__":
    app.run_server(debug=False)
