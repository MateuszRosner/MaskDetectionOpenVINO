import plotly
import pandas as pd
import numpy as np
import os

from datetime import datetime
from plotly.graph_objects import Bar, Pie, Histogram
from plotly import tools
from plotly import subplots


daily_rep_path = datetime.now().strftime("%Y_%m_%d")
path = os.getcwd()
path = os.path.join(path, 'reports', '2020_11_03')

data = pd.read_csv(os.path.join(path, 'stats.csv'), sep=';')

occupation_rep = Histogram(x=data["Time"], y=data["Count"], marker_color='lightcoral', name='Occupancy', opacity=0.7)
sex_rep = Bar(x=[6,3], y=['Females', 'Males'], width=0.7, marker_color='cadetblue', orientation='h', name='Sex')
age_rep = Histogram(x=data["Age"],  name='Age', opacity=0.7, xbins=dict(start=0, end=100, size=1))
age_time_rep = Bar(x=data["Time"], y=data["Age"])
viola_rep = Pie(values=[24, 36], labels=['Mask', 'No mask'], hole=0.3, name='Violations')

data_to_plot = subplots.make_subplots(rows=3, cols=2, subplot_titles=['Occupancy', 'Sex', 'Age', 'Violations'], 
                                      specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "domain"}]])

data_to_plot.append_trace(occupation_rep, 1, 1)
data_to_plot.append_trace(sex_rep, 1, 2)
data_to_plot.append_trace(age_rep, 2, 1)
data_to_plot.append_trace(age_time_rep, 2, 2)
data_to_plot.append_trace(viola_rep, 3, 2)

data_to_plot.update_layout(
    bargap=0.1, # gap between bars of adjacent location coordinatesbargroupgap=0.1 # gap between bars of the same location coordinates
)

plotly.offline.plot(data_to_plot, filename=os.path.join(path, "daily_report.html"))