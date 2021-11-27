import pandas as pd
import numpy as np
import pickle
from os.path import exists as file_exists

from pandas.core.frame import DataFrame
from preprocessing import merge_data
from bokeh.io import curdoc, output_notebook
from bokeh.plotting import figure, ColumnDataSource, show
from bokeh.tile_providers import get_provider
from bokeh.palettes import PRGn, RdYlGn, Category20
from bokeh.transform import linear_cmap
from bokeh.layouts import column
from bokeh.models import ColorBar, NumeralTickFormatter
from bokeh.models import Select
from datetime import time

# To run this file and visualize the grographic map,
# start a bokeh server and pass the file name from command line.
# bokeh serve --show map_visuals.py

def read_data(file):
    '''
    To read data from corresponding pickle file if available or create a pickle file for the csv file and return a DataFrame of the same.

    Parameters
    ----------
    file : str
        The input file name.

    Returns
    -------
    DataFrame
        This the output DataFrame of the file to be read.
    '''

    assert isinstance(file, str), "The input file name is not a string"

    if (file_exists(file)):
        print("Reading file from Cache.")

    else:
        print("Reading and Storing Data.")
        merge_data()
        print("Data Stored as pickle.")

    print("Reading dataframe from pickle file.")
    df = pd.read_pickle(file)
    print("Data read.")
    return df

def read_station_id_dict(file):
    '''
    To read data from station id dictionary pickle file if available or create a pickle file and return a dictionary of the same.

    Parameters
    ----------
    file : str
        The input file name.

    Returns
    -------
    dcitionary
        This the output dictionary containing station id, name and city.
    '''

    assert isinstance(file, str), "The input file name is not a string"

    if (file_exists(file)):
        print("Reading station ID dictionary file from Cache.")

    else:
        print("Reading and Storing Data.")
        merge_data()
        print("Data Stored as pickle.")

    with open(file, 'rb') as f:
        station_id_dict = pickle.load(f)
    print("Data Read.")
    return station_id_dict

def mercator_coor(x,y):
    '''
    To convert latitude and longitude to mercator coordinates.

    Parameters
    ----------
    x : float
        The input latitutde.

    y : float
        The input longitude

    Returns
    -------
    tuple
        This the output mercator x and y coordinates.
    '''

    assert isinstance(x, float) and isinstance(y, float), "The input latitude and longitude values are not float values."

    lat = x
    lon = y

    r_major = 6378137.000
    x = r_major * np.radians(lon)
    scale = x/lon
    y = 180.0/np.pi * np.log(np.tan(np.pi/4.0 +
        lat * (np.pi/180.0)/2.0)) * scale
    return (x, y)

def coordinate_dataframe(df):
    '''
    To add mercator x and y coordinates as columns to the given DataFrame corresponding to the latitude and longitude values.

    Parameters
    ----------
    df : DataFrame
        This is the input dataframe to which the mercator coordinate columns need to be added.

    Returns
    -------
    DataFrame
        This is the output dataframe with mercator x and y coordinates as columns.
    '''

    assert isinstance(df, DataFrame), "The input is not a DataFrame"

    df['coordinates'] = list(zip(df['lat'], df['long']))
    mercators = [mercator_coor(x,y) for x,y in df['coordinates']]
    df['mercator'] = mercators
    df[['mercator_x', 'mercator_y']] = df['mercator'].apply(pd.Series)
    return df

def station_map_visual():
    '''
    To visualize the locations of the stations on an interactive map and reprsent the dock count of each station with color and size.
    The station file is read and the latitude and longitude values are converted to mercator coordinates,
    and is plotted as points on a geographic map.
    '''

    bike_share_df = read_data('./dataset_bike_share/cached_station_dataframe.pkl')
    bike_share_df = bike_share_df.dropna()
    print("Creating coordinate DataFrame.")
    coord_df = coordinate_dataframe(bike_share_df)
    coord_df['point_size'] = coord_df['dock_count']

    print("Creating Map.")
    chosentile = get_provider('CARTODBPOSITRON_RETINA')
    palette = Category20[17]
    source = ColumnDataSource(data=coord_df)
    color_mapper = linear_cmap(field_name = 'dock_count', palette = palette, low = coord_df['dock_count'].min(), high = coord_df['dock_count'].max())

    tooltips = [("Station Name","@name"), ("City","@city"), ("Dock Count","@dock_count")]
    print("Parameters are set.")

    print("Plotting figure.")
    p = figure(title = 'Bay Area Bike Share Map',
               x_axis_type="mercator", y_axis_type="mercator",
               x_axis_label = 'Longitude', y_axis_label = 'Latitude', tooltips = tooltips)
    p.add_tile(chosentile)
    print("Plotting points on Map.")
    p.circle(x = 'mercator_x', y = 'mercator_y', color = color_mapper, source=source, size='point_size', fill_alpha = 0.5)

    print("Creating color bar for reference.")
    color_bar = ColorBar(color_mapper=color_mapper['transform'],
                     formatter = NumeralTickFormatter(format='0.0[0000]'),
                     label_standoff = 13, width=17, location=(0,0))
    p.add_layout(color_bar, 'right')
    print("Color bar created.")

    curdoc().add_root(column(p))
    output_notebook()
    show(p)

def value_conversion(select_val):
    '''
    To convert the time option from the select widget to corresponding column values in the dataframe.

    Parameters
    ----------
    select_val : string
        This is the option which is selected in the widget.

    Returns
    -------
    time_val : int
        This is the corresponding time value in the hour column of the dataframe.
    '''

    if select_val == "12 AM":
        time_val = 0
    elif select_val == "4 AM":
        time_val = 4
    elif select_val == "8 AM":
        time_val = 8
    elif select_val == "12 PM":
        time_val = 12
    elif select_val == "4 PM":
        time_val = 16
    elif select_val == "8 PM":
        time_val = 20
    return time_val

def seasonal_bike_availability():
    '''
    To visualize the availabilty of bikes in different stations on an intereactive map based on the season and hour of the day.
    The Select widgets are used to choose the season and hour, and the average bike availablity at a specific hour for each station during
    Fall, Winter, Spring, and Summer over the years 2013 - 2015 are plotted. From this, we can infer where and when to add more bikes or remove bikes.
    '''

    status_df = read_data('./dataset_bike_share/cached_status_dataframe.pkl')
    station_df = read_data('./dataset_bike_share/cached_station_dataframe.pkl')
    station_df = station_df.drop(['installation_date', 'dock_count'],axis = 1).dropna()
    station_df = station_df.rename(columns={'id':'station_id'})

    status_df['time'] = pd.to_datetime(status_df['time'])
    status_df = status_df.drop(['docks_available'], axis=1).dropna()

    status_df = status_df.loc[((status_df['time'].dt.time >= time(00,00,00)) & (status_df['time'].dt.time < time(1,00,00)))
        | ((status_df['time'].dt.time >= time(4,00,00)) & (status_df['time'].dt.time < time(5,00,00)))
        | ((status_df['time'].dt.time >= time(8,00,00)) & (status_df['time'].dt.time < time(9,00,00)))
        | ((status_df['time'].dt.time >= time(12,00,00)) & (status_df['time'].dt.time < time(13,00,00)))
        | ((status_df['time'].dt.time >= time(16,00,00)) & (status_df['time'].dt.time < time(17,00,00)))
        | ((status_df['time'].dt.time >= time(20,00,00)) & (status_df['time'].dt.time < time(21,00,00)))]

    status_df['hour'] = status_df['time'].dt.hour

    seasons = ['Fall', 'Winter', 'Spring', 'Summer']
    season_conditions = [(status_df['time'].dt.month >= 9) & (status_df['time'].dt.month <=11),
    (status_df['time'].dt.month == 12) | (status_df['time'].dt.month <=2),
    (status_df['time'].dt.month >= 3) & (status_df['time'].dt.month <=5),
    (status_df['time'].dt.month >= 6) & (status_df['time'].dt.month <=8)]

    status_df['season'] = np.select(season_conditions, seasons)

    df = status_df.groupby(['station_id', 'hour', 'season']).mean()
    df = df.astype({"bikes_available" : int})
    df.reset_index(level=[0,1,2], inplace = True)

    status_df_new = pd.merge(df,station_df,on='station_id',how='left')

    coord_df = coordinate_dataframe(status_df_new)
    coord_df['point_size'] = 2 * coord_df['bikes_available']

    source_df = coord_df.loc[(coord_df['season'] == "Fall") & (coord_df['hour'] == 0)]
    source = ColumnDataSource(data=source_df)

    select1 = Select(title="Season:", value="Fall", options=["Fall", "Winter", "Spring", "Summer"])
    select2 = Select(title="Time", value="12 AM", options=["12 AM", "4 AM", "8 AM", "12 PM", "4 PM", "8 PM"])

    def update1(attr,old,new):
        '''
        To update the dataframe corresponding to the season that has been chosen in the widget.
        '''

        time_val = value_conversion(select2.value)
        updated_df = coord_df.loc[(coord_df['season'] == select1.value) & (coord_df['hour'] == time_val)]
        source.data.update(updated_df)

    def update2(attr,old,new):
        '''
        To update the dataframe corresponding to the time that has been chosen in the widget.
        '''

        time_val = value_conversion(select2.value)
        updated_df = coord_df.loc[(coord_df['season'] == select1.value) & (coord_df['hour'] == time_val)]
        source.data.update(updated_df)

    select1.on_change('value', update1)
    select2.on_change('value', update2)

    print("Creating Map.")
    chosentile = get_provider('CARTODBPOSITRON_RETINA')
    palette = Category20[17]
    color_mapper = linear_cmap(field_name = 'bikes_available', palette = palette, low = source_df['bikes_available'].min(), high = source_df['bikes_available'].max())

    tooltips = [("Station Name","@name"), ("City","@city"), ("Bikes Available","@bikes_available")]
    print("Parameters are set.")

    print("Plotting figure.")
    p = figure(title = 'Bike Availability Map',
               x_axis_type='mercator', y_axis_type='mercator',
               x_axis_label = 'Longitude', y_axis_label = 'Latitude', tooltips = tooltips)
    p.add_tile(chosentile)
    print("Plotting points on Map.")

    p.circle(x = 'mercator_x', y = 'mercator_y', color = color_mapper, source=source, size= 'point_size', fill_alpha = 0.5)

    print("Creating color bar for reference.")
    color_bar = ColorBar(color_mapper=color_mapper['transform'],
                     formatter = NumeralTickFormatter(format='0.0[0000]'),
                     label_standoff = 13, width=17, location=(0,0))
    p.add_layout(color_bar, 'right')
    print("Color bar created.")

    curdoc().add_root(column(select1, select2, p))
    output_notebook()
    show(p)

station_map_visual()
seasonal_bike_availability()
