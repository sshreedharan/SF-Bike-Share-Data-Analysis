import pandas as pd
from os.path import exists as file_exists
import pickle

def skip(index):
    '''
    To read every 60th value from the status.csv file i.e to get the status at every hour.

    Parameters
    ----------
    index : int
        The index of the status.csv file.

    Returns
    -------
    bool
        If true, the row is not read into the dataframe, else, it is read.
    '''
    
    assert isinstance(index, int) and index >= 0, "The index passed is not a positive integer."
    
    if (index == 1):
        return False
    else:     
        if (index % 60 == 0):
            return False
        else:
            return True

def read_cache_df(file_name, skip_rows = False):
    '''
    To check if a pickle file is available for the corresponding csv file and load it as a dataframe,
    else read the csv file as dataframe and store it in pickle format.
    
    Parameters
    ----------
    file_name : string
        This is the input csv filename.
    skip_rows : bool
        If true, the skip rows logic is called. The default is False.

    Returns
    -------
    df : DataFrame
        The output dataframe for the corresponding csv file.
    '''
    
    assert isinstance(file_name, str), "The input filename is not a string."
    assert isinstance(skip_rows, bool), "The skip_rows argument is not boolean."
    
    if file_exists('./dataset_bike_share/cached_{}_dataframe.pkl'.format(file_name)):
        return(pd.read_pickle("./dataset_bike_share/cached_{}_dataframe.pkl".format(file_name)))
    else:
        if skip_rows:
            df = pd.read_csv('./dataset_bike_share/{}.csv'.format(file_name), skiprows = lambda x : skip(x))
        else:
            df = pd.read_csv('./dataset_bike_share/{}.csv'.format(file_name))
        df.to_pickle("./dataset_bike_share/cached_{}_dataframe.pkl".format(file_name))
        return df

def clean_data(file1,file2,file3,file4):
    '''
    To create dataframe for each file (station, status, trip, weather) and remove the rows and columns which has Null values. 
    Also, the unwanted columns are dropped from the dataframes.
    A new events dataframe is created by taking the dates and corresponding events from the weather dataframe.
    A station id dictionary is also created which contains station id as keys, and station name and city as its values. 

    Parameters
    ----------
    file1 : string
        This is the first input file (station).
    file2 : string
        This is the second input file (status).
    file3 : string
        This is the third input file (trip).
    file4 : string
        This is the fourth input file (weather).

    Returns
    -------
    station_df_new : DataFrame
        This is the dataframe of the station file after cleaning.
    status_df_new : DataFrame
        This is the dataframe of the status file after cleaning.
    trip_df_new : DataFrame
        This is the dataframe of the trip file after cleaning.
    weather_df_new : DataFrame
        This is the dataframe of the weather file after cleaning.
    events_df_new : DataFrame
        This dataframe contains dates and the corresponding weather events like rain, fog, etc.
    station_id_dict : dict
        This dictionary contains the station_id as keys, and the station name and city as its corresponding values.
    '''
    
    assert isinstance(file1, str) and isinstance(file2, str) and isinstance(file3, str) and isinstance(file4, str), "The input filenames are not strings."
    
    station_df = read_cache_df(file1)
    status_df = read_cache_df(file2, True)
    trip_df = read_cache_df(file3)
    weather_df = read_cache_df(file4)
    
    weather_df['date'] = pd.to_datetime(weather_df['date'])
    events_df = weather_df[['date', 'events']]   
    events_df_new = events_df.dropna()
    weather_df_new = weather_df.drop(['events','max_gust_speed_mph'], axis = 1).dropna()
    
    station_id_df = station_df[['id', 'name', 'city']]
    station_id_dict = station_id_df.set_index('id').T.to_dict('list')
    station_df_new = station_df.drop(['name','city','installation_date'], axis = 1).dropna()
    station_df_new = station_df_new.rename(columns={'id':'station_id'})
    
    status_df['time'] = pd.to_datetime(status_df['time'])
    status_df['date'] = pd.to_datetime(status_df['time'].dt.date)
    status_df['time'] = status_df['time'].dt.time
    status_df_new = status_df.dropna()
    
    trip_df_new = trip_df.drop(['id','start_station_name','end_station_name','bike_id'], axis = 1).dropna()
    
    return station_df_new, status_df_new, trip_df_new, weather_df_new, events_df_new, station_id_dict
    
def merge_data(file1 = 'station', file2 = 'status', file3 = 'trip', file4 = 'weather'):
    '''
    To merge the cleaned station and status dataframe by the station id, and the resultant dataframe
    is merged with weather dataframe based on date. The final merged dataframe is stored in pickle format,
    along with the cleaned trip and events dataframe, and the station id dictionary.

    Parameters
    ----------
    file1 : string
        This is the first input file. The default is 'station'.
    file2 : string
        This is the second input file. The default is 'status'.
    file3 : string
        This is the third input file. The default is 'trip'.
    file4 : string
        This is the fourth input file. The default is 'weather'.

    Returns
    -------
    None.
    '''
    
    assert isinstance(file1, str) and isinstance(file2, str) and isinstance(file3, str) and isinstance(file4, str), "The input filenames are not strings."
    
    station_df_new, status_df_new, trip_df_new, weather_df_new, events_df_new, station_id_dict = clean_data(file1,file2,file3,file4)
    
    #Use sqlite3 if possible
    df_new = pd.merge(status_df_new,station_df_new,on='station_id',how='left')
    df_new = pd.merge(df_new,weather_df_new,on='date',how='left')
    
    df_new.to_pickle("./dataset_bike_share/cached_dataframe.pkl")
    trip_df_new.to_pickle("./dataset_bike_share/cached_trip_dataframe_new.pkl")
    events_df_new.to_pickle("./dataset_bike_share/cached_events_dataframe_new.pkl")
    with open('./dataset_bike_share/station_id_dict.pkl', 'wb') as file:
        pickle.dump(station_id_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

