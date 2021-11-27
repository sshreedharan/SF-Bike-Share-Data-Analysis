import pandas as pd
import time
import holoviews as hv
import bokeh
hv.extension('bokeh', 'matplotlib')

def weather_date_format(x):
    '''
    covert date format from "%m/%d/%Y" to "%m/%d/%Y"
    :param x: str
    :return: str
    '''
    date = time.strftime("%m/%d/%Y", time.strptime(x, "%m/%d/%Y"))
    return date

def rental_num_date_format(x):
    '''
    covert date format from "%m/%d/%Y" to "%Y/%m/%d"
    :param x: str
    :return: str
    '''
    date = time.strftime("%Y/%m/%d", time.strptime(x, "%m/%d/%Y"))
    return date

def inv_rental_num_date_format(x):
    '''
    covert date format from "%Y/%m/%d" to "%m/%d/%Y"
    :param x: str
    :return: str
    '''
    date = time.strftime("%m/%d/%Y", time.strptime(x, "%Y/%m/%d"))
    return date

def preprocessing_csv():
    '''
    preprocess to generate a new csv that contains number of rentals per day
    :return: None
    '''
    def date_preserve(date_time):
        date = time.strftime("%m/%d/%Y", time.strptime(date_time, "%m/%d/%Y %H:%M"))
        return date
    df = pd.read_csv('dataset_bike_share/trip.csv')
    df.loc[:, 'date'] = df.start_date.map(date_preserve)
    df_grp = df.groupby('date').count()
    df_grp.loc[:, 'id'].to_csv('dataset_bike_share/rental_count_everyday.csv')

def visualization_weather():
    '''
    this function is used to visualize the relationship between Number of Rentals and Weather Factors
    :return: None
    '''
    weather = pd.read_csv('dataset_bike_share/weather.csv')
    weather.loc[:,'date'] = weather.date.map(weather_date_format)
    rental_num = pd.read_csv('dataset_bike_share/rental_count_everyday.csv')
    rental_num.loc[:,'date'] = rental_num.loc[:,'date'].map(rental_num_date_format)
    sort_rental_num = rental_num.sort_values(by=['date'])
    sort_rental_num.loc[:,'date'] = sort_rental_num.loc[:,'date'].map(inv_rental_num_date_format)
    rental_num = pd.Series(data=list(sort_rental_num.loc[:,'id']), index=list(sort_rental_num.loc[:,'date']))

    new_weather = weather[weather.zip_code==94107]
    new_weather.loc[:, 'num_rental'] = list(rental_num)
    print(new_weather)
    plot1 = hv.Scatter(new_weather, 'mean_temperature_f', 'num_rental')\
        .options(width=800,height=500,color='r',size=2, title="Relatonship between Number of Rentals and mean_temperature (in Farenheit)")
    plot2 = hv.Scatter(new_weather, 'mean_dew_point_f', 'num_rental')\
        .options(width=800,height=500,color='b',size=2, title="Relatonship between Number of Rentals and mean_dew_point")
    plot3 = hv.Scatter(new_weather, 'mean_humidity', 'num_rental')\
        .options(width=800,height=500,color='g',size=2, title="Relatonship between Number of Rentals and mean_humidity")
    plot4 = hv.Scatter(new_weather, 'mean_sea_level_pressure_inches', 'num_rental')\
        .options(width=800,height=500,color='b',size=2, title="Relatonship between Number of Rentals and mean_sea_level_pressure_inches")
    plot5 = hv.Scatter(new_weather, 'mean_visibility_miles', 'num_rental')\
        .options(width=800,height=500,color='b',size=2, title="Relatonship between Number of Rentals and mean_visibility_miles")
    plot6 = hv.Scatter(new_weather, 'mean_wind_speed_mph', 'num_rental')\
        .options(width=800,height=500,color='b',size=2, title="Relatonship between Number of Rentals and mean_wind_speed_mph")
    plot7 = hv.Scatter(new_weather, 'cloud_cover', 'num_rental')\
        .options(width=800,height=500,color='b',size=2, title="Relatonship between Number of Rentals and cloud_cover")
    plot8 = hv.Scatter(new_weather[new_weather.precipitation_inches!='T'].sort_values(by=['precipitation_inches']), 'precipitation_inches', 'num_rental')\
        .options(width=1000,height=500,color='b',size=2, title="Relatonship between Number of Rentals and precipitation_inches")

    bokeh.plotting.show(hv.render(plot1))
    bokeh.plotting.show(hv.render(plot2))
    bokeh.plotting.show(hv.render(plot3))
    bokeh.plotting.show(hv.render(plot4))
    bokeh.plotting.show(hv.render(plot5))
    bokeh.plotting.show(hv.render(plot6))
    bokeh.plotting.show(hv.render(plot7))
    bokeh.plotting.show(hv.render(plot8))


if __name__ == '__main__':
    '''
    Datasets should be put in the 'dataset_bike_share' folder
    '''
    # preprocessing_csv()  # This function only needs to be run the first time.
    visualization_weather()
