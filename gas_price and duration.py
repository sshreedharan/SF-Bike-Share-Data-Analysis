import numpy as np
import pandas as pd
import time
import holoviews as hv
import bokeh
import matplotlib.pyplot as plt
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

def date_range_convert(x):
    '''
    covert date format from "%Y-%m-%d %H:%M:%S" to "%m/%d/%Y"
    :param x: timestamp
    :return: str
    '''
    date = time.strftime("%m/%d/%Y", time.strptime(str(x), "%Y-%m-%d %H:%M:%S"))
    return date

def inv_rental_num_date_format(x):
    '''
    covert date format from "%Y/%m/%d" to "%m/%d/%Y"
    :param x: str
    :return: str
    '''
    date = time.strftime("%m/%d/%Y", time.strptime(x, "%Y/%m/%d"))
    return date

def trip_date_convert(x):
    '''
    covert date format from "%m/%d/%Y %H:%M" to "%H:%M"
    :param x: str
    :return: str
    '''
    date = time.strftime("%H:%M", time.strptime(x, "%m/%d/%Y %H:%M"))
    return date

def visualization_gas_price():
    '''
    this function is used to visualize the relationship between Number of Rentals and Gas Price
    :return: None
    '''
    gas_price_df = pd.read_csv('Weekly_San_Francisco_CA_Regular_Reformulated_Retail_Gasoline_Prices.csv')
    gas_price = pd.Series(list(gas_price_df.loc[:,'price']),index=gas_price_df.loc[:,'date'].map(weather_date_format))
    dates = (pd.date_range('20130826', periods=736)).map(date_range_convert)
    gas_price_everyday = gas_price.reindex(dates).interpolate().iloc[3:]
    rental_num = pd.read_csv('rental_count_everyday.csv')
    rental_num.loc[:,'date'] = rental_num.loc[:,'date'].map(rental_num_date_format)
    sort_rental_num = rental_num.sort_values(by=['date'])
    sort_rental_num.loc[:,'date'] = sort_rental_num.loc[:,'date'].map(inv_rental_num_date_format)
    rental_num = pd.Series(data=list(sort_rental_num.loc[:,'id']), index=list(sort_rental_num.loc[:,'date']))
    gas_rental = pd.DataFrame(data={'rental_num':list(rental_num),'gas_price':list(gas_price_everyday)}, index=gas_price_everyday.index)
    plot1 = hv.Scatter(gas_rental, 'gas_price', 'rental_num')\
        .options(width=1000,height=500,color='#1f77b4',size=2, title="Relatonship between Number of Rentals and Gas Price (Dollars per Gallon)")
    bokeh.plotting.show(hv.render(plot1))

def visualization_duration():
    '''
    this function is used to visualize the relationship between Start Time and Duration
    :return: None
    '''

    plt.style.use("bmh")
    trip = pd.read_csv('trip.csv')
    trip.loc[:, 'start_time'] = trip.loc[:, 'start_date'].map(trip_date_convert)
    trip = trip.loc[:5000].sort_values(by=['start_time'])
    my_x_ticks = np.arange(0, 1000, 110)
    plt.figure(1, figsize=(16, 6), dpi=100)
    plt.xticks(my_x_ticks)
    plt.xlabel("start_time")
    plt.ylabel("duration")
    plt.plot(trip[trip.duration < 10000].loc[:, 'start_time'], trip[trip.duration < 10000].loc[:, 'duration'], 'o', markersize=0.5)
    plt.title('Relationship between Start Time and Duration')
    plt.show()

if __name__ == '__main__':
    '''
    Datasets and code should be put in the same folder
    '''
    visualization_gas_price()
    # visualization_duration()