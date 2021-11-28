'''
ECE 143 Final Project - Number of Rentals
Author: Aditi Anand
Created on: 20 Nov 2021
Description: Use 'trip.csv' to count number of rentals per month / year and visualize the data.
'''

from numpy.lib.arraysetops import isin
import pandas as pd
import matplotlib.pyplot as plt

def add_month_yr(x, timestamp_col):
    """Use the given timestamp column of a pandas DataFrame object and add a new 'month-yr' column.

    Args:
        x (pandas DataFrame): Input DataFrame object to add 'month-yr' column to.
        timestamp_col (str): Name of column in x to extract time information from.

    Returns:
        pandas DataFrame: x with the new 'month-yr' column added. Formatted as Jan-2000.
    """

    assert isinstance(x, pd.DataFrame), 'Input data must be a pandas DataFrame object.'
    assert len(x) > 0, 'Input data must not be empty.'
    assert timestamp_col in x.columns, 'Input must have the specified timestamp column.'

    try:
        x_timestamp = pd.to_datetime(x[timestamp_col])
    except:
        assert False, 'Invalid timestamp column name.'

    x['month-yr'] = x_timestamp.dt.strftime('%b-%Y')
    return x


def count_month_yr(x):
    """Count the number of occurrences of each month-year pair in DataFrame x. Sort chronologically.

    Args:
        x (pandas DataFrame): Input DataFrame object to get occurrences and sort.

    Returns:
        pandas DataFrame: Contains occurrences of each month-year pair, sorted chronologically by month.
    """

    assert isinstance(x, pd.DataFrame), 'Input data must be a pandas DataFrame object.'
    assert len(x) > 0, 'Input data must not be empty.'
    assert 'month-yr' in x.columns, 'Input must have a \'month-yr\' column.'

    months_count = x['month-yr'].value_counts()

    df = pd.DataFrame({'month-yr': months_count.index, 'number_trips': months_count}).set_index('month-yr')
    df['month-yr_time'] = pd.to_datetime(df.index, format='%b-%Y')

    return df.sort_values(by='month-yr_time').drop('month-yr_time', axis=1)



def visualize_rentals_year(rentals):
    """Visualize number of bike rentals by month and year from Sep 2013 - Aug 2015.

    Args:
        rentals (pandas DataFrame): Input DataFrame object to visualize number of rentals.

    Returns:
        None
    """

    assert isinstance(rentals, pd.DataFrame), 'Input must be a DataFrame object.'

    count = count_month_yr(rentals)
    count.drop('Aug-2013', inplace=True)

    year1 = count.loc['Sep-2013':'Aug-2014']
    year2 = count.loc['Sep-2014':'Aug-2015']
    months = ['Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug']

    plt.plot(months, year1/1000, label='2013-2014')
    plt.plot(months, year2/1000, label='2014-2015')
    plt.xlabel('Month')
    plt.ylabel('Number of Trips (thousands)')
    plt.title('Number of Bike Trips Taken from Sep 2013 - Aug 2015')
    plt.ylim(bottom=0)
    plt.legend()
    plt.show()

    return

def visualize_rentals_month(rentals):
    """Visualize average number of bike rentals by month.

    Args:
        rentals (pandas DataFrame): Input DataFrame object to visualize number of rentals.

    Returns:
        None
    """

    assert isinstance(rentals, pd.DataFrame), 'Input must be a DataFrame object.'

    count = count_month_yr(rentals)
    count.drop('Aug-2013', inplace=True)

    month_yr = pd.Series(count.index)
    split_expand = month_yr.str.split('-', expand=True)
    split_expand.index = count.index
    count['month'] = split_expand[0]

    year1 = count.loc['Sep-2013':'Aug-2014'].set_index('month')
    year2 = count.loc['Sep-2014':'Aug-2015'].set_index('month')

    per_month = pd.DataFrame({'trips_13-14': year1['number_trips'], 'trips_14-15': year2['number_trips']})
    avg_per_month = per_month.mean(axis=1)
    ordered_avg1 = avg_per_month['Jan':]
    ordered_avg2 = avg_per_month[:'Dec']

    plt.bar(ordered_avg1.index, ordered_avg1/1000, color='#348ABD')
    plt.bar(ordered_avg2.index, ordered_avg2/1000, color='#348ABD')
    plt.xlabel('Month')
    plt.ylabel('Number of Trips (thousands)')
    plt.title('Average Number of Bike Trips Taken per Month')
    plt.ylim(bottom=0)
    plt.show()

    return
