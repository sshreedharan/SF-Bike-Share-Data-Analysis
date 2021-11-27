import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_file, show, output_notebook
from bokeh.tile_providers import CARTODBPOSITRON, get_provider
from bokeh.models import ColumnDataSource, GMapOptions

# all functions need to be called after preprocess trip data with preprocessing(trip)
def preprocessing(data):
    """
    Args:
        pd.DataFrame
    Returns:
        pd.DataFrame
    This function adds time related labels to the original data frame
    """
    data = data.drop(data[data['duration'] > 14400].index.tolist())
    data['s_date'] = data['start_date'].str.split(' ', expand = True)[0]
    data['s_time'] = data['start_date'].str.split(' ', expand = True)[1]
    data['s_date'] = pd.to_datetime(data['s_date'])
    data['month-yr'] = [d.strftime('%Y-%m') for d in data['s_date']]
    data['s_year'] = data['s_date'].dt.year
    data['s_time'] = pd.to_datetime(data['s_time']).dt.hour
    data['s_dayofweek'] = pd.to_datetime(data['s_date']).dt.dayofweek
    data['s_month'] = pd.to_datetime(data['s_date']).dt.month
    data = data.drop(data[data['month-yr'] == '2013-08'].index.tolist())
    return data

# mean duration daily
def daily_avg_duration(data):
    """
    Args:
        data (pd.DataFrame) preprocessed trip data
    This function computes daily average duration from preprocessed data and visualizes the data with a line chart
    """
    mean_d = pd.DataFrame(data[['s_year','s_date','duration']])
    mean_d = mean_d.groupby('s_date', as_index = False).mean()
    mean_d = mean_d.sort_values('s_date')
    mean_d['duration'] = mean_d['duration']//60
    mean_d_13 = mean_d[mean_d['s_year'] == 2013]
    mean_d_14 = mean_d[mean_d['s_year'] == 2014]
    mean_d_15 = mean_d[mean_d['s_year'] == 2015]
    plt.figure(1,figsize = (24,12),dpi = 100)

    plt.subplot(3,1,1)
    plt.plot(mean_d_13['s_date'],mean_d_13['duration'],'o--',label = '2013')
    plt.legend()

    plt.subplot(3,1,2)
    plt.plot(mean_d_14['s_date'],mean_d_14['duration'],'r',label = '2014')
    plt.legend()

    plt.subplot(3,1,3)
    plt.plot(mean_d_15['s_date'],mean_d_15['duration'],'g--',label = '2015')
    plt.legend()

    plt.title('Daily Average Trip Duration (minutes)')
    plt.show()

# mean duration monthly
def monthly_avg_duration(data):
    """
    Args:
        data (pd.DataFrame) preprocessed trip data
    This function computes monthly average duration from preprocessed data and visualizes the data with a line chart
    """
    mean_m = pd.DataFrame(data[['month-yr','duration']])
    mean_m = mean_m.groupby('month-yr', as_index = False).mean()
    mean_m = mean_m.sort_values('month-yr')
    mean_m['duration'] = mean_m['duration']//60
    plt.figure(1,figsize = (24,8),dpi = 100)
    plt.plot(mean_m['month-yr'],mean_m['duration'],'o--')

    plt.title('Monthly Average Trip Duration (minutes)')
    plt.show()

# mean duration day of week
def dayofweek_avg_duration(data):
    """
    Args:
        data (pd.DataFrame) preprocessed trip data
    This function computes day of week average duration from preprocessed data and visualizes the data with a bar chart
    """
    mean_dow = pd.DataFrame(data[['s_dayofweek','duration']])
    mean_dow = mean_dow.groupby('s_dayofweek', as_index = False).mean()
    mean_dow = mean_dow.sort_values('s_dayofweek')
    mean_dow['duration'] = mean_dow['duration']//60
    plt.figure(1,figsize = (24,8),dpi = 100)
    plt.bar(mean_dow['s_dayofweek'],mean_dow['duration'], width = 0.5)

    plt.title('Average Trip Duration (minutes) Day of Week')
    plt.show()

# mean duration hourly
def hourly_avg_duration(data):
    """
    Args:
        data (pd.DataFrame) preprocessed trip data
    This function computes hourly average duration from preprocessed data and visualizes the data with a line chart
    """
    mean_h = pd.DataFrame(data[['s_time','duration']])
    mean_h = mean_h.groupby('s_time',as_index = False).mean()
    mean_h = mean_h.sort_values('s_time')
    mean_h['duration'] = mean_h['duration']//60
    plt.figure(1,figsize = (24,8),dpi = 100)
    plt.plot(mean_h['s_time'],mean_h['duration'],'o--')

    plt.title('Hourly Average Trip Duration (minutes)')
    plt.show()

# monthly subscriber vs customer trip count
def monthly_trip_count_vs(data):
    """
    Args:
        data (pd.DataFrame) preprocessed trip data
    This function compares monthly trip count between subscriber and customer from preprocessed data and visualizes the data with a line chart
    """
    # subscriber vs customer: trip count
    sub = data[data['subscription_type'] == 'Subscriber']
    cus = data[data['subscription_type'] == 'Customer']
    # monthly trip count trend
    m_sub_cnt = pd.DataFrame(sub['month-yr'])
    m_cus_cnt = pd.DataFrame(cus['month-yr'])
    m_sub_cnt = pd.DataFrame(m_sub_cnt.groupby('month-yr',as_index=False).size())
    m_cus_cnt = pd.DataFrame(m_cus_cnt.groupby('month-yr',as_index=False).size())
    plt.figure(1,figsize = (24,8),dpi = 100)
    plt.plot(m_sub_cnt['month-yr'],m_sub_cnt['size'],'o--',label = 'subscriber')
    plt.plot(m_cus_cnt['month-yr'],m_cus_cnt['size'],'o--',label = 'customer')
    plt.legend()
    plt.title('Monthly Trip Count Subscriber vs Customer')
    plt.show()

# dayofweek subscriber vs customer trip count
def dayofweek_trip_count_vs(data):
    """
    Args:
        data (pd.DataFrame) preprocessed trip data
    This function compares day of week trip count between subscriber and customer from preprocessed data and visualizes the data with a line chart
    """
    # subscriber vs customer: trip count
    sub = data[data['subscription_type'] == 'Subscriber']
    cus = data[data['subscription_type'] == 'Customer']
    d_sub_cnt = pd.DataFrame(sub['s_dayofweek'])
    d_cus_cnt = pd.DataFrame(cus['s_dayofweek'])
    d_sub_cnt = pd.DataFrame(d_sub_cnt.groupby('s_dayofweek',as_index=False).size())
    d_cus_cnt = pd.DataFrame(d_cus_cnt.groupby('s_dayofweek',as_index=False).size())
    plt.figure(1,figsize = (16,6),dpi = 100)
    plt.plot(d_sub_cnt['s_dayofweek'],d_sub_cnt['size'],'o--',label = 'subscriber')
    plt.plot(d_cus_cnt['s_dayofweek'],d_cus_cnt['size'],'o--',label = 'customer')
    plt.legend()
    plt.title('Day of Week Trip Count Subscriber vs Customer')
    plt.show()

# subscriber vs customer: avg duration monthly
def monthly_avg_duration_vs(data):
    """
    Args:
        data (pd.DataFrame) preprocessed trip data
    This function compares monthly average duration between subscriber and customer from preprocessed data and visualizes the data with a line chart
    """
    sub = data[data['subscription_type'] == 'Subscriber']
    cus = data[data['subscription_type'] == 'Customer']
    m_sub_dur = pd.DataFrame(sub[['s_month','duration']])
    m_cus_dur = pd.DataFrame(cus[['s_month','duration']])
    m_sub_dur = pd.DataFrame(m_sub_dur.groupby('s_month',as_index=False).mean())
    m_cus_dur = pd.DataFrame(m_cus_dur.groupby('s_month',as_index=False).mean())
    m_sub_dur['duration'] = m_sub_dur['duration']//60
    m_cus_dur['duration'] = m_cus_dur['duration']//60
    plt.figure(1,figsize = (16,6),dpi = 100)
    plt.plot(m_sub_dur['s_month'],m_sub_dur['duration'],'o--',label = 'subscriber')
    plt.plot(m_cus_dur['s_month'],m_cus_dur['duration'],'o--',label = 'customer')
    plt.legend()
    plt.title('Monthly Trip Duration Subscriber vs Customer')
    plt.show()

# subscriber vs customer: avg duration day of week
def dayofweek_avg_duration_vs(data):
    """
    Args:
        data (pd.DataFrame) preprocessed trip data
    This function compares day of week average duration between subscriber and customer from preprocessed data and visualizes the data with a line chart
    """
    sub = data[data['subscription_type'] == 'Subscriber']
    cus = data[data['subscription_type'] == 'Customer']
    d_sub_dur = pd.DataFrame(sub[['s_dayofweek','duration']])
    d_cus_dur = pd.DataFrame(cus[['s_dayofweek','duration']])
    d_sub_dur = pd.DataFrame(d_sub_dur.groupby('s_dayofweek',as_index=False).mean())
    d_cus_dur = pd.DataFrame(d_cus_dur.groupby('s_dayofweek',as_index=False).mean())
    d_sub_dur['duration'] = d_sub_dur['duration']//60
    d_cus_dur['duration'] = d_cus_dur['duration']//60
    plt.figure(1,figsize = (16,6),dpi = 100)
    plt.plot(d_sub_dur['s_dayofweek'],d_sub_dur['duration'],'o--',label = 'subscriber')
    plt.plot(d_cus_dur['s_dayofweek'],d_cus_dur['duration'],'o--',label = 'customer')
    plt.legend()
    plt.title('Day of Week Trip Duration Subscriber vs Customer')
    plt.show()

# 5 most popular start station weekday vs weekend
def m_pop_s_station(trip,station):
    pop_s_day = trip[['s_dayofweek','start_station_name']]
    pop_s_day = pd.DataFrame(pop_s_day)
    pop_s_day = pop_s_day.groupby(['s_dayofweek','start_station_name'], as_index = False).size()
    pop_s_day = pd.DataFrame(pop_s_day)
    pop_s_wk = pop_s_day[pop_s_day['s_dayofweek']==0].nlargest(5,'size')
    m = pop_s_day[pop_s_day['s_dayofweek']==0].nlargest(5,'size')
    t = pop_s_day[pop_s_day['s_dayofweek']==1].nlargest(5,'size')
    w = pop_s_day[pop_s_day['s_dayofweek']==2].nlargest(5,'size')
    th = pop_s_day[pop_s_day['s_dayofweek']==3].nlargest(5,'size')
    f = pop_s_day[pop_s_day['s_dayofweek']==4].nlargest(5,'size')
    s = pop_s_day[pop_s_day['s_dayofweek']==5].nlargest(5,'size')
    su = pop_s_day[pop_s_day['s_dayofweek']==6].nlargest(5,'size')
    for i in range(1,7):
        tmp = pop_s_day[pop_s_day['s_dayofweek']==i].nlargest(5,'size')
        pop_s_wk = pd.concat([pop_s_wk,tmp])

    station = station[['name','lat','long']]
    station = station.rename(columns={'name':'start_station_name'})
    pop_s_wk = pd.merge(station,pop_s_wk,on='start_station_name',how = 'right')

    # convert latitude and longitude to mercator coordinates
    pop_s_wk['x'] = np.radians(pop_s_wk['long'])*6378137
    pop_s_wk['y'] = 180/np.pi*np.log(np.tan(np.pi/4+pop_s_wk['lat']*(np.pi/180)/2))*pop_s_wk['x']/pop_s_wk['long']

    # circle radius defined by trip count
    pop_s_wk['radius'] = pop_s_wk['size']//10

    output_file("tile.html")

    tile_provider = get_provider(CARTODBPOSITRON)

    # range bounds supplied in web mercator coordinates
    p = figure(x_range=(-1.363606e+07, -1.362017e+07), y_range=(4.507915e+06, 4.561883e+06),
               x_axis_type="mercator", y_axis_type="mercator")
    p.add_tile(tile_provider)

    source1 = ColumnDataSource(pop_s_wk[pop_s_wk['s_dayofweek']<5]) # weekday
    source2 = ColumnDataSource(pop_s_wk[pop_s_wk['s_dayofweek']>=5]) # weekend

    p.circle(x="x", y="y", size=10, fill_color="yellow", fill_alpha=0.4, radius = 'radius', source=source1)
    p.circle(x="x", y="y", size=10, fill_color="red", fill_alpha=0.4, radius = 'radius', source=source2)

    output_notebook()
    show(p)
    return pop_s_wk
