# Project: SF-Bike-Share-Data-Analysis

## Description
  ECE 143 Final Project, Group 0.
## Date
  11/07/2021

## File Structures
### dataset_bike_share
  A folder that contains our dataset, including 'station.csv',
  'status.csv', 'trip.csv', 'weather.csv', and 'Weekly_San_Francisco_CA_Regular_Reformulated_Retail_Gasoline_Prices.csv'.

  The first four files should be downloaded from https://www.kaggle.com/benhamner/sf-bay-area-bike-share.
  The last one is downloaded from https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=EMM_EPMRR_PTE_Y05SF_DPG&f=W. We then manually select
  the data from 08/26/2013 to 08/31/2015.

### gas_price_and_duration.py
  Analyze the relations between gas price and number of rentals, and between trip
  start time and durations.
### map_visuals.py
  Visualize the stations' locations, docker size and availability in different times.
### number_rentals.py
  Count number of rentals per month / year and visualize the data.
### preprocessing.py
  Prepare the data for map_visuals.py.
### rentals_models.py
  Build models to predict number of rentals using weather data and date.
### trip_dataset_visualization.py
  Visualize the trip durations, difference between subscribers and customers, and
  the popularity of stations.
### visualizations.ipynb
  Notebook that contains all visualizations.
### weather.py
  Visualize the relations between number of rentals and weather.


## How to Run Our Code
  1. Download the data and save them in 'dataset_bike_share'.
  2. Run visualizations.ipynb. Please download it to view our bokeh graphs.


## Third-Party Modules
  numpy

  pandas

  matplotlib.pyplot

  bokeh

  holoviews

  sklearn

  tensorflow.keras
