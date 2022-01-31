# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 18:22:39 2021

@author: Da Huo -- Aug 2021

Example for plotting Carbon Monitor-Cities emission data for a given city

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter

country = 'France'
city_name = 'Paris'

# Load data for that country
filename = '/Data/carbon-monitor-cities-%s-v0105.csv' %(country)

# function for plot Carbon Monitor-Cities emission data for a given city
def plot_CM_cities(df, days, city_name, x_lim, year):    
    fig = plt.figure(figsize=(8,5))
    ax = plt.axes()
    # Plot data for each sector
    plt.plot(days, df.loc[(df["city"] == city_name) & (df["sector"] == 'Power'), "value"],'r', label='Power')
    plt.plot(days, df.loc[(df["city"] == city_name) & (df["sector"] == 'Industry'), "value"],'k', label='Industry')
    plt.plot(days, df.loc[(df["city"] == city_name) & (df["sector"] == 'Residential'), "value"],'b', label='Residential')
    plt.plot(days, df.loc[(df["city"] == city_name) & (df["sector"] == 'Ground Transport'), "value"],'g', label='Ground Transport')
    plt.plot(days, df.loc[(df["city"] == city_name) & (df["sector"] == 'Aviation'), "value"],'y', label='Aviation')
    ax.set_xlim(0, x_lim)  
    ax.set_ylabel('Daily $CO_2$ Emissions (kt $CO_2$)', fontsize = 16)
    ax.set_title((city_name +'  ' + year), fontsize = 16)
    # Define the date format
    date_form = DateFormatter("%b-%d")
    ax.xaxis.set_major_formatter(date_form)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.legend(prop={'size': 10},loc = 'upper right')  
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)    

# Load and plot
data = pd.read_csv(filename)#.to_numpy()
year = '(Jan 2019-Jun 2021)'
ndays = 365+366+181 # 1/1/2019 to 30/6/2021
days = np.arange(1, ndays+1, 1)
plot_CM_cities(data, days, city_name, ndays, year) 
