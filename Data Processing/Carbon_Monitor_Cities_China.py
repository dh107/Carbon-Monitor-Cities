# -*- coding: utf-8 -*-
"""
Created on Tue May 10 12:16:13 2022

@author: Da Huo
"""

import numpy as np
import pandas as pd
import csv
import collections
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import os
import xlsxwriter
import glob
import pandas as pd


# code for producingm, saving and plotting Carbon Monitor-Cities China (CMCC) dataset


def plot_CM_cities(df, days, city_name, x_lim, year):    
    df['date'] = pd.to_datetime(df['date'])  
    fig = plt.figure(figsize=(8,5))
    ax = plt.axes()
    # Plot data for each sector
    plt.plot(days, df.loc[(df["city"] == city_name) & (df["sector"] == 'Power'), "value (KtCO2 per day)"],'r', alpha=0.8, linewidth=2.5, label='Power')
    plt.plot(days, df.loc[(df["city"] == city_name) & (df["sector"] == 'Industry'), "value (KtCO2 per day)"],'orange', alpha=0.8, linewidth=2.5,label='Industry')
    plt.plot(days, df.loc[(df["city"] == city_name) & (df["sector"] == 'Residential'), "value (KtCO2 per day)"],'g', alpha=0.8, linewidth=2.5,label='Residential')
    plt.plot(days, df.loc[(df["city"] == city_name) & (df["sector"] == 'Ground Transport'), "value (KtCO2 per day)"],'y', alpha=0.8, linewidth=2.5,label='Ground Transport')
    plt.plot(days, df.loc[(df["city"] == city_name) & (df["sector"] == 'Aviation'), "value (KtCO2 per day)"],'b', alpha=0.8, linewidth=2.5,label='Aviation')
    ax.set_xlim(0, x_lim-2) 
    #ax.set_ylim(0, 0.2) 
    ax.set_ylabel('Daily $CO_2$ Emissions (kt $CO_2$)', fontsize = 16)
    ax.set_title((city_name +'  ' + year), fontsize = 20)
    # Define the date format
    date_form = DateFormatter("%b-%d")
    ax.xaxis.set_major_formatter(date_form)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.legend(prop={'size': 10},loc = 'upper right')  
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)    
        
def save_CM_cities(data, num_sect, country, city, year, output_path):          
    output = pd.DataFrame([[city, country, '', '', '',''] for x in range(data.shape[0]*num_sect)], columns=['city','Province','date','sector','value (KtCO2 per day)','timestamp'])
        
    if year=="y2021":
        date_sec_stamp = pd.read_csv("/Users/CM_2021_template_date_sector_timestamp_power.csv")
    if year=="y2020":
        date_sec_stamp = pd.read_csv("/Users/CM_2020_template_date_sector_timestamp_power.csv")
    if year=="y2019":
        date_sec_stamp = pd.read_csv("/Users/CM_2019_template_date_sector_timestamp_power.csv")

    output["date"] = date_sec_stamp["date"]
    output["sector"] = date_sec_stamp["sector"]
    output["timestamp"] = date_sec_stamp["timestamp"]
                
    for i in range(num_sect):
        output["value (KtCO2 per day)"][data.shape[0]*i:data.shape[0]*(i+1)] = data[:,i]
              
    output.to_csv(output_path + 'CMCC-%s-%s-%s-%s.csv' %(country,city,year,date_sec_stamp["sector"][0]), index=False, encoding = 'utf_8_sig')              

def update_CM_using_baseyear(CMCC, CM_prov, city, prov, country, baseyear, year, resolution, output_path):
    """
     get emis data for newyear based on ratio computed from CM data
     year: the new year: 2019 or 2021
     baseyear: 2020
    :return: save results into files (for city)
    """
    year_ratio = collections.defaultdict(dict)
    
    if year=='2019' or year=='2021':
        last_day = 365
        data = np.zeros([365,5])
    if year=='2020':
        last_day = 366
        data = np.zeros([366,5])
    CMCC['date'] = pd.to_datetime(CMCC['date'])
  
    if baseyear == '2020':
        # drop Feb 29
        CMCC.drop(CMCC.loc[CMCC['date']=='2020-02-29'].index, inplace=True)
        
    for i, sector in enumerate(CM_sectors):
        
        data[:,i]= emis_diff_between_years(CM_prov, baseyear, year, sector, prov, resolution) * CMCC.loc[(CMCC["city"] == city) & (CMCC["sector"] == sector) & (CMCC['date'] >= str(baseyear)+'-01-01') & (CMCC['date'] <= str(baseyear)+'-12-31'), "value (KtCO2 per day)"].to_numpy()

        year_ratio[city][sector] = emis_diff_between_years(CM_prov, baseyear, year, sector, prov, resolution)

    print('complted: ', city)
    save_CM_cities(data, len(CM_sectors), prov, city, 'y'+year, output_path)

    return year_ratio 
    
    
def correct_save_flightradar24(fr24_data, fr24_data_baseyear, annual_total, city2prov, city, year, baseyear, output_path):
    fr24_data['time'] = pd.to_datetime(fr24_data['time'])
    fr24_data_baseyear['time'] = pd.to_datetime(fr24_data_baseyear['time'])
    
    # link city to airport
    airport = city2prov.loc[(city2prov["City"] == city), "airport"].values[0]
    
    # some cities has another airport
    airport2 = city2prov.loc[(city2prov["City"] == city), "airport2"].values[0]
    
    # cities with more than one airports
    if city == 'Beijing' or city == 'Shanghai':
        # only departure flights accounted
        aviation_flightradar_daily = fr24_data.loc[(fr24_data['time'] >= str(year)+'-01-01') & (fr24_data['time'] <= str(year)+'-12-31'), 'co2_' + str(airport) +'_dep'] + \
            fr24_data.loc[ (fr24_data['time'] >= str(year)+'-01-01') & (fr24_data['time'] <= str(year)+'-12-31'), 'co2_' + str(airport2) +'_dep']
        
        aviation_flightradar_daily_base = fr24_data_baseyear.loc[(fr24_data_baseyear['time'] >= str(baseyear)+'-01-01') & (fr24_data_baseyear['time'] <= str(baseyear)+'-12-31'), 'co2_' + str(airport) +'_dep'] + \
            fr24_data_baseyear.loc[ (fr24_data_baseyear['time'] >= str(baseyear)+'-01-01') & (fr24_data_baseyear['time'] <= str(baseyear)+'-12-31'), 'co2_' + str(airport2) +'_dep']
    else:
        aviation_flightradar_daily = fr24_data.loc[(fr24_data['time'] >= str(year)+'-01-01') & (fr24_data['time'] <= str(year)+'-12-31'), 'co2_' + str(airport) +'_dep']
        aviation_flightradar_daily_base = fr24_data_baseyear.loc[(fr24_data_baseyear['time'] >= str(baseyear)+'-01-01') & (fr24_data_baseyear['time'] <= str(baseyear)+'-12-31'), 'co2_' + str(airport) +'_dep']
    
    cur_year_sum = aviation_flightradar_daily.sum()
    base_year_sum = aviation_flightradar_daily_base.sum()
    yearly_ratio = cur_year_sum/base_year_sum
    
    aviation_daily_ratio = aviation_flightradar_daily/cur_year_sum
    
    emis_aviation_daily_corrected = yearly_ratio*annual_total.loc[annual_total['City Name'] == city, 'Aviation'].values[0]*10*aviation_daily_ratio
    
    # save results to file    
    data[:,0] = emis_aviation_daily_corrected
    prov = city2prov.loc[(city2prov["City"] == city), "Province"].values[0]
    save_CM_cities(data, 1, prov, city, 'y'+year, output_path)
    
    
    return emis_aviation_daily_corrected


def correct_save_power(power_daily_data, power_daily_data_baseyear, annual_total, city2prov, city, year, baseyear, sector, output_path):
    '''

    Not just power, any sector with daily data to be corrected with annual totoal (2020 as base year)
    
    Parameters
    ----------

    Returns
    -------
    emis_power_daily_corrected : 

    '''
    power_daily_data['time'] = pd.to_datetime(power_daily_data['time'])
    power_daily_data_baseyear['time'] = pd.to_datetime(power_daily_data_baseyear['time'])
    

    power_daily = power_daily_data.loc[(power_daily_data['time'] >= str(year)+'-01-01') & (power_daily_data['time'] <= str(year)+'-12-31'), city]
    power_daily_baseyear = power_daily_data_baseyear.loc[(power_daily_data_baseyear['time'] >= str(baseyear)+'-01-01') & (power_daily_data_baseyear['time'] <= str(baseyear)+'-12-31'), city]
    
    cur_year_sum = power_daily.sum()
    base_year_sum = power_daily_baseyear.sum()
    yearly_ratio = cur_year_sum/base_year_sum
    
    power_daily_ratio = power_daily/cur_year_sum
    
    emis_power_daily_corrected = yearly_ratio*annual_total.loc[annual_total['City Name'] == city, sector].values[0]*10*power_daily_ratio
    
    # save results to file
    data[:,0] = emis_power_daily_corrected
    prov = city2prov.loc[(city2prov["City"] == city), "Province"].values[0]
    save_CM_cities(data, 1, prov, city, 'y'+year, output_path)
    
    
    return emis_power_daily_corrected

country = 'China'
city_list = CCG['City Name'].values.tolist()


#%%
# ============ daily residential ==================================

def correct_save_resid(city_df, annual_total, city2prov, city, year, baseyear, sector, output_path):
    
    city_df['date'] = pd.to_datetime(city_df['date'])
    
    daily_curyear = city_df.loc[(city_df['date'] >= str(year)+'-01-01') & (city_df['date'] <= str(year)+'-12-31'), 'value (KtCO2 per day)']
    daily_baseyear = city_df.loc[(city_df['date'] >= str(baseyear)+'-01-01') & (city_df['date'] <= str(baseyear)+'-12-31'), 'value (KtCO2 per day)']
    
    cur_year_sum = daily_curyear.sum()
    base_year_sum = daily_baseyear.sum()
    yearly_ratio = cur_year_sum/base_year_sum
    
    daily_ratio = daily_curyear/cur_year_sum
    
    emis_power_daily_corrected = yearly_ratio*annual_total.loc[annual_total['City Name'] == city, sector].values[0]*10*daily_ratio
    
    # save results to file
    if year=='2019' or year=='2021':
        last_day = 365
        data = np.zeros([365,5])
    if year=='2020':
        last_day = 366
        data = np.zeros([366,5])
    
    data[:,0] = emis_power_daily_corrected
    prov = city2prov.loc[(city2prov["City"] == city), "Province"].values[0]
    save_CM_cities(data, 1, prov, city, 'y'+year, output_path)
    
    
    return emis_power_daily_corrected


building_model = pd.read_excel(path + 'building_model.xlsx', sheet_name = 'building_model_output')
#%%

CM_cities = pd.read_csv(path +'data.csv')
building_energy_share = {}
daily_energy_building_type = {}
final_daily_resid = pd.DataFrame([['', '', ''] for x in range(365*2+366)], columns=['city','date','value (KtCO2 per day)'])

sector = 'Residential'
year = '2021'
baseyear = '2020'
#writer = pd.ExcelWriter(output_path + 'Daily-building-energy-48-cities-2019-2021.xlsx', engine='xlsxwriter')

for city in city_list:
    prov = city2prov.loc[(city2prov["City"] == city), "Province"].values[0]
    
    building_energy_share_full = pd.read_excel(path + '', sheet_name = prov+'2019')
    
    building_energy_share[city] = building_energy_share_full.iloc[1] #e.g., Shanghai 0.4917 => 49.17% of residential energy consump is buildings

    CM_cities['date'] = pd.to_datetime(CM_cities['date'])
  
    final_daily_resid['city'] = city

    city0 = city
        
    final_daily_resid['date'] = CM_cities.loc[(CM_cities["city"] == city0) & (CM_cities["sector"] == "Residential"), 'date'].values
    
    final_daily_resid = final_daily_resid.sort_values(by="date",ignore_index=True)
    
    daily_builings = building_type_share.loc[building_type_share["City Name"] == city,'Residential'].values/100  * daily_energy_building_type['Residential'] \
        + building_type_share.loc[building_type_share["City Name"] == city,'Office'].values/100  * daily_energy_building_type['Office'] \
            + building_type_share.loc[building_type_share["City Name"] == city,'Mall and restaurant'].values/100  * daily_energy_building_type['Mall and restaurant'] \
                + building_type_share.loc[building_type_share["City Name"] == city,'Public and others'].values/100  * daily_energy_building_type['Public and others']

#%%
# ============ daily power ==================================
year = '2021'
baseyear = '2020'
sector = 'Power'

power_daily_data = pd.read_excel(path + '' + year +'.xlsx')
power_daily_data_baseyear = pd.read_excel(path + '' + baseyear +'.xlsx')

out_path = path + '/Power/'

for city in city_list:
    prov = city2prov.loc[(city2prov["City"] == city), "Province"].values[0]
    data = correct_save_power(power_daily_data, power_daily_data_baseyear, CCG, city2prov, city, year, baseyear, sector, out_path)

#%%
# ============ daily Transport ==================================
year = '2021'
baseyear = '2020'
sector = 'Ground Transport'

power_daily_data = pd.read_excel(path + '' + year +'.xlsx')
power_daily_data_baseyear = pd.read_excel(path + '' + baseyear +'.xlsx')

for city in city_list:
    prov = city2prov.loc[(city2prov["City"] == city), "Province"].values[0]
    data = correct_save_power(power_daily_data, power_daily_data_baseyear, CCG, city2prov, city, year, baseyear, sector, out_path)


#%%
# ============ daily aviation ==================================
year = '2021'
baseyear = '2020'

fr24_data = pd.read_excel(path + ''+ year +'.xlsx')
fr24_data_baseyear = pd.read_excel(path + '0'+ baseyear +'.xlsx')

for city in city_list:
    prov = city2prov.loc[(city2prov["City"] == city), "Province"].values[0]
    data = correct_save_flightradar24(fr24_data, fr24_data_baseyear, CCG, city2prov, city, year, baseyear, out_path)

#%%
# merge separate files (one file for each sector each city) and save as single csv (one city with all sectors per sheet)

def merge_all_cities_in_folder(Path_merged_data, path_file_to_merge, writer, city_list):
    # create an empty holder for all cities
    merged = pd.DataFrame([], columns=['city', 'Province', 'date', 'sector','value (KtCO2 per day)', 'timestamp'])
    os.chdir(path_file_to_merge)
    listOfFiles = os.listdir(path_file_to_merge)
    count = 0

    for filename in listOfFiles:
        df = pd.read_csv(filename)
        merged = pd.concat([merged, df], axis=0) 
        count += 1
        print('completed:', count)     
    
    for city in city_list:
        data_one_city = merged.loc[merged['city' ]==city]
        data_one_city.to_excel(writer, sheet_name=city, index = False, encoding = 'utf_8_sig')
    
    writer.save()    
        
path_file_to_merge = '/Users/'
outpath = '/Users/'
# Create a Pandas Excel writer
writer = pd.ExcelWriter(outpath + 'CMCC_48_cities_v0629.xlsx', engine='xlsxwriter')
merge_all_cities_in_folder(outpath, path_file_to_merge, writer, city_list)


# ============ plot final dataset =============================

year = '2020'
year1 = '2021'

for city_name in city_list:
    out_path = '/Users/'
    CMCC20 = pd.read_excel(out_path + 'CMCC_48_cities_v0629.xlsx', sheet_name = city_name)
    title = '(2020/01/01-2021/12/31)'
    ndays = 366 + 365
    days = np.arange(1, ndays+1, 1)
    ndays1 = 365 + 366
    days1 = np.arange(1, ndays1+1, 1)
    plot_CM_cities(CMCC20, days, city_name, ndays, title)



