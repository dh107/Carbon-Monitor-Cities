# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 20:45:22 2021

@author: Da Huo

Workdflow to generate Carbon Monitor Cities Dataset (v1.3)

Email: daniel.huo2013@gmail.com
Or:    dh2107@tsinghua.edu.cn
"""
import numpy as np
import pandas as pd
import os
import calendar
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import datetime as dt
import time
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import difflib
import re

# Load gridded data and city list
FUA_type = 'FUA'
server_path = '/home/hda/'
    
if FUA_type == 'FUA':
    # FUA
    world_city = Dataset(server_path+'FUA_cities.nc')
    wp = pd.read_excel(server_path+'FUA.xls', header=0)
    index_country_prov_city = wp[['FID', 'NAME_0', 'NAME_1', 'NAME_2']].to_numpy() 
    city_index = wp[['FID']].to_numpy()
    city_array = world_city.variables[u'layer'][:]
    
if FUA_type == 'GADM':
    world_city = Dataset(server_path+'GADM.nc')
    wp = pd.read_excel(server_path+'GADM.xls', header=0)
    index_country_prov_city = wp[['FID', 'NAME_0', 'NAME_1', 'NAME_2']].to_numpy() 
    city_index = wp[['FID']].to_numpy()
    city_array = world_city.variables[u'citycode'][:]
    
output_path ='/home/hda/FUA/'
path_file_to_merge =  output_path
years = ['y2019', 'y2020', 'y2021']

Path_tomtom = server_path +'all_tomtom/' 
output_name = 'update_2021'
output_version = 'v03'
Path_merged_data = server_path +'merged/carbon-monitor-cities-2021full-merged.csv'
merged_output_path = server_path +'merged/'

city_rows = city_array.shape[0]
city_cols = city_array.shape[1]

# print number of cities
unique, counts = np.unique(city_array, return_counts=True)
cmsect = ['Power','Industry','Residential','Ground Transport','Aviation']
sectname = ['Power', 'Industry', 'Residential', 'GroundTransportation','DomesticAviation']

# number of months and days
nmon = 12
mdays=np.array(calendar.mdays[1:1+nmon])
mdays[1] = 29 # Feb 29, 2020
nday=np.sum(mdays)

# load HDD data
HDD_city = pd.read_csv(server_path +'HDD_all.csv')
HDD_country = pd.read_csv(server_path +'HDD_country_all.csv')
all_country_HDD = HDD_country['Nation'].drop_duplicates().dropna().tolist()
HDD_city = HDD_city.dropna()
          
def save_CM_cities(data, num_sect, country, city, year, output_path):          
    output = pd.DataFrame([[city, country, '', '', '',''] for x in range(data.shape[0]*num_sect)], columns=['city','country','date','sector','value (KtCO2 per day)','timestamp'])
        
    if year=="y2021":
        date_sec_stamp = pd.read_csv("/home/hda/format_templates/CM_2021_template_date_sector_timestamp.csv")
    if year=="y2020":
        date_sec_stamp = pd.read_csv("/home/hda/format_templates/CM_2020_template_date_sector_timestamp.csv")
    if year=="y2019":
        date_sec_stamp = pd.read_csv("/home/hda/format_templates/CM_2019_template_date_sector_timestamp.csv")
    # col #3
    output["date"] = date_sec_stamp["date"]
    # col #4
    output["sector"] = date_sec_stamp["sector"]
    # col #6
    output["timestamp"] = date_sec_stamp["timestamp"]
                
    for i in range(num_sect):
        output["value (KtCO2 per day)"][data.shape[0]*i:data.shape[0]*(i+1)] = data[:,i]
              
    output.to_csv(output_path + 'carbon-monitor-city-%s-%s-%s.csv' %(country,city,year), index=False, encoding = 'utf_8_sig')              

def correct_outliers(data):
    threshold = np.std(data)*3
    for i in range(1,len(data)-1):
        if abs(data[i] - data[i-1]) > threshold and abs(data[i] - data[i+1]) > threshold:       
            data[i] = np.nanmean([data[i-1], data[i+1]])
            print('corrected at:',i)
    return data           

def plot_CM_cities(data, days, city_name, x_lim, year):    
    fig = plt.figure(figsize=(8,5))     
    ax = plt.axes()
    plt.plot(days, data[:,0],'r', label='Power')
    plt.plot(days, data[:,1],'k', label='Industry')
    plt.plot(days, data[:,2],'b', label='Residential')
    plt.plot(days, data[:,3],'g', label='Ground Transportation')
    plt.plot(days, data[:,4],'y', label='Aviation')
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

def get_close_cityname(cityname, all_cityname, num, threshold = 0.1):
    n = 1.1
    city_name_result = []
    while n>=threshold:
        n = n - 0.1
        city_name_result.append(difflib.get_close_matches(cityname, all_cityname, num, cutoff=n))
    if city_name_result[-1] == []:
        return []
    return next(s for s in city_name_result if s)

def get_HDD_correction_factor(country, city, year, days):
    not_in_HDD = False
    country_HDD = get_close_cityname(country, all_country_HDD, 1)  
    all_city_HDD = HDD_city.loc[HDD_city['Nation']==country_HDD[0],'fuaname'].drop_duplicates().dropna().tolist()
    city_HDD = get_close_cityname(city, all_city_HDD, 1, 0.8)

    if city_HDD == []:
        not_in_HDD = True
        return 1, not_in_HDD
   
    HDD_country_avg = HDD_country.loc[HDD_country['Nation']== country_HDD[0]].to_numpy()
    HDD_city_avg = HDD_city.loc[HDD_city['Nation']== country_HDD[0]].loc[HDD_city['fuaname']== city_HDD[0]].to_numpy()  
    correct_factor_daily = (HDD_country_avg - HDD_city_avg)/np.amax(HDD_country_avg)     
    return correct_factor_daily, not_in_HDD

def residential_correction(data, city, country, year, days):
    # get HDD correction factor for current city
    correct_factor_daily, not_in_HDD = get_HDD_correction_factor(country, city, year, days)
    
    if not_in_HDD:
        return     
    data[:,2]= correct_factor_daily*data[:,2]
    data[:,2] = correct_outliers(data[:,2])
    return data


def merge_all_cities_in_folder(Path_merged_data, path_file_to_merge):
    # create an empty holder for all cities
    merged = pd.DataFrame([], columns=['city', 'country', 'date', 'sector','value (KtCO2 per day)', 'timestamp'])
    os.chdir(path_file_to_merge)
    listOfFiles = os.listdir(path_file_to_merge)
    for filename in listOfFiles:
        df = pd.read_csv(filename)
        merged = pd.concat([merged, df], axis=0)    
        merged.to_csv(Path_merged_data, index = False, encoding = 'utf_8_sig')
    return merged

def transport_correction_save_by_contry(Path_tomtom, Path_merged_data, merged_output_path, output_name, output_version):
    Path_1 = Path_tomtom # tomtom txt files
    Path_2 = Path_merged_data #final merged all city data
    Path_3 = merged_output_path 
    
    Sector_Name = 'Ground Transport'    
    #####read TomTom emissions ##########################################
    filePath = Path_1
    country = os.listdir(filePath)
    data_name = []
    for dbtype in country:
        if os.path.isfile(os.path.join(filePath,dbtype)):
            data_name.append(dbtype)
            
    data_name = [data_name[i] for i,x in enumerate(data_name) if x.find('txt')!=-1] 
    
    #get city name
    City_Name = []
    for i in data_name:
        title = re.compile(r'.*?_(?P<Name>.*?)[.]', re.S)
    
        City_Name_result = title.finditer(i) 
        for it in City_Name_result:
            City_Name.append(it.group("Name"))
    
    result = []
    for x,y in zip(data_name,City_Name):
        df = pd.read_csv(filePath+x, sep=',')
        df['city'] = y
        result.append(df)
    df_all = pd.DataFrame(np.concatenate(result), columns = df.columns) 
    
    df_data = pd.read_csv(Path_2)
    
    # formatting
    df_data['city'] = df_data['city'].str.title()
    df_data['sector'] = df_data['sector'].str.title()
    
    df_data_timestamp = df_data['timestamp'].tolist()
    date_time = []
    for x in df_data_timestamp:
        timeArray = time.gmtime(x)
        date_time.append(time.strftime("%Y-%m-%d", timeArray))
        #date_time.append(time.strftime("%d/%m/%Y", timeArray))
    df_data['date'] = date_time
    df_data = df_data.sort_values(by=['city','sector','date'],ascending=True,na_position='first').reset_index(drop = True)
    

    for x in df_data.columns.tolist():
        if type(df_data[x].tolist()[0]) == float:
            Value_col = x
                
    df_all = df_all.set_index(['city','Mon/Day']).stack().reset_index().rename(columns={'level_2':'year', 0:Value_col})   
    year_list = df_all['year'].tolist()
    
    Year_Name = []
    for i in year_list:
        title = re.compile(r'\d{4}') # get 4-digit year
        Year_Name.append(title.findall(i)) 
    Year_Name = [int(x) for item in Year_Name for x in item] # to int
    df_all['year'] = Year_Name
    #separate month and day
    df_all = pd.concat([df_all[['city','year',Value_col]],df_all['Mon/Day'].str.split('-', expand=True).rename(columns = {0:'month',1:'day'})], axis = 1) 
    #to full date
    df_all['date'] = pd.to_datetime(df_all[['year', 'month','day']].assign(), errors='coerce')
    
    #add sector
    df_all['sector'] = Sector_Name
    
    null_list = df_all[df_all[['date']].isnull().T.any()].index.tolist()
    
    df_all = df_all.drop(null_list)


    first_date = df_data['date'].drop_duplicates().tolist()[0]
    last_date = df_data['date'].drop_duplicates().tolist()[-1]
    df_all = df_all[(df_all['date'] >= first_date) & (df_all['date'] <= last_date)].reset_index(drop = True)
    
    df_all = df_all[['city','date','sector',Value_col]]
    
    df_all['city'] = df_all['city'].str.title()
    df_all['sector'] = df_all['sector'].str.title()
    
    #get values only
    df_all[Value_col] = df_all[Value_col].astype('float')
    
    #rank cities
    df_all = df_all.sort_values(by=['city','date'],ascending=True,na_position='first').reset_index(drop = True)
    
    df_all[Value_col]=df_all[Value_col].fillna(df_all[Value_col].interpolate())
    #unit convert to KtCO2
    df_all[Value_col]=df_all[Value_col]/1000
    
    #====================== City-level correction for Transport Sector ======================
    # drop empty rows                
    df_data[Value_col].replace('', np.nan, inplace=True)
    df_data.dropna(subset = [Value_col], inplace=True) 
    
    #cityname list
    city_df_data = df_data['city'].drop_duplicates().tolist()
    city_df_all = df_all['city'].drop_duplicates().tolist()

    n = 1.1
    city_name_result = []
    while len(city_name_result) != len(city_df_all):
        n = n - 0.1
        city_name_result.clear()
        for x in city_df_all:
            if difflib.get_close_matches(x, city_df_data, 1, cutoff=n) == []:
                pass
            else:
                city_name_result.append(difflib.get_close_matches(x, city_df_data, 1, cutoff=n))
        city_name_result = [str(x) for item in city_name_result for x in item]

    for x,y in zip(city_df_all,city_name_result):
        print(f'{x} match to {y}')
    for x,y in zip(city_df_all,city_name_result):
        df_all['city'] = df_all['city'].str.replace(x,y)
    
    country_name_result = []
    for x in city_name_result:
        result_max = []
        country_list = df_data[(df_data['sector'] == Sector_Name) & (df_data['city'].str.contains(x))]['country'].drop_duplicates().tolist()
        for y in country_list:          
            value_data = df_data[(df_data['sector'] == Sector_Name) & (df_data['city'].str.contains(x)) & (df_data['country'] == y)][Value_col].sum()
            result_max.append(value_data)
        country_name_result.append(country_list[result_max.index(max(result_max))]) 
    
    prev_x = ''    
    for x,y in zip(city_name_result,country_name_result):
        index_df_all = df_all[(df_all['city'] == x) &(df_all['sector'] == Sector_Name) &(df_all['date'] >= first_date) &(df_all['date'] <= last_date)].index.tolist()
        index_df_data = df_data[(df_data['city'].str.contains(x)) & (df_data['country'] == y) &(df_data['sector'] == Sector_Name) &(df_data['date'] >= first_date) &(df_data['date'] <= last_date)].index.tolist()
        for t,h in zip(index_df_data,index_df_all):
            if df_data.loc[t,Value_col] < df_all.loc[h,Value_col]:
                df_data.loc[t,Value_col] = df_all.loc[h,Value_col]
                if x != prev_x:
                    prev_x = x
    
    df_data.to_csv(Path_3 +'carbon-monitor-cities-'+ output_name + '-' + output_version +'-transport-corrected.csv', index = False, encoding = 'utf_8_sig')
    # save by country
    list_country = df_data['country'].drop_duplicates().tolist()
    for ctry in list_country:   
        df_data_ctry = df_data.loc[df_data['country']==ctry]
        df_data_ctry.to_csv(Path_3 + 'carbon-monitor-cities-%s-v0319.csv' %(ctry), index = False, encoding = 'utf_8_sig')
    
    print('Total number of countires:', len(list_country))
    print('Total number of cities:', len(df_data['city'].drop_duplicates().tolist()))
    
#%% Step 1  Compute city emissions and correct for residential sector
city_spatialsum_emission = np.zeros([nday,len(sectname)])

icity = 0
num_city = 1500
for current_city_index in city_index[icity : icity+num_city]:
    city_mask = np.zeros((city_rows, city_cols))
    for i in range(city_rows):
        for j in range(city_cols):
            if city_array[i,j] == current_city_index:
                city_mask[i,j] = 1
    for year in years:
        for isect in [0,1,2,3,4]:
            day = 0
            for imon in range(12):
                nc_obj = Dataset('/data/CarbonMonitor_%i_%s/CarbonMonitor_%i_%s_%s_m%2.2i.nc' % (isect,sectname[isect],isect,sectname[isect],year,imon+1))                   
                for iday in range(mdays[imon]):
                    if imon==1 and iday == 28 and year!="y2020" : # Feb 29 for years other than 2020
                        daily_gridded_emission = nc_obj.variables[u'emission'][iday-1, :, :]
                    else:         
                        daily_gridded_emission = nc_obj.variables[u'emission'][iday, :, :]
                    city_emi_G = city_mask*daily_gridded_emission                       
                    city_spatialsum_emission[day, isect] = np.nansum(city_emi_G)                                    
                    day += 1       
                nc_obj.close()

        city_spatialsum_emission[:, 0] = correct_outliers(city_spatialsum_emission[:, 0])
        city_spatialsum_emission[:, 1] = correct_outliers(city_spatialsum_emission[:, 1])
        city_spatialsum_emission[:, 2] = correct_outliers(city_spatialsum_emission[:, 2])
        city_spatialsum_emission[:, 3] = correct_outliers(city_spatialsum_emission[:, 3])
        city_spatialsum_emission[:, 4] = correct_outliers(city_spatialsum_emission[:, 4]) 

        country = index_country_prov_city[icity,1]
        city = index_country_prov_city[icity,2]
        if FUA_type == 'GADM' or FUA_type == 'FUA':
            city = index_country_prov_city[icity,3]
            
        # Convert Unit from C to KtCO2/day
        data = city_spatialsum_emission*44*24/(12*1000000)
        
        days = np.arange(1, 367, 1)
        if year=='y2019' or year=='y2021':
            data= np.delete(data, 59, axis=0)
            days = np.arange(1, 366, 1)
        
        data = residential_correction(data, city, country, year, days)
        save_CM_cities(data, 5, country, city, year, output_path)     
    print('completed ' + str(city) +' of '+ str(country))
    icity += 1   
      

#%% Step 2  Merge cities into one file
merged = merge_all_cities_in_folder(Path_merged_data, path_file_to_merge)

#%% Step 3 Correction for transportation sector and save by country
transport_correction_save_by_contry(Path_tomtom, Path_merged_data, merged_output_path, output_name, output_version)
  








   