from datetime import date
import covidcast
import pandas as pd

data_source = 'hospital-admissions'
signal = 'smoothed_adj_covid19_from_claims'
#
df = covidcast.signal(data_source, 'smoothed_covid19_from_claims', start_day=date(2020, 3, 1), geo_type='state')
df = df.groupby('time_value').aggregate({'value': 'mean'}).rename(columns={'value': 'hospital admission '
                                                                                    '(percentage of new admissions '
                                                                                    'that are covid)'})

#
temp = covidcast.signal(data_source, 'smoothed_adj_covid19_from_claims', start_day=date(2020, 3, 1), geo_type='state')
temp = temp.groupby('time_value').aggregate({'value': 'mean'}).rename(columns={'value': 'hospital admission-adj '
                                                                                        '(percentage of new admissions '
                                                                                        'that are covid)'})
df = pd.merge(df, temp, left_on='time_value', right_on='time_value', how='inner')

###
data_source = 'indicator-combination'
temp = covidcast.signal(data_source, 'confirmed_incidence_num', start_day=date(2020, 3, 1), geo_type='state')
temp = temp.groupby('time_value').aggregate({'value': 'sum'}).rename(columns={'value': 'confirmed cases'})
df = pd.merge(df, temp, left_on='time_value', right_on='time_value', how='inner')

###
temp = covidcast.signal(data_source, 'confirmed_cumulative_num', start_day=date(2020, 3, 1), geo_type='state')
temp = temp.groupby('time_value').aggregate({'value': 'sum'}).rename(columns={'value': 'total confirmed cases'})
df = pd.merge(df, temp, left_on='time_value', right_on='time_value', how='inner')

###
data_source = 'indicator-combination'
temp = covidcast.signal(data_source, 'deaths_incidence_num', start_day=date(2020, 3, 1), geo_type='state')
temp = temp.groupby('time_value').aggregate({'value': 'sum'}).rename(columns={'value': 'death cases'})
df = pd.merge(df, temp, left_on='time_value', right_on='time_value', how='inner')

###
data_source = 'indicator-combination'
temp = covidcast.signal(data_source, 'deaths_cumulative_num', start_day=date(2020, 3, 1), geo_type='state')
temp = temp.groupby('time_value').aggregate({'value': 'sum'}).rename(columns={'value': 'total death cases'})
df = pd.merge(df, temp, left_on='time_value', right_on='time_value', how='inner')

print('a')
df.to_csv('covid-forecast.csv')
