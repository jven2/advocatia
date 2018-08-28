import googlemaps
import numpy as np
import pandas as pd
import json
import time
def load_data(adv_data_name):
	adv_data_df = load_adv_data(adv_data_name)
	return adv_data_df

def load_adv_data(adv_data_name):
	adv_data_df = pd.read_csv(adv_data_name)
	adv_data_df['County'] = adv_data_df['County'].str.replace(" ","")
	adv_data_df['RoundedAddress1'] = adv_data_df['RoundedAddress1'].str.lstrip()
	adv_data_df['City'] = adv_data_df['City'].str.lstrip()
	adv_data_df['Ethnicity'] = adv_data_df['Ethnicity'].str.replace(" ","")
	adv_data_df = adv_data_df[adv_data_df['TotalIncome'] != 0]
	adv_data_df['City'].replace('',np.nan, inplace=True)
	adv_data_df['County'].replace('',np.nan, inplace=True)
	adv_data_df['RoundedAddress1'].replace('',np.nan, inplace=True)
	adv_data_df.dropna(subset=['City'],inplace=True)
	adv_data_df.dropna(subset=['County'],inplace=True)
	adv_data_df.dropna(subset=['RoundedAddress1'],inplace=True)
	adv_data_df['Sex'].replace(['Female','Male'], [1,0], inplace=True)
	adv_data_df = adv_data_df.reset_index(drop=True)
	return adv_data_df

def main():
	gmaps = googlemaps.Client(key='AIzaSyBuHPwVk8twRke7pX2fg26lGQLjq1y7Myw')
	adv_data_df = load_data("10KData+Address.csv")
	lat_arr = np.zeros((adv_data_df.shape[0],1),np.float32)
	lng_arr = np.zeros((adv_data_df.shape[0],1),np.float32)
	print adv_data_df.shape[0]
	for i in range(0,adv_data_df.shape[0]):
		try:
			latlng = gmaps.geocode(adv_data_df.iloc[i,9] + ', ' + adv_data_df.iloc[i,10]+ ', ' + adv_data_df.iloc[i,12])[0]['geometry']['location']
			lat_arr.put(i,latlng['lat'])
			lng_arr.put(i, latlng['lng'])
		except IndexError:
			lat_arr.put(i,0)
			lng_arr.put(i,0)
		time.sleep(0.02)
	adv_data_df = adv_data_df.assign(lat=lat_arr)
	adv_data_df = adv_data_df.assign(lng=lng_arr)
	adv_data_df.to_csv('newvals.csv')


if __name__ == "__main__":
	main()