import numpy as np
import pandas as pd
import pickle
import argparse
import googlemaps
from sklearn.neighbors import KNeighborsRegressor


def lat_lng(street, city, state):
	gmaps = googlemaps.Client(key='AIzaSyBuHPwVk8twRke7pX2fg26lGQLjq1y7Myw')
	latlng = gmaps.geocode(street + ", " + city + ", " + state)[0]['geometry']['location']
	return latlng['lat'], latlng['lng']

def county_poverty(county_level_name, state, county):
    county_level_df = pd.read_csv(county_level_name, quotechar='"', skipinitialspace=True, header=3, usecols=[2,3,7])
    county_level_df.columns = ['state_id', 'county', 'poverty_percent']
    county_level_df['county'] = county_level_df['county'].str.replace(" ","")
    county = county.replace(" ","")
    return float((county_level_df.query('state_id == \'' + state + '\' and county == \'' + county +'\'').iloc[0]['poverty_percent']).replace(',',''))

def main():
	parser = argparse.ArgumentParser(description="Interact with the machine learning model")
	parser.add_argument("age", type=int, help="The patient's current age")
	parser.add_argument("household_Size", type=int, help="The number of people in the patient's household")
	parser.add_argument("us_citizen", type=int, help="Whether or not the patient is a US citizen. Enter 1 for True, 0 for False")
	parser.add_argument("street_address", help="The patient's street address. Must be inside quotations")
	parser.add_argument("city", help="The patient's city")
	parser.add_argument("county", help="The patient's county, including the word county, e.g. \"Hamilton County\". Must be inside quotations")
	parser.add_argument("state", help="The acronym for the patient's state, e.g. OH")
	args = parser.parse_args()
	lat,lng = lat_lng(args.street_address,args.city,args.state)
	county_poverty_percent = county_poverty("est16all.csv",args.state,args.county)
	model_object = open("model",'r')
	model = pickle.load(model_object)
	print model.predict(np.reshape([county_poverty_percent, args.age, args.household_Size, args.us_citizen, lat, lng],(1,-1)))
	model_object.close()


if __name__ == "__main__":
	main()