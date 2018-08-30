from sklearn.externals import joblib
#from boto.s3.key import Key
#from boto.s3.connection import S3Connection
from flask import Flask
from flask import request
from flask import json
import pandas as pd
import numpy as np
import googlemaps


BUCKET_NAME = 'your-s3-bucket-name'
MODEL_FILE_NAME = 'model.pkl'
MODEL_LOCAL_PATH = '' + MODEL_FILE_NAME

app = Flask(__name__)

@app.route('/', methods=['POST'])
def index():
  payload = json.loads(request.get_data().decode('utf-8'))
  prediction = predict(payload)
  data = {}
  data['data'] = prediction[-1]
  return json.dumps(data)

def load_model():
  """conn = S3Connection()
  bucket = conn.create_bucket(BUCKET_NAME)
  key_obj = Key(bucket)
  key_obj.key = MODEL_FILE_NAME

  contents = key_obj.get_contents_to_filename(MODEL_LOCAL_PATH)"""
  return joblib.load(MODEL_LOCAL_PATH)

def predict(payload):
  # Process your data, create a dataframe/vector and make your predictions
  pl = payload.keys()
  if (('street' in pl) and ('city' in pl) and ('state' in pl)): 
	lat,lng = lat_lng(payload['street'], payload['city'], payload['state'])
  else:
	lat = 0
	lng = 0
  if (('county' in pl) and ('state' in pl)):
	county_poverty_percent = county_poverty(payload['state'],payload['county'])
  else:
  	county_poverty_percent = 14.85
  if 'age' in pl:
  	age = payload['age']
  else:
  	age = 43.59
  if 'household_size' in pl:
  	household_size = payload['household_size']
  else:
  	household_size = 2
  if 'us_citizen' in pl:
  	us_citizen = payload['us_citizen']
  else:
  	us_citizen = 1
  if 'income' in pl:
  	income = payload['income']
  else:
  	income = 2000

  final_formatted_data = np.reshape([county_poverty_percent, age, household_size, us_citizen, lat, lng, income],(1,-1))
  model = load_model()
  return model[0].predict(model[1].transform(final_formatted_data.astype('float')))

def lat_lng(street, city, state):
	gmaps = googlemaps.Client(key='AIzaSyD02AHzjuYDfYElNA1YxPaLYODmc4V6SBM')
	latlng = gmaps.geocode(street + ", " + city + ", " + state)[0]['geometry']['location']
	return latlng['lat'], latlng['lng']

def county_poverty(state, county):
	county_level_df = pd.read_csv('est16all.csv', quotechar='"', skipinitialspace=True, header=3, usecols=[2,3,7])
	county_level_df.columns = ['state_id', 'county', 'poverty_percent']
	county_level_df['county'] = county_level_df['county'].str.replace(" ","")
	county = county.replace(" ","")
	return float((county_level_df.query('state_id == \'' + state + '\' and county == \'' + county +'\'').iloc[0]['poverty_percent']).replace(',',''))