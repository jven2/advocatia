import numpy as np
import pandas as pd
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn import preprocessing

"""
load_data(adv_data_name, county_level_name)
    Loads the provided filenames into pandas dataframes, then returns the dataframes
    Calls subfunctions to create and manipulate the dataframes
"""
def load_data(adv_data_name, county_level_name):
    adv_data_df = load_adv_data(adv_data_name)
    county_level_df = load_county_data(county_level_name)
    return adv_data_df, county_level_df

"""
load_adv_data(adv_data_name)
    Loads the data from the Medicaid_Most_Recent_1000.txt file.
    Manipulates the data to remove spaces from the county column
    Replaces Male and Female in the Sex column with 0 and 1, respectively
    Filters out any rows where the income is zero or there is no county listed
    Returns a pandas dataframe
"""
def load_adv_data(adv_data_name):
    adv_data_df = pd.read_csv(adv_data_name)
    adv_data_df = adv_data_df.sample(frac=1)
    return adv_data_df

"""
load_county_data(county_level_name)
    Reads the relevant data from the est16all.csv file into a pandas dataframe
    Manipulates the county column to remove spaces
    Returns a pandas dataframe
"""
def load_county_data(county_level_name):
    county_level_df = pd.read_csv(county_level_name, quotechar='"', skipinitialspace=True, header=3, usecols=[2,3,7,22])
    county_level_df.columns = ['state_id', 'county', 'poverty_percent','median_income']
    county_level_df['county'] = county_level_df['county'].str.replace(" ","")
    return county_level_df
"""
create_feature_arr(adv_data_df, county_level_df)
    This function creates the array of features (value used to predict)
    First, an ndarray (numpy array) of the required size is initialized to all zeros.
    Next, county level income values and poverty percentages corresponding to the county of each person in the advocatia data is placed into the array.
    Various other values from the advocatia data are placed into the array.
    Returns a numpy ndarray
"""
def create_feature_arr(adv_data_df, county_level_df):
    number_of_features = 7
    feature_arr = np.zeros((adv_data_df.shape[0],number_of_features), np.int32)
    for index, row in adv_data_df.iterrows():
        #feature_arr.put(index*number_of_features, int((county_level_df.query('state_id == \'' + str(row['StateId']) + '\' and county == \'' + str(row['County']) +'\'').iloc[0]['median_income']).replace(',','')))
        feature_arr.put(index*number_of_features, float((county_level_df.query('state_id == \'' + str(row['StateId']) + '\' and county == \'' + str(row['County']) +'\'').iloc[0]['poverty_percent']).replace(',','')))
    feature_arr[:,1]=adv_data_df['AgeAsOfNow']
    feature_arr[:,2]=adv_data_df['HouseholdSize']
    feature_arr[:,3]=adv_data_df['UsCitizen']
    feature_arr[:,4]=adv_data_df['lat']
    feature_arr[:,5]=adv_data_df['lng']
    feature_arr[:,6]=adv_data_df['TotalIncome']
    return feature_arr

"""
create_value_arr(adv_data_df)
    Creates the class array (values we are trying to predict)
    Returns a numpy ndarray.
"""
def create_value_arr(adv_data_df):
    return adv_data_df["MedicaidEligible"]
    
"""
split_arrs(feature_arr, value_arr)
    Splits the feature and value arrays into training and testing subsets.
    Returns a tuple containing training and testing subsets of both arrays.
"""
def split_arrs(feature_arr, value_arr):
    feature_train_arr, feature_test_arr = np.split(feature_arr,[feature_arr.shape[0]-1])
    value_train_arr, value_test_arr = np.split(value_arr,[value_arr.shape[0]-1])
    return (feature_train_arr, value_train_arr, feature_test_arr, value_test_arr)

def split_arrs_test(feature_arr, value_arr):
    feature_train_arr, feature_test_arr = np.split(feature_arr,[4700])
    value_train_arr, value_test_arr = np.split(value_arr,[4700])
    return (feature_train_arr, value_train_arr, feature_test_arr, value_test_arr)


def preprocess_data(arr_tuple):
    scaler = preprocessing.StandardScaler().fit(arr_tuple[0].astype('float'))
    scaled_train=scaler.transform(arr_tuple[0].astype('float'))
    scaled_test=scaler.transform(arr_tuple[2].astype('float'))
    return (scaled_train,arr_tuple[1].astype('int'),scaled_test,arr_tuple[3].astype('int')), scaler

def knn_model(arr_tuple):
    neigh=KNeighborsClassifier(n_neighbors=13)
    neigh.fit(arr_tuple[0].astype('float'),arr_tuple[1].astype('float'))
    score = neigh.score(arr_tuple[2].astype('float'),arr_tuple[3].astype('float'))
    return score

def knregressor(arr_tuple, scaler):
    neigh = KNeighborsRegressor(n_neighbors = 13)
    neigh.fit(arr_tuple[0],arr_tuple[1])
    file_Name = "model.pkl"
    fileObject = open(file_Name, 'wb')
    pickle.dump((neigh,scaler),fileObject)
    fileObject.close()

def run_classifiers(adv_data_df, county_level_df):
    feature_arr = create_feature_arr(adv_data_df, county_level_df)
    value_arr = create_value_arr(adv_data_df)
    arr_tuple = split_arrs_test(feature_arr, value_arr)
    arr_tuple, scaler = preprocess_data(arr_tuple)
    print "KNN score: " + str(knn_model(arr_tuple))
    print np.sum(arr_tuple[3])

def run_regressions(adv_data_df, county_level_df):
    feature_arr = create_feature_arr(adv_data_df, county_level_df)
    value_arr = create_value_arr(adv_data_df)
    arr_tuple = split_arrs(feature_arr, value_arr)
    arr_tuple, scaler = preprocess_data(arr_tuple)
    knregressor(arr_tuple, scaler)


def main():
    adv_data_df,county_level_df = load_data("data_with_latlng.csv", "est16all.csv")
    #run_classifiers(adv_data_df, county_level_df)
    run_regressions(adv_data_df, county_level_df)





if __name__ == "__main__":
    main()