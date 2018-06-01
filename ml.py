import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
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
    adv_data_df['County'] = adv_data_df['County'].str.replace(" ","")
    adv_data_df = adv_data_df[adv_data_df['TotalIncome'] != 0]
    adv_data_df['County'].replace('',np.nan, inplace=True)
    adv_data_df.dropna(subset=['County'],inplace=True)
    adv_data_df['Sex'].replace(['Female','Male'], [1,0], inplace=True)
    adv_data_df = adv_data_df.reset_index(drop=True)
    return adv_data_df

def create_coord_array(adv_data_df):
    val_arr = np.zeros((adv_data_df.shape[0],3))
    val_arr[:,0]=adv_data_df["Latitude"]
    val_arr[:,1]=adv_data_df["Longitude"]
    val_arr[:,2]=adv_data_df["MedicaidEligible"]
    val_arr = val_arr[~np.isnan(val_arr).any(axis=1)]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(val_arr[:,0],val_arr[:,1],val_arr[:,2], c='r', marker='.')
    plt.show()

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
    number_of_features = 6
    feature_arr = np.zeros((adv_data_df.shape[0],number_of_features), np.int32)
    for index, row in adv_data_df.iterrows():
        feature_arr.put(index*number_of_features, int((county_level_df.query('state_id == \'' + str(row[11]) + '\' and county == \'' + str(row[10]) +'\'').iloc[0]['median_income']).replace(',','')))
        feature_arr.put(index*number_of_features + 1, float((county_level_df.query('state_id == \'' + str(row[11]) + '\' and county == \'' + str(row[10]) +'\'').iloc[0]['poverty_percent']).replace(',','')))
    feature_arr[:,2]=adv_data_df['Sex']
    feature_arr[:,3]=adv_data_df['AgeAsOfNow']
    feature_arr[:,4]=adv_data_df['HouseholdSize']
    feature_arr[:,5]=adv_data_df['UsCitizen']
    return feature_arr

"""
create_value_arr(adv_data_df)
    Creates the class array (values we are trying to predict)
    Returns a numpy ndarray.
"""
def create_value_arr(adv_data_df):
    return adv_data_df.values[:,1]
    
"""
split_arrs(feature_arr, value_arr)
    Splits the feature and value arrays into training and testing subsets.
    Returns a tuple containing training and testing subsets of both arrays.
"""
def split_arrs(feature_arr, value_arr):
    feature_train_arr, feature_test_arr = np.split(feature_arr,[300])
    value_train_arr, value_test_arr = np.split(value_arr,[300])
    print feature_train_arr.shape
    print feature_test_arr.shape
    return (feature_train_arr, value_train_arr, feature_test_arr, value_test_arr)

def logisticregression_model(arr_tuple):
    logreg = linear_model.LogisticRegression()
    logreg.fit(arr_tuple[0].astype('int'),arr_tuple[1].astype('int'))
    score = logreg.score(arr_tuple[2].astype('int'),arr_tuple[3].astype('int'))
    print score

def gaussian_naive_bayes(arr_tuple):
    gnb = GaussianNB()
    gnb.fit(arr_tuple[0].astype('float'),arr_tuple[1].astype('float'))
    score = gnb.score(arr_tuple[2].astype('float'),arr_tuple[3].astype('float'))
    print score

def sgd_model(arr_tuple):
    sgd = linear_model.SGDClassifier(loss="hinge",penalty="l2", max_iter = 20000, tol = 1e-3, random_state=9, alpha=1e-6)
    sgd.fit(arr_tuple[0].astype('int'),arr_tuple[1].astype('int'))
    score = sgd.score(arr_tuple[2].astype('int'),arr_tuple[3].astype('int'))
    print score

def plot_visualization(feature_arr, value_arr):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for idx, row in enumerate(feature_arr):
        ax.scatter(row[3],row[0],row[2], c=str(value_arr[idx]), marker='.')
    plt.show()

def knn_model(arr_tuple):
    neigh=KNeighborsClassifier(n_neighbors=5)
    neigh.fit(arr_tuple[0].astype('float'),arr_tuple[1].astype('float'))
    score = neigh.score(arr_tuple[2].astype('float'),arr_tuple[3].astype('float'))
    print score

def main():
    adv_data_df,county_level_df = load_data("Medicaid_Most_Recent_1000.txt", "est16all.csv")
    feature_arr = create_feature_arr(adv_data_df, county_level_df)
    value_arr = create_value_arr(adv_data_df)
    arr_tuple = split_arrs(feature_arr, value_arr)
    create_coord_array(adv_data_df)
    #knn_model(arr_tuple)
    #sgd_model(arr_tuple)
    #plot_visualization(feature_arr, value_arr)
    #gaussian_naive_bayes(arr_tuple)
    #logisticregression_model(arr_tuple)


if __name__ == "__main__":
    main()