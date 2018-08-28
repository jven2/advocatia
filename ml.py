import numpy as np
import pandas as pd
import pickle
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

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
create_coord_array(adv_data_df)
Uses the coordinate data in the Advocatia dataset to create a plot of location vs medicaid eligibility
"""
def create_coord_array(adv_data_df):
    val_arr = np.zeros((adv_data_df.shape[0],4))
    val_arr[:,0]=adv_data_df["lat"]
    val_arr[:,1]=adv_data_df["lng"]
    val_arr[:,2]=adv_data_df["MedicaidEligible"]
    val_arr[:,3]=adv_data_df["AgeAsOfNow"]
    val_arr = val_arr[~np.isnan(val_arr).any(axis=1)]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(val_arr[:,0],val_arr[:,1],val_arr[:,2], c=val_arr[:,3], marker='.')
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
    feature_train_arr, feature_test_arr = np.split(feature_arr,[4800])
    value_train_arr, value_test_arr = np.split(value_arr,[4800])
    return (feature_train_arr, value_train_arr, feature_test_arr, value_test_arr)

def preprocess_data(arr_tuple):
    scaler = preprocessing.StandardScaler().fit(arr_tuple[0].astype('float'))
    scaled_train=scaler.transform(arr_tuple[0].astype('float'))
    scaled_test=scaler.transform(arr_tuple[2].astype('float'))
    return (scaled_train,arr_tuple[1].astype('int'),scaled_test,arr_tuple[3].astype('int'))

def logisticregression_model(arr_tuple):
    logreg = linear_model.LogisticRegression()
    logreg.fit(arr_tuple[0].astype('int'),arr_tuple[1].astype('int'))
    score = logreg.score(arr_tuple[2].astype('int'),arr_tuple[3].astype('int'))
    return score

def gaussian_naive_bayes(arr_tuple):
    gnb = GaussianNB()
    gnb.fit(arr_tuple[0].astype('float'),arr_tuple[1].astype('float'))
    score = gnb.score(arr_tuple[2].astype('float'),arr_tuple[3].astype('float'))
    print "Class priors: " + str(gnb.class_prior_)
    return score

def sgd_model(arr_tuple):
    sgd = linear_model.SGDClassifier(loss="hinge",penalty="l2", max_iter = 20000, tol = 1e-3, random_state=9, alpha=1e-5)
    sgd.fit(arr_tuple[0].astype('int'),arr_tuple[1].astype('int'))
    score = sgd.score(arr_tuple[2].astype('int'),arr_tuple[3].astype('int'))
    print "SGD Coefficients: " + str(sgd.coef_)
    print "SGD n_iter: " + str(sgd.n_iter_)
    return score

"""
Creates a 3D visualization of the datapoints
"""
def plot_visualization(feature_arr, value_arr):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for idx, row in enumerate(feature_arr):
        ax.scatter(row[3],row[0],row[2], c=str(value_arr[idx]), marker='.')
    plt.show()

def knn_model(arr_tuple):
    neigh=KNeighborsClassifier(n_neighbors=13)
    neigh.fit(arr_tuple[0].astype('float'),arr_tuple[1].astype('float'))
    score = neigh.score(arr_tuple[2].astype('float'),arr_tuple[3].astype('float'))
    return score

def rfc_model(arr_tuple):
    clf = RandomForestClassifier()
    clf.fit(arr_tuple[0],arr_tuple[1])
    score = clf.score(arr_tuple[2],arr_tuple[3])
    return score

def logreg_model(arr_tuple):
    lr = linear_model.LinearRegression()
    lr.fit(arr_tuple[0],arr_tuple[1])
    print lr.coef_
    print lr.score(arr_tuple[2],arr_tuple[3])

def knregressor(arr_tuple):
    neigh = KNeighborsRegressor(n_neighbors = 7)
    neigh.fit(arr_tuple[0],arr_tuple[1])
    file_Name = "model"
    fileObject = open(file_Name, 'wb')
    pickle.dump(neigh,fileObject)
    fileObject.close()

def run_classifiers(adv_data_df, county_level_df):
    feature_arr = create_feature_arr(adv_data_df, county_level_df)
    value_arr = create_value_arr(adv_data_df)
    arr_tuple = split_arrs(feature_arr, value_arr)
    arr_tuple = preprocess_data(arr_tuple)
    print "KNN score: " + str(knn_model(arr_tuple))
    print "SGD score: " + str(sgd_model(arr_tuple))
    print np.sum(arr_tuple[3])

def run_regressions(adv_data_df, county_level_df):
    feature_arr = create_feature_arr(adv_data_df, county_level_df)
    value_arr = create_value_arr(adv_data_df)
    arr_tuple = split_arrs(feature_arr, value_arr)
    arr_tuple = preprocess_data(arr_tuple)
    logreg_model(arr_tuple)
    knregressor(arr_tuple)


def main():
    adv_data_df,county_level_df = load_data("newvals.csv", "est16all.csv")
    run_classifiers(adv_data_df, county_level_df)
    #run_regressions(adv_data_df, county_level_df)





if __name__ == "__main__":
    main()