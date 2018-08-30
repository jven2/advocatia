Notes:

Before using the model locally, you must have anaconda for python 2.7 installed.
The download link is at https://www.anaconda.com/download/
This will install the required python libraries to run the model.

You will also need to install the google maps api for python,
which can be accomplished with the terminal command 
pip install -U googlemaps

Before running the model, you will have to register for a google maps api key.
This is a fairly simple process, and you can refer to Google's guides for help.
Once you have the api key, copy and paste it into the predict.py file on line 69
where it says (key=''). You should limit the api key to only use the geocoding api.


Before building the model again, look over the dataset to ensure consistency.  There
were minor data inconsistencies in some of the previously uploaded datasets where
an extra value would be inserted in a small number of samples, causing the rest of
the data in that sample to be off by one and the geolookup to fail.

In order to build the model again, upload a new csv file into the working directory.
Next, put the file name into the latlng file on line 7 where it says file_name = "".
Additionally, put the google maps api key into the same file on line 32. Then, run
latlng.py. Once it has finished running, run ml.py, which will create a new model.pkl
file.
