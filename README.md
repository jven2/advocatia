Notes:

Before using the model, you must have anaconda for python 2.7 installed.
The download link is at https://www.anaconda.com/download/
This will install the required python libraries to run the model.

You will also need to install the google maps api for python,
which can be accomplished with the terminal command 
pip install -U googlemaps

Before running the model, you will have to register for a google maps api key.
This is a fairly simple process, and you can refer to Google's guides for help.
Once you have the api key, copy and paste it into the model_run.py file on line 10
where it says (key=''). You should limit the api key to only use the geocoding api.

Once all of those steps are completed, you are ready to use the model.
The command line arguments required to use it are explained by running
python model_run.py -h