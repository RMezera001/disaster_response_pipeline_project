# Disaster Response Pipeline Project
This project from the Udacity Data Science Nano Degree program. The purpose of the project is to use previous messages related to natural disasters and create a model that can be used to identify messages that are important to emergency personnel during a disaster when resources are all ready strained.  

## Files of Interest

There are three major portions of this project.  

-  process_data.py
  -  This script will load the data(disaster_messages.csv  and disaster_categories.csv), merge, clean, and save it into a SQL database.  The process of creating this file can be seen in file 'ETL Pipeline Preparation' TPNYB/html.
-  train_classifier.py
  -  This script will load the SQL database created in the previous script and implement a natural language pipeline to categorize future messages into related categories.  The data will be trained using the data and the model will be saved to be used in the webpage implementation.  The process of creating this file can be seen in file 'ML Pipeline Preparation' TPNYB/html.
-  run.py
  -  This script launches the webpage with two graphs and the implementation of the NLP model.


## Installation

$  pip install plotly


## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
-  When using the Udacity IDE terminal use 'env | grep WORK' to get SPACEDOMAIN and SPACEID.
-  Use https://SPACEID-3001.SPACEDOMAIN when

## Complications

##### Model
Training the model using GirdSearchCV significantly increased the training time required.  In the interest of keeping training time down when running the training script, improving the model was sacrificed.  The current runtime in order to train the model is a little over 30 mins.  

##### Data
The data categories were very imbalanced so metrics and ability to predict under represented categories is negatively effected.

## Authors and acknowledgment

Author: Ryan Mezera

Acknowledgement:  Direction and templates were provided from the Udacity project files.
