## Introduction

The repo is a project in the [Udacity Data Scientist Nanodegree Program](https://www.udacity.com/course/data-scientist-nanodegree--nd025).

The dataset (in the data folder) comes from [appen](https://appen.com) which contains about 26k text messages from news, social media, and some other sources when some disasters happened. 

Our goal here is to build a machine learning model to identify if these messages are related to disaster or not, and further label the nature of these messages. This would be of great help for some disaster relief agencies. We have 36 labels for these messages in total. Note, however, these labels are not mutually exclusive. Hence it is a multi-label classification problem.

The most obvious feature of those data messages is they are highly imbalanced. 
Several categories getting very few labels. To improve the accuracy, we implement a up-sample
scheme before training. 

After building and training such a model, we can next launch a web service which can label new messages from users' input.


### Instructions:



1. Run the following commands in the project's root directory to set up the database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
      Note by default the training data will be up-sampled before training. You can change this setting in the train_classifier.py by setting `ML_classifier(df, sample = False)` when instantiating the ML model class. It should take less than a minute to train and save the model.

2. Run the following command in the app's directory to run the web app
    `python run.py`

3. Go to http://0.0.0.0:3001/ to use the web app to query your own message and see some visualizations about the original dataset.

4. If you are curious about the details of data processing and machine learning model building, you can check two jupyter notebooks in the main directory.


### Requirements

* *Python 3.5+*
* *NLTK* for natural language processing (converting text data into numerical data)
* *Pandas, Numpy, scikit-learn, sqlalchemy* for data processing and machine learning
* *Matplotlib, seaborn, plotly*  for data visualizations
* *Flask*, back-end of our minimalistic web app