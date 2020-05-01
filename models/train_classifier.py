import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(database_filepath):
    '''
    INPUT
    file path of the database

    OUTPUT
    X: message and genre data (model input)
    Y: labels of the message (target)
    category_names: names of the labels
    '''
    from sqlalchemy import create_engine
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("SELECT * FROM RawDataClean", engine)
    engine.dispose()

    categories = df.drop(columns = ['id', 'message', 'original', 'genre'])
    X, Y, category_names = df[['message', 'genre']], categories.to_numpy(), categories.columns.values

    return X, Y, category_names


import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

def tokenize(text):
    '''
    INPUT
    str variables

    OUTPUT
    tokenized words
    '''
    import re # using regex here

    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # tokenize text
    tokens = word_tokenize(text)
    # lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens



def build_model():
    '''
    INPUT
    None

    OUTPUT
    the pipeline for building the model
    '''
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder

    # use one-hot encoding for 'genre' column
    column_trans = ColumnTransformer(
    [('genre_categroy', OneHotEncoder(dtype='int'),['genre']),
     ('message_tfidf', TfidfVectorizer(tokenizer = tokenize), 'message')])

    # build pipeline
    base_lr = LogisticRegression() 
    # we use one_vs_rest method to deal with multilabel problem
    pipe = Pipeline([
        ('pre_proc', column_trans),
        ('clf', OneVsRestClassifier(base_lr)),
        ])

    return pipe



def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT
    model pipeline built using build_model() method and fitted using the training data. 
    X_test, Y_test, category_names
    

    OUTPUT
    The jaccard_score calculated using sample averages
    '''
    from sklearn.metrics import jaccard_score
    from sklearn.metrics import classification_report
    
    y_pred = model.predict(X_test)   
    # calculate jaccard_score, which is a good measure for the multilabel classification problem
    ovr_jaccard_score = jaccard_score(Y_test, y_pred, average='samples')
   
    target_names = category_names
    report = classification_report(Y_test, y_pred, target_names = target_names, output_dict=True)
    # convert the report to pandas DataFrame
    report = pd.DataFrame(report).transpose()
    # save the evaluation metrics to the csv file
    report.to_csv('/data/performance_evaluation.csv')

    return ovr_jaccard_score

    


def save_model(model, model_filepath):
    '''
    INPUT
    model: model fitted using the training dataset
    model_filepath: filepath to save the model
    OUTPUT
    None
    '''
    import pickle
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)

    


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()