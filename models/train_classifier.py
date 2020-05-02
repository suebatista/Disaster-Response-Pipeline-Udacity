import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



class ML_classifier():
    '''
    This Recommender uses FunkSVD to make predictions of exact ratings.  And uses either FunkSVD or a Knowledge Based recommendation (highest ranked) to make recommendations for users.  Finally, if given a movie, the recommender will provide movies that are most similar as a Content Based Recommender.
    '''
    def __init__(self):
        self.clf = LogisticRegression(max_iter=500) # default classifier

    def load_data(self, df):
        categories = df.drop(columns = ['id', 'message', 'original', 'genre'])
        x, y = df['message'].to_numpy(), categories.to_numpy()
    
        return train_test_split(x, y, test_size = 0.3, shuffle = True, random_state=0)

    def build_model(self, clf):
        # build pipeline
        self.clf = clf
        self.pipe = Pipeline([
        ('tfidf_vect', TfidfVectorizer(tokenizer = tokenize)),
        ('clf', OneVsRestClassifier(self.clf))])

        return self.pipe

    def evalute(self, df):        
        x_train, x_test, y_train, y_test = self.load_data(df)
        model = self.build_model()
        model.fit(x_train, y_train)
        self.y_pred = model.predict(x_test)
        self.jaccard_score = jaccard_score(y_test, y_pred, average='samples')
    
        target_names = df.drop(columns = ['id', 'message', 'original', 'genre']).columns.values
        report = classification_report(y_test, self.y_pred, target_names = target_names, output_dict=True, zero_division = 0)
        self.report_df = pd.DataFrame(report).transpose()
    
        return self.jaccard_score


class sample_data():
    '''
    This class provides two data up-sampling methods described in the jupyternotebook.
    The up_sample method is better 
    '''
    def __init__(self, df, threshold = 0.05):
        '''
        '''
        self.df = df
        self.threshold = threshold
        self.sub_cats = self.df[self.df['related'] == 1].drop(columns = ['id', 'message', 'original', 'genre', 'related'])
        label_counts = self.sub_cats.sum().values
        self.upsample_num = np.sort(label_counts)[::-1][0]
        criteria = label_counts/self.df.shape[0] < self.threshold
        self.sparse_label = list(self.sub_cats.columns[criteria])
                    
    
    def simple_sample(self):
        msg_simple_sample = self.df[self.df[self.sparse_label].any(axis = 1)].sample(n = self.upsample_num, replace = True, random_state = 0)
        df_simple_sample = pd.concat([msg_simple_sample, self.df[~self.df[self.sparse_label].any(axis = 1)]])
        
        return df_simple_sample
                     
        
    def up_sample(self):
        self.pop_label = list(self.sub_cats.sum().sort_values(ascending = False)[:3].index)        
        # messages without any label in those popular categories 
        sparse_msg = self.sub_cats[~self.sub_cats[self.pop_label].any(axis = 1)]
        msg_to_sample = sparse_msg[(sparse_msg.sum(axis = 1) > 0)]
        # upsampling 
        msg_up_sample = msg_to_sample.sample(n = self.upsample_num, replace = True, random_state = 0)
        df_sample = pd.concat([self.df.loc[msg_up_sample.index], self.df.loc[list(set(self.df.index.values) - set(msg_to_sample.index.values))]])
        
        return df_sample    
        
        
        
    
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

    X, Y, category_names = df['message'].to_numpy(), categories.to_numpy(), categories.columns.values

    return X, Y, category_names


# load NLP related modules and files
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
    # from sklearn.compose import ColumnTransformer
    # from sklearn.preprocessing import OneHotEncoder

    # use one-hot encoding for 'genre' column
    # column_trans = ColumnTransformer(
    # [('genre_categroy', OneHotEncoder(dtype='int'),['genre']),
    #  ('message_tfidf', TfidfVectorizer(tokenizer = tokenize), 'message')])

    # build pipeline
    base_lr = LogisticRegression(max_iter=500) 
    pipe = Pipeline([
    ('tfidf_vect', TfidfVectorizer(tokenizer = tokenize)),
    ('clf', OneVsRestClassifier(base_lr))])

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
   
    report = classification_report(Y_test, y_pred, target_names=category_names, output_dict=True, zero_division=0)
    # convert the report to pandas DataFrame
    report = pd.DataFrame(report).transpose()
    # save the evaluation metrics to the csv file
    report.to_csv('data/performance_evaluation.csv')

    return ovr_jaccard_score

    

def save_model(model, model_filepath):
    '''
    INPUT
    model: model fitted using the training dataset
    model_filepath: filepath to save the model
    OUTPUT
    None
    '''
    # import pickle
    # with open(model_filepath, 'wb') as f:
    #     pickle.dump(model, f)
    from joblib import dump
    dump(model, '{}'.format(model_filepath)) 

    
# def warn(*args, **kwargs):
#     pass
# import warnings
# warnings.warn = warn


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