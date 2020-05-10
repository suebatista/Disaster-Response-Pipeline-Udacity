import sys
import pandas as pd
import numpy as np


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

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import jaccard_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from joblib import dump
import seaborn as sns
import matplotlib.pyplot as plt
from tempfile import mkdtemp
from shutil import rmtree

class ML_classifier():

    def __init__(self, df, upsample = True, split = 0.7):
        '''
        INPUT:
        df: the dataframe for our modeling
        split: ratio of the train set over the entire set
        upsample: determine whether to upsample the training set
        
        OUTPUT:
        None
        '''
        self.clf = LogisticRegression(max_iter=500, class_weight = 'balanced') # default classifier
        self.df = df
        self.split = split
        self.upsample = upsample
        if self.upsample:
            self.up_sample()
        else:
            self.categories = self.df.drop(columns = ['id', 'message', 'original', 'genre'])
            self.category_names = self.categories.columns.values
            # divide input and output data
            x, y = self.df['message'].to_numpy(), self.categories.to_numpy()
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size = 1 - self.split, shuffle = True, random_state=0)


    def up_sample(self):
        '''
        Upsample under-represented messages in the training dataset
        INPUT:
        None

        OUTPUT:
        None
        '''
        self.df = self.df.sample(frac = 1, random_state = 0) # random shuffle df
        # self.df = self.df.reset_index()
        df_train, df_test = self.df.iloc[:int(self.split*len(self.df))], self.df.iloc[int(self.split*len(self.df)):]
        sub_cats = df_train[df_train['related'] == 1].drop(columns = ['id', 'message', 'original', 'genre', 'related'])
        
        # counts how many labels per category
        label_counts = sub_cats.sum().values
        # choose the boostrap sampling number equal to the most popular label
        self.upsample_num = np.sort(label_counts)[::-1][0]
        # choose the most 3 popular categories
        pop_label = list(sub_cats.sum().sort_values(ascending = False)[:3].index) 

        ## messages without any label in the most popular categories 
        sparse_msg = sub_cats[~sub_cats[pop_label].any(axis = 1)]
        # avoid messages with 'related' = 1 and rest = 0
        msg_to_sample = sparse_msg[(sparse_msg.sum(axis = 1) > 0)]
        # upsampling rare messages
        msg_up_sample = msg_to_sample.sample(n = self.upsample_num, replace = True, random_state = 0)
        # combine the sampled with unsampled
        df_train_sample = pd.concat([df_train.loc[msg_up_sample.index], df_train.loc[list(set(df_train.index.values) - set(msg_to_sample.index.values))]])
        

        ## divide train, test dataset
        categories_train_sample = df_train_sample.drop(columns = ['id', 'message', 'original', 'genre'])
        categories_test = df_test.drop(columns = ['id', 'message', 'original', 'genre'])
        self.df = pd.concat([df_train_sample, df_test])
        self.categories = self.df.drop(columns = ['id', 'message', 'original', 'genre'])
        self.category_names = self.categories.columns.values
        self.x_train, self.y_train = df_train_sample['message'].to_numpy(), categories_train_sample.to_numpy()
        self.x_test, self.y_test = df_test['message'].to_numpy(), categories_test.to_numpy()
      

    def build_model(self):
        '''
        INPUT:
        None

        OUTPUT:
        a ML pipeline by first transforming 
        the text data to TF-IDF matrix then follows a multilabel classifier
        '''
        self.cachedir = mkdtemp() # build a temp dir to store the transformer result

        # build pipeline
        self.pipe = Pipeline([
        ('tfidf_vect', TfidfVectorizer(tokenizer = tokenize)),
        ('clf', OneVsRestClassifier(self.clf))], memory=self.cachedir)

        return self.pipe

    
    def grid_search(self):
        '''
        Perform grid search over a set of parameters to improve the model accuracy

        INPUT: 
        None

        OUTPUT:
        None
        '''    
        model = self.build_model()
        parameters = {
            'tfidf_vect__max_df': (0.75, 1.0),
            #'tfidf_vect__max_features': (None, 5000, 10000),
            'tfidf_vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
            'clf__estimator__max_iter': (100, 500), 
            'clf__estimator__C': (0.1, 1.0, 10, 100),
            }

        grid_search = GridSearchCV(model, parameters, n_jobs=-1, verbose=1)
        grid_search.fit(self.x_train, self.y_train)
        self.best_parameters = grid_search.best_estimator_.get_params()
                
        # after grid search, update the model with new parameters
        self.pipe = Pipeline([
            ('tfidf_vect', self.best_parameters['tfidf_vect']),
            ('clf', self.best_parameters['clf']),
            ])

        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, self.best_parameters[param_name]))

        # remove temp files after search
        rmtree(self.cachedir)

    
    def fit(self):
        '''
        INPUT:
        None

        OUTPUT:
        None

        fit the model using training data
        '''     
        self.model = self.build_model()
        self.model.fit(self.x_train, self.y_train)

    
    def evaluate(self):
        '''
        INPUT:
        None

        OUTPUT:
        jaccard_score, it is a good performance measure of multilabel tasks
        '''
        self.y_pred = self.model.predict(self.x_test)
        self.jaccard_score = jaccard_score(self.y_test, self.y_pred, average='samples')
        report = classification_report(self.y_test, self.y_pred, target_names = self.category_names, output_dict=True, zero_division = 0)
        self.report = pd.DataFrame(report).transpose()
        
        print('The Jaccard score on the test data set is: {}'.format(self.jaccard_score))
                   
    
    def plot_dist(self):
        '''
        INPUT:
        None

        OUTPUT:
        a plot of Distribution of Category Labels
        '''
        cats = list(range(self.categories.shape[1])) # replace the name of categories into numbers
        counts = self.categories.sum().values
        sns.set(font_scale = 2)
        plt.figure(figsize=(16,9))
        ax= sns.barplot(cats, counts)
        plt.title("Distribution of Category Labels", fontsize=24)
        plt.ylabel('Number of Messages', fontsize=18)
        plt.xlabel('Message Labels', fontsize=18)
        #adding the text labels
        rects = ax.patches
        for rect, count in zip(rects, counts):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2, height + 5, count, ha='center', va='bottom', fontsize=12)
        
        plt.show()


    def save(self, model_filepath):
        '''
        INPUT
        model_filepath: filepath to save the model
        OUTPUT
        None

        save the trained model
        '''        
        dump(self.model, '{}'.format(model_filepath)) 
    
    
    def __repr__(self):
    
        '''Function to output the characteristics of the model
        
        INPUT:
        None
        
        OUTPUT:
        string: characteristics of the model
        
        '''
        
        return "A multilabel machine learning model using {} as the classifier".format(self.clf)


from sqlalchemy import create_engine
class Data_Process():
    '''
    This class serves to load the data from a SQL database, 
    '''
    def __init__(self):
        '''
        INPUT:  
        None        
        OUTPUT:
        None
        '''

    def load_data(self, database_filepath):
        '''
        INPUT       
        database_filepath: filepath of the SQL database

        OUTPUT
        splitted test and training data
        '''
        # load data from the SQL database     
        engine = create_engine('sqlite:///{}'.format(database_filepath))
        self.df = pd.read_sql("SELECT * FROM RawDataClean", engine)
        engine.dispose()
        return self.df

        

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]     
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        data = Data_Process()
        df = data.load_data(database_filepath)   
       
        print('Building model...')
        model = ML_classifier(df)
        model.build_model()
        
        print('Training model...')
        model.fit()
        
        print('Evaluating model...')
        model.evaluate()

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        model.save(model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()