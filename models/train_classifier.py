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
from joblib import dump
import seaborn as sns
import matplotlib.pyplot as plt

class ML_classifier():

    def __init__(self, df):
        '''
        INPUT:
        df: the dataframe for our modeling
        
        OUTPUT:
        None
        '''
        self.clf = LogisticRegression(max_iter=500) # default classifier
        self.df = df
        self.categories = self.df.drop(columns = ['id', 'message', 'original', 'genre'])
        self.category_names = self.categories.columns.values
        # divide input and output data
        self.x, self.y = self.df['message'].to_numpy(), self.categories.to_numpy()

    def build_model(self):
        '''
        INPUT:
        None

        OUTPUT:
        a ML pipeline by first transforming 
        the text data to TF-IDF matrix then follows a multilabel classifier
        '''
        # build pipeline
        self.pipe = Pipeline([
        ('tfidf_vect', TfidfVectorizer(tokenizer = tokenize)),
        ('clf', OneVsRestClassifier(self.clf))])

        return self.pipe
    
    def fit(self, split = 0.2):
        '''
        INPUT:
        None

        OUTPUT:
        None

        fit the model using training data
        '''
        self.split = split
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size = self.split, shuffle = True, random_state=0)
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
    
        """Function to output the characteristics of the model
        
        INPUT:
        None
        
        OUTPUT:
        string: characteristics of the model
        
        """
        
        return "A multilabel machine learning model using {} as the classifier".format(self.clf)


from sqlalchemy import create_engine
class data_process():
    '''
    This class serves to load the data from a SQL database, 
    and optionally upsample the data with under-represented 
    categories to improve the model performance.
    '''
    def __init__(self, sample = True):
        '''
        INPUT:  
        sample: whether or not to up-sample the data
        
        OUTPUT:
        None
        '''
        self.sample = sample

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

        if self.sample:
            return self.up_sample()            
        else:
            return self.df
                          
                             
    def up_sample(self):
        '''
        INPUT:
        None

        OUTPUT:
        dataframe upsampled using the more sophiscated method described in the notebook
        '''
        sub_cats = self.df[self.df['related'] == 1].drop(columns = ['id', 'message', 'original', 'genre', 'related'])
        # counts how many labels per category
        label_counts = sub_cats.sum().values
        # choose the boostrap sampling number equal to the most popular label
        self.upsample_num = np.sort(label_counts)[::-1][0]
        # choose the most 3 popular categories
        self.pop_label = list(sub_cats.sum().sort_values(ascending = False)[:3].index) 

        # messages without any label in the most popular categories 
        sparse_msg = sub_cats[~sub_cats[self.pop_label].any(axis = 1)]
        # avoid messages with 'related' = 1 and rest = 0
        msg_to_sample = sparse_msg[(sparse_msg.sum(axis = 1) > 0)]
        # upsampling 
        msg_up_sample = msg_to_sample.sample(n = self.upsample_num, replace = True, random_state = 0)
        self.df_sample = pd.concat([self.df.loc[msg_up_sample.index], self.df.loc[list(set(self.df.index.values) - set(msg_to_sample.index.values))]])

        return self.df_sample    
        

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]     
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        data = data_process()
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