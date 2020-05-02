import sys
import pandas as pd
import numpy as np

def load_data(messages_filepath, categories_filepath):
    '''
    INPUT
    file paths of the message and categories files in cvs format

    OUTPUT
    a dataframe contains both dataset
    '''

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, how='inner')    

    return df


def clean_data(df):
    '''
    INPUT
    a dataframe with both messages and categories for data cleaning

    OUTPUT
    cleaned dataframe, with new expanding columns for each message category
    '''
    # Split `categories` into separate category columns
    categories = df.categories.str.split(';', expand = True)
    new_names = pd.Series(categories.loc[0].values).str.split('-', expand = True)[0].values
    new_names = dict(zip(np.arange(categories.shape[0]), new_names))
    # rename the new splitted columns
    categories = categories.rename(columns = new_names)

    # Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        # using regex from https://stackoverflow.com/questions/37683558/pandas-extract-number-from-string
        categories[column] = categories[column].str.extract('(\d+)').astype(int)

    # Replace `categories` column in `df` with new category columns
    df = df.drop(columns = 'categories')
    df = pd.concat([df, categories], axis = 1)

    # remove duplicated rows
    df = df.drop_duplicates(subset = 'id')
    
    # set labels in the 'related' category from 2 to 0
    # correcting these mislabels is vital for the ML pipeline processing
    df.loc[df['related'] > 1,'related'] = 0

    # drop this column with all the labels are 0
    df = df.drop(columns = 'child_alone')

    return df

def save_data(df, database_filepath):
    '''
    INPUT
    cleaned dataframe and the filepath for the SQL database for saving the dataframe

    OUTPUT
    None
    '''
    from sqlalchemy import create_engine   
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df.to_sql('RawDataClean', engine, if_exists = 'replace', index=False) 
    engine.dispose()



def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()