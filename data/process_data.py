import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load input datafiles, join them and return a concatenated dataframe.

    Parameters
    ----------
    messages_filepath : STR
        PATH TO THE FIRST INPUT FILE (CSV).
    categories_filepath : TYPE
        PATH TO THE SECOND INPUT FILE (CSV).

    Returns
    -------
    df : DATAFRAME
        DATAFRAME CONCATENATING DATA FROM BOTH INPUT FILES.

    """
    # load message file into dataframe
    messages = pd.read_csv(messages_filepath, index_col='id')
    # load categories file into dataframe
    categories = pd.read_csv(categories_filepath, index_col='id')
    # join the two dataframes
    df = messages.join(categories)
    return df


def clean_data(df):
    """
    Perform cleaning operations on the dataframe. This includes handling
    categorical variables, dropping duplicates and formatting category columns.

    Parameters
    ----------
    df : DATAFRAME
        DATAFRAME TO BE CLEANED.

    Returns
    -------
    df : DATAFRAME
        CLEANED DATAFRAME.

    """
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';',expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = [x.split('-')[0] for x in row]
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # drop the original categories column from `df`
    df = df.drop(columns=["categories"])
    df = df.join(categories)
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    """
    Stores a given dataframe into a sql database. The table name is set to 
    'Messages'.

    Parameters
    ----------
    df : DATAFRAME
        DATAFRAME TO BE STORED.
    database_filename : STR
        TARGET FILENAME OF THE SQLITE DB.

    Returns
    -------
    None.

    """
    sqlite_db = 'sqlite:///' + database_filename
    engine = create_engine(sqlite_db)
    df.to_sql('Messages', engine, index=False)  


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