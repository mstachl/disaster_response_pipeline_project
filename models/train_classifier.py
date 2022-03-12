import sys
import pandas as pd
import re

from nltk.tokenize import word_tokenize
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline 
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import nltk
import pickle
nltk.download(['punkt','stopwords','wordnet'])


def load_data(database_filepath):
    """
    Load sqlite database file and split it into independent variable X 
    (message) and dependent variable Y (categories).

    Parameters
    ----------
    database_filepath : STR
        PATH TO DATABASE FILE.

    Returns
    -------
    X : DATAFRAME
        MESSAGE DATAFRAME.
    Y : DATAFRAME
        DESCRIPTION.
    category_names : DATAFRAME
        CATEGORIES DATAFRAME.

    """
    # Set up db engine
    sqlite_db = 'sqlite:///' + database_filepath
    engine = create_engine(sqlite_db)
    
    # Connect to database and load Message table into Dataframe
    df = pd.read_sql_table('Messages', con = engine.connect())
    
    # Extract category names from the dataframe
    category_names = list(df.columns)[4:]
    
    # Extract dependent and independent variables
    X = df['message']  # Message Column
    Y = df[category_names] # Classification label
    
    return X,Y,category_names


def tokenize(text):
    """
    Split text into words and return the root form of the words

    Parameters
    ----------
    text : STR
        THE MESSAGE.

    Returns
    -------
    lemm : LIST(STR)
        A LIST OF THE ROOT FORM OF MESSAGE WORDS.

    """
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize text
    words = word_tokenize(text)
    
    # Remove stop words
    stop = stopwords.words("english")
    words = [t for t in words if t not in stop]
    
    # Lemmatization
    lemm = [WordNetLemmatizer().lemmatize(w) for w in words]
    
    return lemm


def build_model():
    """
    Build a pipeline with a Bag-of-Words Wordcounter, TF-IDF and Adaboost 
    Classifier parameterized using parameters obtained from GridSearch.

    Returns
    -------
    gridsearch_pipeline : GRIDSEARCHCV
        MODEL CONTAINING NLP PIPELINE.

    """
    # Create pipeline with Bag-of-Words, TF-IDF and Adaboost Classifier
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf',  MultiOutputClassifier(AdaBoostClassifier()))
        
    ])
    
    # Create Grid search parameters for Adaboost Classifier   
    parameters = {
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [10,20,50],
    }

    gridsearch_pipeline = GridSearchCV(pipeline, param_grid = parameters)
    return gridsearch_pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Predict model results on test set. Print F1-score, precision and recall
    for each feature and print overall accuracy on test values.

    Parameters
    ----------
    model : GRIDSEARCHCV
        THE NLP PIPELINE (MODEL).
    X_test : DATAFRAME
        THE TEST MESSAGES.
    Y_test : DATAFRAME
        THE TEST CATEGORIES.
    category_names : LIST(STR)
        LIST OF CATEGORIES, EQUAL TO COLUMN NAMES OF Y_TEST.

    Returns
    -------
    None.

    """
    # Prediction with Adaboost Classifier  
    y_pred = model.predict(X_test)
    # Plot classification reports(f1-score, precision, recall)  for each feature
    for idx, col in enumerate(Y_test):
        print('Feature: {}'.format(col))
        print(classification_report(Y_test[col], y_pred[:, idx]))
    # compute and plot model accuracy   
    accuracy = (Y_test.values == y_pred).mean()       
    print('Model accuracy: {}'.format(accuracy))    


def save_model(model, model_filepath):
    """
    Exports the model as a pickle file.

    Parameters
    ----------
    model : ESTIMATOR
        THE ADABOOST MODEL.
    model_filepath : STR
        DESIRED FILE LOCATION FOR STORED MODEL.

    Returns
    -------
    None.

    """
    # Export model as pickle file
    with open (model_filepath, 'wb') as file:
        pickle.dump(model, file)


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