import sys
import time
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import AdaBoostClassifier
import pickle

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def load_data(database_filepath):
    print('Loading data ...')

    # load data from database
    engine_name = 'sqlite:///' + database_filepath
    engine = create_engine(engine_name)
    df = pd.read_sql_table('DisasterResponse',engine)

    #split data into X,y
    X = df['message']
    Y = df[df.columns.difference(['id','message','original','genre'])]

    #Get category_names
    category_names = list(Y.columns)

    return X, Y, category_names


def tokenize(text):
    #tokenize text
    tokens = word_tokenize(text)
    #initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    #iterate through each token
    for tok in tokens:
        # lemmatize, normalize case, and remove white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():

    # Create pipeline
    pipeline_ada = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
        ])),

        ('clf',  MultiOutputClassifier(AdaBoostClassifier()))
    ])

    # Set up GridSearchCV to improve model
    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2))
        }
    cv_ada = GridSearchCV(pipeline_ada, parameters)

    return cv_ada



def evaluate_model(model, X_test, Y_test, category_names):
    #Predict using model
    Y_pred = model.predict(X_test)

    #Evaluating Model,  Here we will set up how to calculate
    # Accuracy, Precision, Recall, and F1 Scores

    # accuracy is the total correct divided by the total to predict
    def calc_accuracy(actual, preds):
        '''
        INPUT
        preds - predictions as a numpy array or pandas series
        actual - actual values as a numpy array or pandas series

        OUTPUT:
        returns the accuracy as a float
        '''
        return np.sum(preds == actual)/len(actual)

    # precision is the true positives over the predicted positive values
    def calc_precision(actual, preds):
        '''
        INPUT
        (assumes positive = 1 and negative = 0)
        preds - predictions as a numpy array or pandas series
        actual - actual values as a numpy array or pandas series

        OUTPUT:
        returns the precision as a float
        '''
        total = len( np.intersect1d(np.where(preds == 1),np.where(actual==1)))
        pred_pos = (preds==1).sum()

        #check for division by zero
        if pred_pos == 0:
            result = 0
        else:
            result = total/(pred_pos)

        return result

    # recall is true positives over all actual positive values
    def calc_recall(actual, preds):
        '''
        INPUT
        preds - predictions as a numpy array or pandas series
        actual - actual values as a numpy array or pandas series

        OUTPUT:
        returns the recall as a float
        '''

        total = len( np.intersect1d(np.where(preds == 1),np.where(actual==1)))
        act_pos = (actual==1).sum()

        #check for division by zero
        if act_pos == 0:
            result = 0
        else:
            result = total/(act_pos)

        return result

    # f1_score is 2*(precision*recall)/(precision+recall))
    def calc_f1(preds, actual):
        '''
        INPUT
        preds - predictions as a numpy array or pandas series
        actual - actual values as a numpy array or pandas series

        OUTPUT:
        returns the f1score as a float
        '''
        rec = calc_recall(actual,preds)
        prec = calc_precision(actual,preds)

        #check for division by zero
        if prec+rec == 0:
            result = 0
        else:
            result = (2 *  (prec * rec)/(prec+rec))

        return result

    # Outputs the average accuracy,precision,recall, and f1 score
    def model_results(y_test,y_pred):
        #Create lists to append results
        accuracy = []
        precision = []
        recall = []
        f1_score = []

        # Show f1 score, precision, and recall for each category
        for col in list(range(y_pred.shape[1])):
            actual = y_test.iloc[:,col]
            pred = y_pred[:,col]
            accuracy.append(calc_accuracy(actual,pred))
            precision.append(calc_precision(actual,pred))
            recall.append(calc_recall(actual,pred))
            f1_score.append(calc_f1(pred,actual))
            print('Results for ',category_names[col])
            print('Accuracy',accuracy[col])
            print('Precision',precision[col])
            print('Recall',recall[col])
            print('F1_score',f1_score[col])


        model1_results = pd.DataFrame({'Accuracy': accuracy,
                                       'Precision': precision,
                                       'Recall': recall,
                                       'F1_Score': f1_score})
        results = model1_results.sum()/model1_results.shape[0]

        print('Overall Scores: ',results)

    # Show results
    print(model_results(Y_test,Y_pred))



def save_model(model, model_filepath):
    #Save as pickle file
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        # Improve adaboost model
        print('Current model training usually takes approximatly 30mins')
        start = time.time()
        model.fit(X_train,Y_train)

        end = time.time()
        print('Model training time: ',end - start)

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
