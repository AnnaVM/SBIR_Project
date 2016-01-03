import pandas as pd
import numpy as np

##############################
###   UTILITIES
##############################
#utilities are functions used to i) modify data

def YN(label):
    '''
    meant for Yes/No (as 'Y'/'N') labels
    transforms it into INT (1/0)

    useful in an apply method
    '''
    if label == 'Y':
        return 1
    else :
        return 0

def modify_column_YN(df, col_name, new_col_name):
    '''
    parameters:
    -----------
    df: pandas DataFrame with a column "col_name" with 'Y'/'N' values
    col_name and new_col_name are STR

    results:
    --------
    adds a new column (named 'new_col_name') with 1 for 'Y' and '0' for 'N'
    '''
    df[new_col_name] = df[col_name].apply(YN)

#utilities are functions used to ii) prepara data
def set_label_topics(df, col_name, tfidf_vectorizer, model_NMF):
    '''
    create a column with topic number
    parameters:
    ---------------
    df: pandas DataFrame to which a column should be added
    col_name as STR
    model_NMF and tfidf_vectorizer
    '''
    X_values = df[col_name]
    vector = tfidf_vectorizer.transform(X_values)
    W = model_NMF.transform(vector)
    labels = np.argmax( W, axis=1)
    df['topic'] = labels

def feature_engineering(df, training=False, list_columns=None):
    '''
    Gets the data ready for modeling (boolean to int, dummy variables)

    parameters:
    -----------
    df: a pandas DataFrame, must have 'Abstract' and 'Topic', as well as
    'Hubzone Owned','Socially and Economically Disadvantaged','Woman Owned'
    training: boolean, indicates if the feature engineering is done on the
        training set (list_columns need to be updated)

    if training is True:
    -----------------
    updated: list_columns has the topics as dummy
    '''
    #feature engineering: abstract length in characters
    df['Abstract Length'] = df['Abstract'].apply(len)

    #Y/N columns -> 1 for Yes, 0 for No
    list_YN_columns= ['Hubzone Owned',
                      'Socially and Economically Disadvantaged',
                      'Woman Owned']
    for label in list_YN_columns:
        new_label = label + ' as_int'
        modify_column_YN(df, label, new_label)

    #get dummy
    df_dummy = pd.get_dummies(df['topic'])

    for i in xrange( len(df_dummy.columns) ):
        df[ 'topic '+str(i) ] = df_dummy[i]
        if training:
            list_columns.insert(0, 'topic '+str(i))
#utilities are functions used to iii) assess models

def feature_engineering(df, model_NMF, training=False, list_columns=None):
    '''
    Gets the data ready for modeling (boolean to int, dummy variables)

    parameters:
    -----------
    df: a pandas DataFrame, must have 'Abstract' and 'Topic', as well as
    'Hubzone Owned','Socially and Economically Disadvantaged','Woman Owned'
    model_NMF: a NMF object
    training: boolean, indicates if the feature engineering is done on the
        training set (list_columns need to be updated)

    if training is True:
    -----------------
    updated: list_columns has the topics as dummy
    '''
    #feature engineering: abstract length in characters
    df['Abstract Length'] = df['Abstract'].apply(len)

    #Y/N columns -> 1 for Yes, 0 for No
    list_YN_columns= ['Hubzone Owned',
                      'Socially and Economically Disadvantaged',
                      'Woman Owned']
    for label in list_YN_columns:
        new_label = label + ' as_int'
        modify_column_YN(df, label, new_label)

    #get dummy
    df_dummy = pd.get_dummies(df['topic'])

    for i in xrange( model_NMF.n_components ):
        if i in df_dummy:
            df[ 'topic '+str(i) ] = df_dummy[i]
        else:
            df[ 'topic '+str(i) ] = 0
        if training:
            list_columns.insert(0, 'topic '+str(i))


def predictions(labels, predicted_labels):
    '''
    parameters:
    -----------
    labels: as LST values of the actual labels
    predicted_labels: as LST values given by the model

    returns:
    --------
    tp, tn, fp, fn
    '''
    labels = np.array(labels)
    predicted_labels = np.array(predicted_labels)
    tp = sum((labels == predicted_labels) & (labels == 1))
    tn = sum((labels == predicted_labels) & (labels == 0))
    fp = sum((labels != predicted_labels) & (predicted_labels == 1))
    fn = sum((labels != predicted_labels) & (predicted_labels == 0))
    return tp, tn, fp, fn

def print_results(tp, tn, fp, fn):
    '''
    simple visualization function of predictions
    '''
    print 'tp | ', tp
    print 'tn | ',tn
    print 'fp | ', fp
    print 'fn | ', fn
    accuracy = (tp+tn)*1./(tp+tn+fp+fn)
    recall = tp*1./(tp + fn)
    precision = tp*1./(tp + fp)
    print 'accuracy : ',accuracy
    print 'recall : ', recall
    print 'precision : ', precision
