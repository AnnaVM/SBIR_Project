import cPickle as pickle
import pandas as pd
import numpy as np
import sys
sys.path.append('../code')
from sklearn.cross_validation import KFold

from prepare_data import subset_data
from get_prediction import get_prediction
from model import Model

def build_model(df):

    kf = KFold(len(df),10, shuffle=True)
    kf_iterator = kf.__iter__()
    train_index, test_index = kf_iterator.next()

    model_trained= Model(df, train_index, test_index)

    ##step 1
    model_trained.process_text('Abstract')
    #get the corresponding tfidf vectorizer and NMF
    tfidf_vectorizer = model_trained.tfidf_vectorizer
    NMF = model_trained.model_NMF
    ##step 2
    model_trained.prepare_data()
    #get the scaler
    scaler = model_trained.scaler
    ##step 3
    model_trained.perform_LogReg()
    #get model
    model = model_trained.model_LogReg
    #get the columns
    list_columns = model_trained.list_columns[:]
    list_columns.remove('to_phase_II')


    return tfidf_vectorizer, NMF, scaler, model, list_columns


def build_scoring_material(df, tfidf_vectorizer, model_NMF,
                           list_columns, scaler, model):
    predictions = get_prediction(df,
                                tfidf_vectorizer,
                                model_NMF,
                                list_columns,
                                scaler,
                                model)

    #generate a list with the thresholds for each percentile
    list_percentiles = []
    for i in xrange(1,10):
        list_percentiles.append(np.percentile(predictions, 10*i))
    list_percentiles = np.array(list_percentiles)

    #generate the list of percentiles for dataset
    percentiles = np.digitize(predictions, list_percentiles)

    df['percentile'] = percentiles
    percentage_success = df.groupby('percentile')['to_phase_II'].mean().values

    return list_percentiles, percentage_success

if __name__ == '__main__':
    df = subset_data('dod', 2012, '/Users/AnnaVMS/Desktop/test2')
    tfidf_vectorizer, NMF, scaler, model, list_columns = build_model(df)
    list_percentiles, percentage_success = build_scoring_material(df,
                                        tfidf_vectorizer, NMF,
                                        list_columns, scaler, model)
    with open('../data/model.pkl', 'w') as f:
        pickle.dump(model, f)
    with open('../data/scaler.pkl', 'w') as f:
        pickle.dump(scaler, f)
    with open('../data/NMF.pkl', 'w') as f:
        pickle.dump(NMF, f)
    with open('../data/tfidf_vectorizer.pkl', 'w') as f:
        pickle.dump(tfidf_vectorizer, f)
    with open('../data/list_columns.pkl', 'w') as f:
        pickle.dump(list_columns, f)
    with open('../data/list_percentiles.pkl', 'w') as f:
        pickle.dump(list_percentiles, f)
    with open('../data/percentage_success.pkl', 'w') as f:
        pickle.dump(percentage_success, f)
