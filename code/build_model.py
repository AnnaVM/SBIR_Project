import cPickle as pickle
import sys
sys.path.append('../code')
from sklearn.cross_validation import KFold

from prepare_data import subset_data
from model import Model

def build_model():
    df = subset_data('dod', 2012, '/Users/AnnaVMS/Desktop/test2')

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
    model_trained.perform_GradientBoosting()
    #get model
    model = model_trained.model_GB
    #get the columns
    list_columns = model_trained.list_columns[:]
    list_columns.remove('to_phase_II')


    return tfidf_vectorizer, NMF, scaler, model, list_columns


if __name__ == '__main__':
    tfidf_vectorizer, NMF, scaler, model, list_columns = build_model()
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
