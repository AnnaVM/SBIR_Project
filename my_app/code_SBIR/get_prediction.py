from utilities import set_label_topics, feature_engineering
import cPickle as pickle


def get_prediction(df, tfidf_vectorizer, model_NMF, list_columns, scaler, model):
    set_label_topics(df, col_name='Abstract',
                        tfidf_vectorizer=tfidf_vectorizer,
                        model_NMF=model_NMF)


    feature_engineering(df, model_NMF)
    test_df = df[list_columns]

    features = \
            scaler.transform(test_df.values)

    return model.predict_proba(features)[:,1]

if __name__ == '__main__':
    model = pickle.load(open('../data/model.pkl', 'rb'))
    tfidf_vectorizer = pickle.load(open('../data/tfidf_vectorizer.pkl', 'rb'))
    model_NMF = pickle.load(open('../data/NMF.pkl', 'rb'))
    scaler = pickle.load(open('../data/scaler.pkl', 'rb'))
    list_columns = pickle.load(open('../data/list_columns.pkl', 'rb'))

    df_2012_dod = pickle.load(open('../data/df_2012_dod.pkl', 'rb'))
    prediction = get_prediction(df_2012_dod,
                                tfidf_vectorizer,
                                model_NMF,
                                list_columns,
                                scaler,
                                model)
    df_2012_dod['probability'] = prediction
    #df = subset_data('dod', 2012, '/Users/AnnaVMS/Desktop/test2')
    #df = pd.DataFrame(df.iloc[0:1])
    #df.pop('to_phase_II')
    #print get_prediction(df, tfidf_vectorizer, model_NMF, list_columns,
                        #scaler, model)
