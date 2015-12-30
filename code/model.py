import pandas as pd
import numpy as np

#for models
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler

############### UTILITIES
def YN(label):
    if label == 'Y':
        return 1
    else :
        return 0

def modify_column_YN(df, col_name, new_col_name):
    df[new_col_name] = df[col_name].apply(YN)

############### CLASS
class Model():

    def __init__(self, data_frame, train_index, test_index):
        self.train_index = train_index
        self.test_index = test_index
        self.list_columns = []
        self.train_data = data_frame.ix[train_index]
        self.test_data = data_frame.ix[test_index]

    def process_text(self,col_name):
        self.perform_vectorize_text()
        self.perform_NMF()
        self.set_label_topics(col_name)

    def perform_vectorize_text(self, col_name='Abstract', max_features=5000):
        '''

        Initiates tfidf_vectorizer, Vocab,
        '''
        #instantiate vectorizer, basic for now
        X_train = self.train_data[col_name]
        X_test = self.test_data[col_name]
        tfidf_vectorizer = TfidfVectorizer(stop_words='english',
                                            max_features=max_features)

        #fit the model on the train set
        tfidf_vectorizer.fit(X_train)

        self.X_train_vectorized = tfidf_vectorizer.transform(X_train)
        self.X_test_vectorized = tfidf_vectorizer.transform(X_test)

        self.tfidf_vectorizer = tfidf_vectorizer
        self.vocab = tfidf_vectorizer.get_feature_names()

    def perform_NMF(self, n_components=5):
        '''
        '''
        #instantiate NMF model
        model_NMF = NMF(n_components=n_components,
                        init='random', random_state=0)

        #fit the model on the train set
        model_NMF.fit(self.X_train_vectorized)

        self.model_NMF = model_NMF


    def get_top_n_words_for_topic(self, i_topic, n=10):
        '''
        INPUT i_topic
        '''
        topic_weight = self.model_NMF.components_[i_topic,:]
        index_topic = np.argsort(topic_weight)[::-1][:n]
        return np.array(self.vocab)[index_topic]

    def set_label_topics(self, col_name):
        '''
        create a column with topic number
        '''
        W = self.model_NMF.transform(self.X_train_vectorized)
        labels = np.argmax( W, axis=1)
        self.train_data['topic'] = labels

        W = self.model_NMF.transform(self.X_test_vectorized)
        labels = np.argmax( W, axis=1)
        self.test_data['topic'] = labels


    def prepare_LogReg(self):
        '''
        Gets the data ready for logistic regression
        (boolean to int, dummy variables)
        '''
        #feature engineering: abstract length in characters
        self.train_data['Abstract Length'] = \
                                        self.train_data['Abstract'].apply(len)
        self.test_data['Abstract Length'] = \
                                        self.test_data['Abstract'].apply(len)

        #Y/N columns -> 1 for Yes, 0 for No
        list_YN_columns= ['Hubzone Owned',
                          'Socially and Economically Disadvantaged',
                          'Woman Owned']
        for label in list_YN_columns:
            new_label = label + ' as_int'
            modify_column_YN(self.train_data, label, new_label)
            modify_column_YN(self.test_data, label, new_label)


        self.list_columns = ['Solicitation Year',
                             'Award Amount',
                             'Hubzone Owned as_int',
                             'Socially and Economically Disadvantaged as_int',
                             'Woman Owned as_int',
                             '# Employees',
                             'Abstract Length',
                             'to_phase_II']

        #get dummy
        train_dummy = pd.get_dummies(self.train_data['topic'])
        test_dummy = pd.get_dummies(self.test_data['topic'])

        for i in xrange( len(train_dummy.columns) ):
            self.train_data[ 'topic '+str(i) ] = train_dummy[i]
            self.test_data[ 'topic '+str(i) ] = test_dummy[i]
            self.list_columns.insert(0, 'topic '+str(i))

        train_df = self.train_data[self.list_columns]
        test_df = self.test_data[self.list_columns]

        scaler = StandardScaler()
        self.LogReg_Xtrain = \
            scaler.fit_transform(train_df[self.list_columns[:-1]].values)
        self.LogReg_Xtest = \
            scaler.transform(test_df[self.list_columns[:-1]].values)
        self.LogReg_ytrain = train_df[self.list_columns[-1]].values
        self.LogReg_ytest = test_df[self.list_columns[-1]].values

    def perform_LogReg(self, penalty='l2', C=1.0):
        model_LogReg = LogisticRegression(penalty=penalty, C=C)
        model_LogReg.fit(self.LogReg_Xtrain, self.LogReg_ytrain)

        score_train = model_LogReg.score(self.LogReg_Xtrain, self.LogReg_ytrain)
        score_test = model_LogReg.score(self.LogReg_Xtest, self.LogReg_ytest)
        self.model_LogReg = model_LogReg
        return score_train, self.get_base_score(), score_test

    def perform_RandomForest(self, n_estimators=10, max_depth=None):
        '''
        random forest predictions with LogReg prepared Data
        '''
        model_RF = RandomForestClassifier(n_estimators=n_estimators,
                                          max_depth=max_depth)
        model_RF.fit(self.LogReg_Xtrain, self.LogReg_ytrain)
        self.model_RF = model_RF

    def perform_GradientBoosting(self, n_estimators=10, learning_rate=0.1):
        '''
        Gradient Boosting predictions with LogReg prepared Data
        '''
        model_GB = GradientBoostingClassifier(n_estimators=n_estimators,
                                          learning_rate=learning_rate)
        model_GB.fit(self.LogReg_Xtrain, self.LogReg_ytrain)
        self.model_GB = model_GB

    def perform_SVM(self,C=1.0, probability=False):
        model_SVM = SVC(C=C, kernel='rbf', probability=probability)
        model_SVM.fit(self.LogReg_Xtrain, self.LogReg_ytrain)
        self.model_SVM = model_SVM


    def get_base_score(self):
        p_s_train = \
            sum( self.train_data['to_phase_II'])*1./len(self.train_data)
        p_s_test = \
            sum( self.test_data['to_phase_II'])*1./len(self.test_data)
        return( p_s_train*p_s_test + (1-p_s_train)*(1-p_s_test) )
