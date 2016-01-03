import pandas as pd
import numpy as np

###for models
#transforming text to vectors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

#scaling features
from sklearn.preprocessing import StandardScaler

#testing models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

#importing utilities functions
import sys
sys.path.append('../my_app')
from utilities import set_label_topics, feature_engineering,\
                           predictions, print_results




##############################
###     CLASS
##############################

class Model():
    '''
    this class is meant to:
    i) prepare data
    ii) train models (LogReg, RF, GB, SVM...)
    iii) return predictions

    attributes:
    -----------
    from init:
     - train index (self.train_index)
     - text index (self.test_index)
     - training dataset (self.train_data)
     - testing dataset (self.test_data)
     - list of columns in model (add description)

     from process_text
     - tfidf vectorizer trained on training dataset (self.tfidf_vectorizer)
     - the vocabulary from training (self.vocab)
     - the NMF (self.model_NMF)
     - updated: training and testing datasets

     from prepare_data
     - scaler (self.scaler)
     - to train models: self.train_features, self.train_labels
     - to test models: self.test_features, self.test_labels

     from perform models:
     - perform_LogReg (self.model_LogReg)
     - perform_RandomForest (self.model_RF)
     - perform_GradientBoosting (self.model_GB)
     - perform_SVM (self.model_SVM)

    '''

    def __init__(self, data_frame, train_index, test_index):
        '''
        parameters:
        -----------
        df is a pandas DataFrame:
            - Abstract
            - 'Hubzone Owned' (as 'Y'/'N'),
            -'Socially and Economically Disadvantaged' (as 'Y'/'N'),
            - 'Woman Owned' (as 'Y'/'N')
            - 'Solicitation Year',
            - 'Award Amount'
            - '# Employees'
        train_index: list of indices of datapoints of df to use as training set
        test_index: list of indices for the test set (should not overlap)

        ex:
        df = subset_data('dod', 2012, '/Users/AnnaVMS/Desktop/test2')
        kf = KFold(len(df),5, shuffle=True)
        kf_iterator = kf.__iter__()
        train_index, test_index = kf_iterator.next()
        model_test = Model(df, train_index, test_index)

        * with column names: Index([u'index', u'Company', u'Award Title', u'Agency', u'Branch', u'Phase',
u'Program', u'Agency Tracking #', u'Contract', u'Award Start Date',
u'Award Close Date', u'Solicitation #', u'Solicitation Year',
u'Topic Code', u'Award Year', u'Award Amount', u'DUNS',
u'Hubzone Owned', u'Socially and Economically Disadvantaged',
u'Woman Owned', u'# Employees', u'Company Website', u'Address1',
u'Address2', u'City', u'State', u'Zip', u'Contact Name',
u'Contact Title', u'Contact Phone', u'Contact Email', u'PI Name',
u'PI Title', u'PI Phone', u'PI Email', u'RI Name', u'RI POC Name',
u'RI POC Phone', u'Research Keywords', u'Abstract', u'to_phase_II'],
dtype='object')
        '''
        self.train_index = train_index
        self.test_index = test_index
        self.list_columns = ['Solicitation Year',
                             'Award Amount',
                             'Hubzone Owned as_int',
                             'Socially and Economically Disadvantaged as_int',
                             'Woman Owned as_int',
                             '# Employees',
                             'Abstract Length',
                             'to_phase_II']
        self.train_data = data_frame.ix[train_index]
        self.test_data = data_frame.ix[test_index]

    def process_text(self,col_name='Abstract'):
        '''
        meant to call several functions to add a 'topic' column to the
        dataframe. The text to vectorize is in column 'col_name'
        Future: more functionalities in text vectorization and NMF
        '''
        #get the tfidf_vectorizer and NMF trained
        self.perform_vectorize_text(col_name=col_name)
        self.perform_NMF(n_components=5, col_name=col_name)

        #get the labels
        set_label_topics(self.train_data, col_name=col_name,
                            tfidf_vectorizer=self.tfidf_vectorizer,
                            model_NMF=self.model_NMF)
        set_label_topics(self.test_data, col_name=col_name,
                            tfidf_vectorizer=self.tfidf_vectorizer,
                            model_NMF=self.model_NMF)

##########functions for text processing
    def perform_vectorize_text(self, col_name='Abstract', max_features=5000):
        '''
        --> trained on the training dataset, for the column in "col_name"
        class attributes created:
        -------------------------
        tfidf_vectorizer
        vocab
        '''
        #instantiate vectorizer, basic for now
        tfidf_vectorizer = TfidfVectorizer(stop_words='english',
                                            max_features=max_features)

        #fit the model on the train set
        X_train = self.train_data[col_name]
        tfidf_vectorizer.fit(X_train)

        #keep model tfidf_vectorizer (and vocabulary)
        self.tfidf_vectorizer = tfidf_vectorizer
        self.vocab = tfidf_vectorizer.get_feature_names()

    def perform_NMF(self, n_components=5, col_name='Abstract'):
        '''
        is meant to get the top topics for the vectorized training text

        class attributes created:
        -------------------------
        model_NMF

        requirements:
        -------------
        uses the tfidf_vectorizer already trained (class attributes)
        '''
        #instantiate NMF model
        model_NMF = NMF(n_components=n_components,
                        init='random', random_state=0)

        #fit the model on the train set
        X_train = self.train_data[col_name]
        X_train_vectorized = self.tfidf_vectorizer.transform(X_train)
        model_NMF.fit(X_train_vectorized)

        self.model_NMF = model_NMF

    def get_top_n_words_for_topic(self, i_topic, n=10):
        '''
        parameters:
        -----------
        i_topic: index of the topic
        n: number of words we want
        needs to have the vocabulary from the tfid vectorizer, and NMF

        return:
        -------
        the top n words for given topic as an array
        '''
        topic_weight = self.model_NMF.components_[i_topic,:]
        index_topic = np.argsort(topic_weight)[::-1][:n]
        return np.array(self.vocab)[index_topic]


##########

    def prepare_data(self):
        '''
        meant to call feature_engineering on training and testing datasets,
        followed by scaling
        Class Attributes:
        -----------------
        self.train_features
        self.train_labels
        self.test_features
        self.test_labels

        self.scaler
        '''
        feature_engineering(self.train_data, self.model_NMF, training=True,
                            list_columns=self.list_columns)
        self.perform_scaling()
        train_df = self.train_data[self.list_columns]

        self.train_features = \
                self.scaler.transform(train_df[self.list_columns[:-1]].values)
        self.train_labels = train_df[self.list_columns[-1]].values

        feature_engineering(self.test_data, self.model_NMF)
        test_df = self.test_data[self.list_columns]

        self.test_features = \
                self.scaler.transform(test_df[self.list_columns[:-1]].values)
        self.test_labels = test_df[self.list_columns[-1]].values

##########functions for data preparation

    def perform_scaling(self):
        '''
        is meant to train scaler to scale the features

        class attributes created:
        -------------------------
        scaler

        requirements:
        -------------
        uses the feature_engineering()
        '''
        scaler = StandardScaler()

        #get the features to train on:
        train_df = self.train_data[self.list_columns]

        #fit scaler:
        scaler.fit(train_df[self.list_columns[:-1]].values)
        self.scaler = scaler

##########
    def perform_LogReg(self, penalty='l2', C=1.0):
        '''
        Class Attributes:
        -----------------
        model_LogReg
        '''
        model_LogReg = LogisticRegression(penalty=penalty, C=C)
        model_LogReg.fit(self.train_features, self.train_labels)

        self.model_LogReg = model_LogReg

    def perform_RandomForest(self, n_estimators=10, max_depth=None):
        '''
        Class Attributes:
        -----------------
        model_RF
        '''
        model_RF = RandomForestClassifier(n_estimators=n_estimators,
                                          max_depth=max_depth)
        model_RF.fit(self.train_features, self.train_labels)
        self.model_RF = model_RF

    def perform_GradientBoosting(self, n_estimators=10, learning_rate=0.1):
        '''
        Class Attributes:
        -----------------
        model_GB
        '''
        model_GB = GradientBoostingClassifier(n_estimators=n_estimators,
                                          learning_rate=learning_rate)
        model_GB.fit(self.train_features, self.train_labels)
        self.model_GB = model_GB

    def perform_SVM(self,C=1.0, probability=False):
        '''
        Class Attributes:
        -----------------
        model_SVM
        '''
        model_SVM = SVC(C=C, kernel='rbf', probability=probability)
        model_SVM.fit(self.train_features, self.train_labels)
        self.model_SVM = model_SVM

########## assess a model:

    def model_assessement(self, model_to_assess):
        '''
        requirements:
        -------------
        model is trained already
        test data is processed
        '''

        predicted_labels = model_to_assess.predict(self.test_features)
        labels = self.test_labels
        tp, tn, fp, fn = predictions(labels, predicted_labels)
        print_results(tp, tn, fp, fn)



    def get_base_score(self):
        p_s_train = \
            sum( self.train_data['to_phase_II'])*1./len(self.train_data)
        p_s_test = \
            sum( self.test_data['to_phase_II'])*1./len(self.test_data)
        return( p_s_train*p_s_test + (1-p_s_train)*(1-p_s_test) )
