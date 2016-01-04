import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold
from sklearn import metrics

from model import Model

class ROC(object):

    def __init__(self, df):
        self.df = df
        #defining a training and testing set (through indices)
        self.kf = KFold(len(df), 5, shuffle=True)


    def plot_LogReg(self, color='blue', smote_option=False,
                          penalty='l2', C=1.0):
        self.plot_helper('LogReg', color, smote_option, penalty=penalty, C=C)

    def plot_RF(self, color='blue', smote_option=False,
                      n_estimators=10, max_depth=None):
        self.plot_helper('RandomForest', color, smote_option,
                            n_estimators=n_estimators, max_depth=max_depth)

    def plot_GB(self, color='blue', smote_option=False,
                      n_estimators=10, learning_rate=0.1):
        self.plot_helper('GradientBoosting', color, smote_option,
                            n_estimators=n_estimators,
                            learning_rate=learning_rate)

    def plot_SVM(self, color='blue', smote_option=False, C=1.):
        self.plot_helper('SVM', color, smote_option, C=C, probability=True)

    def plot_helper(self, model_name, color, smote_option, **kw ):
        index = 0

        mean_tpr = 0.0
        mean_fpr = 0.0
        mean_threshold = np.linspace(0., 1., 100)

        for train_index, test_index in self.kf:
            model_test = Model(self.df, train_index, test_index)
            model_test.process_text('Abstract')
            model_test.prepare_data(smote_option)
            if model_name=='LogReg':
                model_test.perform_LogReg(**kw)
                prob = model_test.model_LogReg.predict_proba(
                                model_test.test_features)

            elif model_name=='RandomForest':
                model_test.perform_RandomForest(**kw)
                prob = model_test.model_RF.predict_proba(
                                model_test.test_features)

            elif model_name=='GradientBoosting':
                model_test.perform_GradientBoosting(**kw)
                prob = model_test.model_GB.predict_proba(
                                model_test.test_features)

            elif model_name=='SVM':
                model_test.perform_SVM(**kw)
                prob = model_test.model_SVM.predict_proba(
                                model_test.test_features)
            labels = model_test.test_labels

            fpr, tpr, thresholds = metrics.roc_curve(labels, prob[:,1])

            mean_tpr += interp(mean_threshold, thresholds[::-1], tpr[::-1])
            mean_fpr += interp(mean_threshold, thresholds[::-1], fpr[::-1])
            #mean_tpr[0] = 0.0
            #mean_fpr[0] = 0.0

            plt.plot(fpr, tpr,
                     color=color, alpha=0.2)
            plt.plot([0,1], [0,1], ls='dashed', c='k')
            plt.xlabel('FPR')
            plt.ylabel('TPR')

            index+=1


        mean_tpr = mean_tpr *1./5
        mean_fpr = mean_fpr *1./5
        self.tpr = mean_tpr
        self.fpr = mean_fpr
        self.thresholds = mean_threshold
        label_options = [ '%s:%s' %(key, value) \
                            for key, value in kw.iteritems() ]
        label = model_name + ' ' + ' '.join(label_options)
        plt.plot(mean_fpr, mean_tpr, lw=2, color=color,
                label=label )
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
