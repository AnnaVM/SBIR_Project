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


    def plot_LogReg(self, color='blue', penalty='l2', C=1.0):
        index = 0

        mean_tpr = 0.0
        mean_fpr = 0.0
        mean_threshold = np.linspace(0., 1., 100)

        for train_index, test_index in self.kf:
            model_test = Model(self.df, train_index, test_index)
            model_test.process_text('Abstract')
            model_test.prepare_LogReg()
            model_test.perform_LogReg(penalty=penalty, C=C)

            prob = model_test.model_LogReg.predict_proba(model_test.LogReg_Xtest)
            labels = model_test.LogReg_ytest

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
        plt.plot(mean_fpr, mean_tpr, lw=2, color=color,
                label='LogReg penalty: %s, C: %s'%(penalty, str(C)))
        plt.legend(loc=4)

    def plot_RF(self, color='blue', n_estimators=10, max_depth=None):
        index = 0

        mean_tpr = 0.0
        mean_fpr = 0.0
        mean_threshold = np.linspace(0., 1., 100)

        for train_index, test_index in self.kf:
            model_test = Model(self.df, train_index, test_index)
            model_test.process_text('Abstract')
            model_test.prepare_LogReg()
            model_test.perform_RandomForest(n_estimators=n_estimators,
                                              max_depth=max_depth)

            prob = model_test.model_RF.predict_proba(model_test.LogReg_Xtest)
            labels = model_test.LogReg_ytest

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
        plt.plot(mean_fpr, mean_tpr, lw=2, color=color,
         label='RandomForest: %s, max depth: %s'%(n_estimators, str(max_depth)))
        plt.legend(loc=4)

    def plot_GB(self, color='blue', n_estimators=10, learning_rate=0.1):
        index = 0

        mean_tpr = 0.0
        mean_fpr = 0.0
        mean_threshold = np.linspace(0., 1., 100)

        for train_index, test_index in self.kf:
            model_test = Model(self.df, train_index, test_index)
            model_test.process_text('Abstract')
            model_test.prepare_LogReg()
            model_test.perform_GradientBoosting(n_estimators=n_estimators,
                                              learning_rate=learning_rate)

            prob = model_test.model_GB.predict_proba(model_test.LogReg_Xtest)
            labels = model_test.LogReg_ytest

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
        plt.plot(mean_fpr, mean_tpr, lw=2, color=color,
         label='GradientBoosting: %s, learning rate: %s'\
                                %(n_estimators, learning_rate))
        plt.legend(loc=4)

    def plot_SVM(self, color='blue', C=1.):
        index = 0

        mean_tpr = 0.0
        mean_fpr = 0.0
        mean_threshold = np.linspace(0., 1., 100)

        for train_index, test_index in self.kf:
            model_test = Model(self.df, train_index, test_index)
            model_test.process_text('Abstract')
            model_test.prepare_LogReg()
            model_test.perform_SVM(C=C, probability=True)

            prob = model_test.model_SVM.predict_proba(model_test.LogReg_Xtest)
            labels = model_test.LogReg_ytest

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
        plt.plot(mean_fpr, mean_tpr, lw=2, color=color,
         label='SVM - C: %s'\
                                %(C))
        plt.legend(loc=4)
