from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
from DF2Instances import DF2Instances
from weka.classifiers import Classifier

def evaluate_auc(y_true, y_pred):
    '''
    evaluate auc
    :param y_true:      list of numeric
    :param y_pred:      list of numeric
    :return:
    '''

    # y_pred = [1-y if x==1 else y for x,y in zip(y_true,y_pred)]
    if len(y_true) == 1 or len(y_true) == 0:
        return 1
    if len(np.unique(y_pred)) == 1:  # bug in roc_auc_score
        y_pred[0] = 1 if y_pred[0]==0 else 0
    if len(np.unique(y_true)) == 1:
        y_true[0] = 1 if y_true[0]==0 else 0
    auc = roc_auc_score(y_true, y_pred)
    return auc

class Baeseline(object):
    '''
    Baseline approaches, including logistic regression, decision tree and logistic model tree
    :param df_train        train data, panda data frame
    :param df_test         test data, panda data frame
    :param attributes      predictive attributes, list of strings
    :param attr_label      label attribute, string
    '''

    def __init__(self, df_train, df_test, attributes, attr_label):
        self.df_train = df_train
        self.df_test = df_test
        self.attributes = attributes
        self.attr_label = attr_label

        df2Instances = DF2Instances(df_train, 'train', attr_label)
        self.data_train = df2Instances.df_to_instances()
        self.data_train.class_is_last()  # set class attribute
        df2Instances = DF2Instances(df_test, 'test', attr_label)
        self.data_test = df2Instances.df_to_instances()
        self.data_test.class_is_last()  # set class attribute


    def logit(self):
        model = Classifier(classname="weka.classifiers.functions.Logistic")
        model.build_classifier(self.data_train)
        preds = []
        for index, inst in enumerate(self.data_test):
            preds.append(model.distribution_for_instance(inst)[1])
        auc = evaluate_auc(self.df_test[self.attr_label].values.tolist(), preds)

        ### scikit learn logit
        # from sklearn.linear_model import LogisticRegression
        # model = LogisticRegression().fit(self.df_train[self.attributes], self.df_train[self.attr_label])
        # pred = model.predict_proba(self.df_test[self.attributes])
        # pred = [x[1] for x in pred]
        # auc = evaluate_auc(self.df_test[self.attr_label].values.tolist(), pred)
        return auc


    def DT(self):
        model = Classifier(classname="weka.classifiers.trees.J48")
        model.build_classifier(self.data_train)
        preds = []
        for index, inst in enumerate(self.data_test):
            preds.append(model.distribution_for_instance(inst)[1])
        auc = evaluate_auc(self.df_test[self.attr_label].values.tolist(), preds)

        ### scikit learn decision tree
        # from sklearn.tree import DecisionTreeClassifier
        # model = DecisionTreeClassifier().fit(self.df_train[self.attributes], self.df_train[self.attr_label])
        # pred = model.predict_proba(self.df_test[self.attributes])
        # pred = [x[1] for x in pred]
        # auc = evaluate_auc(self.df_test[self.attr_label].values.tolist(), pred)
        return auc


    def LMT(self):
        model = Classifier(classname="weka.classifiers.trees.LMT")
        model.build_classifier(self.data_train)
        print(model)

        preds = []
        for index, inst in enumerate(self.data_test):
            preds.append(model.distribution_for_instance(inst)[1])
        auc = evaluate_auc(self.df_test[self.attr_label].values.tolist(), preds)
        return auc

        ### self-implemented LMT
        # from LMT import LinearModelTree
        # from sklearn.metrics import mean_squared_error
        #
        # def fit_linear_model(x, y):
        #     lr = LogisticRegression()
        #     lr.fit(x, y)
        #     return lr
        # lmt = LinearModelTree(100, fit_linear_model, min_split_improvement=10)
        # lmt.build_tree(self.df_train[self.attributes].values, self.df_train[self.attributes], self.df_train[self.attr_label].values)
        # pred = lmt.predict(self.df_test[self.attributes].values, self.df_test[self.attributes])
        # return evaluate_auc(self.df_test[self.attr_label].values.tolist(), pred)