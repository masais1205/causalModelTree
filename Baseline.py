from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import pandas as pd
from weka.core.dataset import Instances, Instance, Attribute
from weka.classifiers import Classifier
import weka.core.jvm as jvm
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

def evaluate_auc(y_true, y_pred):
    '''
    evaluate auc
    :param y_true:      list of numeric
    :param y_pred:      list of numeric
    :return:
    '''

    if len(np.unique(y_pred)) == 1 or len(np.unique(y_true)) == 1:  # bug in roc_auc_score
        auc = accuracy_score(y_true, y_pred)
    else:
        auc = roc_auc_score(y_true, y_pred)
    return auc


def df_to_instances(df, relation, attr_label):
    '''
    transform pandas data frame to arff style data
    :param df:              panda data frame
    :param relation:        relation, string
    :param attr_label:      label attribute, string
    :return:                arff style data
    '''

    atts = []
    for col in df.columns:
        if col != attr_label:
            att = Attribute.create_numeric(col)
        else:
            att = Attribute.create_nominal(col, ['0', '1'])
        atts.append(att)
    nrow = len(df)
    result = Instances.create_instances(relation, atts, nrow)
    # data
    for i in range(nrow):
        inst = Instance.create_instance(df.iloc[i].astype('float64').to_numpy().copy(order='C'))
        result.add_instance(inst)

    return result

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


    def logit(self):
        model = LogisticRegression().fit(self.df_train[self.attributes], self.df_train[self.attr_label])
        pred = model.predict(self.df_test[self.attributes])
        auc = evaluate_auc(self.df_test[self.attr_label].values.tolist(), pred)
        return auc


    def DT(self):
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier().fit(self.df_train[self.attributes], self.df_train[self.attr_label])
        pred = model.predict(self.df_test[self.attributes])
        auc = evaluate_auc(self.df_test[self.attr_label].values.tolist(), pred)
        return auc


    def LMT(self):
        jvm.start()
        data_train = df_to_instances(self.df_train, 'train', self.attr_label)
        data_train.class_is_last()  # set class attribute
        model = Classifier(classname="weka.classifiers.trees.LMT")
        model.build_classifier(data_train)
        print(model)

        data_test = df_to_instances(self.df_test, 'test', self.attr_label)
        data_test.class_is_last()  # set class attribute
        preds = []
        for index, inst in enumerate(data_test):
            preds.append(model.classify_instance(inst))
        jvm.stop()
        auc = evaluate_auc(self.df_test[self.attr_label].values.tolist(), preds)
        return auc

        ### self-implemented LMT
        # from lmt import LinearModelTree
        # from sklearn.metrics import mean_squared_error

        # def fit_linear_model(x, y):
        #     lr = LogisticRegression()
        #     lr.fit(x, y)
        #     return lr
        # lmt = LinearModelTree(100, fit_linear_model, min_split_improvement=10)
        # lmt.build_tree(df_train[attributes].values, df_train[attributes], df_train[attr_label].values)
        # pred = lmt.predict(df_test[attributes].values, df_test[attributes])
        # return roc_auc_score(df_test[attr_label].values.tolist(), pred)