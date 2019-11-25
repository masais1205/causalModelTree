from os import listdir
import pandas as pd
from sklearn.model_selection import train_test_split
from weka.classifiers import Classifier
from weka.core import jvm
from Baseline import Baeseline, evaluate_auc
import RFunctions as RF
from DF2Instances import DF2Instances
from TreeCriteria import TreeCriteria
import argparse
import warnings
import inspect

class Node(object):
    '''
    node of a tree
    attr                attribute, string
    thres               threshold to best split data
    parent              parent node of current node
    left                left child
    right               right child
    leaf                is a leaf node
    predict_label       predicted label, if a leaf
    predict_attr        selected predictive attributes
    model               model to predict outcome
    size                testing data size
    AUC                 AUC on validation data
    '''
    def __init__(self):
        self.attr = None
        self.thres = None
        self.parent = None
        self.left = None
        self.right = None
        self.leaf = False
        self.predict_label = None
        self.predict_attr = None
        self.model = None
        self.size = None
        self.AUC = None

    def toString(self):
        print('Attr:', self.attr, '; Thres:', self.thres, '; Leaf:', self.leaf,
              '; noLeft:', self.left == None, '; noRight:', self.right == None,
              '; Predictive attributes: ', self.predict_attr)


def value_in_dict(dict, key):
    '''
    return value of the key, if key does not exist return 0
    :param dict:    dictionary
    :param key:     key
    :return:        value
    '''
    try:
        return dict[key]
    except:
        return 0


def string2float(aStr, attrs):
    coef = []
    for i in range(len(attrs)-1):
        coef.append(float(aStr[aStr.index(attrs[i]) + len(attrs[i]): aStr.index(attrs[i+1])]))
    return coef


# Chooses the attribute and its threshold with the highest info gain
# from the set of attributes
def choose_attr(df_train, df_val, attributes, attr_label, type):
    '''
    choose attribtue to split data
    :param df_train:            training data, pandas data frame
    :param df_val:              validation data, pandas data frame
    :param attributes:          attributes, list of strings
    :param attr_label:          label attribute, string
    :param type:                criteria to build causal model tree
    :return:                    best attribute and threshold
    '''
    max_info_gain = float("-inf")
    max_coeff_diff = float("-inf")
    max_auc_diff = float("-inf")
    best_attr = None
    threshold = 0
    df_train_val = pd.concat([df_train, df_val], ignore_index=True)
    # Test each attribute (note attributes maybe be chosen more than once)

    if type == 'knockout':
        pcs, model, auc = logit_PC(df_train, df_val, attr_label)
        if len(pcs) <= 1:
            return None, None
        for pc in pcs:
            attrs = [x for x in pcs if x != pc]
            attrs.append(attr_label)
            pcs_tmp, model_tmp, auc_tmp = logit_PC(df_train[attrs], df_val[attrs], attr_label)
            if auc - auc_tmp > max_auc_diff:
                max_auc = auc_tmp
                best_attr = pc
                threshold = .5
    else:
        for attr in attributes:
            treeCriteria = TreeCriteria(df_train_val, attr, attr_label)
            p, n = treeCriteria.num_values()
            if p == 0 or n == 0:
                continue
            thres = treeCriteria.select_threshold()

            if type == 'Gini':
                # ig = info_gain(df_train_val, attr, attr_label, thres)
                ig = treeCriteria.Gini_index(thres)
                if ig > max_info_gain:
                    max_info_gain = ig
                    best_attr = attr
                    threshold = thres
            elif type == 'coeff':
                df_train_l = df_train[df_train[attr] < thres]
                df_train_h = df_train[df_train[attr] > thres]
                df_val_l = df_val[df_val[attr] < thres]
                df_val_h = df_val[df_val[attr] > thres]
                pcs_l, model_l, auc_l = logit_PC(df_train_l, df_val_l, attr_label)
                pcs_h, model_h, auc_h = logit_PC(df_train_h, df_val_h, attr_label)
                coef_dict_l, coef_dict_h = {}, {}

                s_l = str(dict(inspect.getmembers(model_l.__delattr__))['__self__'])
                coef_l = string2float(s_l, pcs_l+['Intercept'])
                s_h = str(dict(inspect.getmembers(model_h.__delattr__))['__self__'])
                coef_h = string2float(s_h, pcs_h+['Intercept'])

                if pcs_l:
                    for p, c in zip(pcs_l, coef_l):
                        coef_dict_l[p] = c
                if pcs_h:
                    for p, c in zip(pcs_h, coef_h):
                        coef_dict_h[p] = c
                coeff_diff = 0
                for pc in set(pcs_l).union(set(pcs_h)):
                    coeff_diff += abs(value_in_dict(coef_dict_l, pc) -
                                      value_in_dict(coef_dict_h, pc))
                if coeff_diff > max_coeff_diff:
                    max_coeff_diff = coeff_diff
                    best_attr = attr
                    threshold = thres
    return best_attr, threshold





def logit_PC(df_train, df_test, attr_label):
    '''
    logistic regression with PC members only
    :param df_train:        training data, pandas data frame
    :param df_test:         testing data, pandas data frame
    :param attr_label:      label attribute, string
    :return:                PC members, logistic regression model and AUC
    '''
    pcs = RF.learnPC_R(df_train, attr_label)
    if pcs:
        # model = LogisticRegression().fit(df_train[pcs], df_train[attr_label])
        # pred = model.predict_proba(df_test[pcs])
        # pred = [x[1] for x in pred]
        # auc = evaluate_auc(df_test[attr_label].values.tolist(), pred)

        df2Instances = DF2Instances(df_train[pcs+[attr_label]], 'train', attr_label)
        data_train = df2Instances.df_to_instances()
        data_train.class_is_last()  # set class attribute
        model = Classifier(classname="weka.classifiers.functions.Logistic")
        model.build_classifier(data_train)

        df2Instances = DF2Instances(df_test[pcs+[attr_label]], 'test', attr_label)
        data_test = df2Instances.df_to_instances()
        data_test.class_is_last()  # set class attribute

        preds = []
        for index, inst in enumerate(data_test):
            preds.append(model.distribution_for_instance(inst)[1])
        auc = evaluate_auc(df_test[attr_label].values.tolist(), preds)

        return pcs, model, auc
    else:
        return pcs, None, None



def num_values(df, attr):
    '''
    Returns the number of positive and negative data
    :param df:          data frame
    :param attr:        attribute, string
    :return:            numbers of values
    '''
    p_df = df[df[attr] == 1]
    n_df = df[df[attr] == 0]
    return p_df.shape[0], n_df.shape[0]


def build_tree(now_tree, df_train, df_val, df_test, attributes, attr_label, type, max_level=5, level=0):
    '''
    Builds the Causal Model Tree based on training data, attributes to train on, and a prediction attribute
    :param now_tree:            tree
    :param df_train:            training data
    :param df_val:              validation data
    :param df_test:             testing data
    :param attributes:          attributes
    :param attr_label:          label attribute
    :param type:                criteria
    :param max_level:           max level of the tree
    :param level:               starting level
    '''
    # Get the number of positive and negative examples in the training data
    df_train_val = pd.concat([df_train, df_val], ignore_index=True)
    # treeCriteria = TreeCriteria(df_train_val, attr, attr_label)
    p, n = num_values(df_train_val, attr_label)
    # If train data has all positive or all negative values
    # then we have reached the end of our tree
    if p == 0 or n == 0 or level >= max_level:
        # Create a leaf node indicating it's prediction
        leaf = True
        if p > n:
            predict_label = 1
        else:
            predict_label = 0
        now_tree.attr = None
        now_tree.thres = None
        now_tree.leaf = leaf
        now_tree.predict_label = predict_label
        now_tree.predict_attr, now_tree.model, now_tree.auc = logit_PC(df_train_val, df_test, attr_label)
        now_tree.size = len(df_test)
        if not now_tree.predict_attr:
            now_tree = now_tree.parent
            now_tree.leaf = leaf
            now_tree.left = None
            now_tree.right = None
        # now_tree.toString()
    else:
        # Determine attribute and its threshold value with the highest
        # information gain
        best_attr, threshold = choose_attr(df_train, df_val, attributes, attr_label, type)
        if best_attr:
            # Create internal tree node based on attribute and it's threshold
            now_tree.attr = best_attr
            now_tree.thres = threshold
            now_tree.leaf = False
            df_train_l = df_train[df_train[best_attr] < threshold]
            df_train_h = df_train[df_train[best_attr] >= threshold]
            df_val_l = df_val[df_val[best_attr] < threshold]
            df_val_h = df_val[df_val[best_attr] >= threshold]
            df_test_l = df_test[df_test[best_attr] < threshold]
            df_test_h = df_test[df_test[best_attr] >= threshold]
            now_tree.predict_attr, now_tree.model, now_tree.auc = logit_PC(df_train_val, df_test, attr_label)
            now_tree.size = len(df_test)
            # Recursively build left and right subtree
            if not now_tree.leaf:
                tree = Node()
                tree.parent = now_tree
                now_tree.left = tree
                build_tree(tree, df_train_l, df_val_l, df_test_l, attributes, attr_label, type, max_level, level + 1)

            if not now_tree.leaf:
                tree = Node()
                tree.parent = now_tree
                now_tree.right = tree
                build_tree(tree, df_train_h, df_val_h, df_test_h, attributes, attr_label, type, max_level, level + 1)
        else:
            now_tree.attr = None
            now_tree.thres = None
            now_tree.leaf = True
            now_tree.predict_attr, now_tree.model, now_tree.auc = logit_PC(df_train_val, df_test, attr_label)
            now_tree.size = len(df_test)


# Given a instance of a training data, make a prediction of healthy or colic
# based on the Decision Tree
# Assumes all data has been cleaned (i.e. no NULL data)
def predict(node, row_df):
    # If we are at a leaf node, return the prediction of the leaf node
    if node.leaf:
        return node.predict_label
    # Traverse left or right subtree based on instance's data
    if row_df[node.attr] <= node.thres:
        return predict(node.left, row_df)
    elif row_df[node.attr] > node.thres:
        return predict(node.right, row_df)


# Given a set of data, make a prediction for each instance using the Decision Tree
def test_predictions(root, df):
    num_data = df.shape[0]
    num_correct = 0
    for index, row in df.iterrows():
        prediction = predict(root, row)
        if prediction == row['label']:
            num_correct += 1
    return round(num_correct / num_data, 2)


# Prints the tree level starting at given level
def print_tree(root, df_test, attr_label, level=0):
    # print(counter*" ", end="")
    global avg_auc
    print(level * '|\t', end='')
    if root.leaf:
        # pred = root.model.predict(df_test[root.predict_attr])
        # if len(np.unique(pred)) == 1 or len(
        #         np.unique(df_test[attr_label].values.tolist())) == 1:  # bug in roc_auc_score
        #     auc = accuracy_score(df_test[attr_label].values.tolist(), pred)
        # else:
        #     auc = roc_auc_score(df_test[attr_label].values.tolist(), pred)
        df2Instances = DF2Instances(df_test[root.predict_attr + [attr_label]], 'test', attr_label)
        data_test = df2Instances.df_to_instances()
        data_test.class_is_last()  # set class attribute
        preds = []
        for index, inst in enumerate(data_test):
            preds.append(root.model.distribution_for_instance(inst)[1])
        auc = evaluate_auc(df_test[attr_label].values.tolist(), preds)
        print('leaf', ';', root.predict_attr, ';',
        string2float(str(dict(inspect.getmembers(root.model.__delattr__))['__self__']), root.predict_attr + ['Intercept']),
              ';', root.auc, ';', root.size, ';', auc)
        avg_auc += auc * root.size
    else:
        # pred = root.model.predict(df_test[root.predict_attr])
        # if len(np.unique(pred)) == 1 or len(
        #         np.unique(df_test[attr_label].values.tolist())) == 1:  # bug in roc_auc_score
        #     auc = accuracy_score(df_test[attr_label].values.tolist(), pred)
        # else:
        #     auc = roc_auc_score(df_test[attr_label].values.tolist(), pred)
        df2Instances = DF2Instances(df_test[root.predict_attr + [attr_label]], 'test', attr_label)
        data_test = df2Instances.df_to_instances()
        data_test.class_is_last()  # set class attribute
        preds = []
        for index, inst in enumerate(data_test):
            preds.append(root.model.distribution_for_instance(inst)[1])
        auc = evaluate_auc(df_test[attr_label].values.tolist(), preds)
        print(root.attr, ';', root.predict_attr, ';',
        string2float(str(dict(inspect.getmembers(root.model.__delattr__))['__self__']), root.predict_attr + ['Intercept']),
              ';', root.auc, ';', root.size, ';', auc)
        # root.toString()

    if root.left:
        df_test_l = df_test[df_test[root.attr] < root.thres]
        print_tree(root.left, df_test_l, attr_label, level + 1)
    if root.right:
        df_test_h = df_test[df_test[root.attr] >= root.thres]
        print_tree(root.right, df_test_h, attr_label, level + 1)


def main():
    rpacknames = ['pcalg']
    RF.import_R_library(rpacknames)
    global avg_auc

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', action='store', dest='filename',
                        help='data set file')
    parser.add_argument('-m', action='store', dest='method', default='MT-PC',
                        help='prediction method')
    parser.add_argument('-l', action='store', dest='max_level', default=2,
                        help='the max level of the tree')
    parser.add_argument('-t', action='store', dest='test_size', default=.2,
                        help='the ratio of testing data')
    args, unknown = parser.parse_known_args()
    filename = args.filename

    if filename: # explorer
        method = args.method
        max_level = args.max_level
        test_size = args.test_size

        df = pd.read_csv(filename)
        attributes = df.columns[:-1]
        label = df.columns[-1]
        df_train, df_test = train_test_split(df, test_size=test_size, random_state=1)
        df_train, df_val = train_test_split(df_train, test_size=test_size, random_state=1)

        baseline = Baeseline(df_train, df_test, attributes, label)
        if 'logit' == method:
            print('logit - AUC:', baseline.logit())
        if 'DT' == method:
            print('DT - AUC:', baseline.DT())
        if 'LMT' == method:
            print('LMT - AUC:', baseline.LMT())
        if 'MT-PC' == method:
            for type in ['Gini', 'knockout', 'coeff']:
                avg_auc = 0
                print('\n------------ MT-PC', type, '----------------------')
                root = Node()
                build_tree(root, df_train, df_val, df_test, attributes, label, type, max_level)
                print('attr;\t\tPC;\t\tInSample AUC;\t\tOutOfSample Size;\t\tOutOfSample AUC')
                print_tree(root, df_test, label)
                print(avg_auc, df_test.shape)
                print('MT_PC', type, '- AUC:', avg_auc / len(df_test))

    else: # experiments
        from os.path import isfile, join
        mypath = '../../Documents/Causality/data/synthetic/'
        # filenames = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        filenames = ['test_x_y.csv']

        max_level = 2
        test_size = .2
        methods = ['logit', 'DT', 'LMT', 'MT-PC'] #
        types = ['Gini', 'knockout', 'coeff']  #
        # filename = join(mypath, 'CollegeDistanceData-binary.csv')
        for filename in filenames:
            print('=================================', filename)

            df = pd.read_csv(join(mypath, filename))
            attributes = df.columns[:-1]
            label = df.columns[-1]
            df_train, df_test = train_test_split(df, test_size=test_size, random_state=1)

            baseline = Baeseline(df_train, df_test, attributes, label)
            if 'logit' in methods:
                print('logit - AUC:', baseline.logit())
            if 'DT' in methods:
                print('J48 - AUC:', baseline.DT())
            if 'LMT' in methods:
                print('LMT - AUC:', baseline.LMT())

            df_train, df_val = train_test_split(df_train, test_size=test_size, random_state=1)
            if 'MT-PC' in methods:
                for type in types:
                    avg_auc = 0
                    print('\n------------ MT-PC', type, '----------------------')
                    root = Node()
                    build_tree(root, df_train, df_val, df_test, attributes, label, type, max_level)
                    print('attr;\t\tPC;\t\tInSample AUC;\t\tOutOfSample Size;\t\tOutOfSample AUC')
                    print_tree(root, df_test, label)
                    print('MT_PC', type, '- AUC:', avg_auc/len(df_test))


def initialize():
    global avg_auc
    avg_auc = 0


# if you have global variables, you must
# initialize them outside the main block
initialize()

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=FutureWarning)
    try:
        jvm.start()
        main()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
