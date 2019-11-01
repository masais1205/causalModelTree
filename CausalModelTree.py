import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from Baseline import Baeseline, evaluate_auc
import RFunctions as RF
from TreeCriteria import TreeCriteria
from sklearn.linear_model import LogisticRegression

class Node(object):
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


def value_in_index(alist, a, vlist):
    assert len(alist) == len(vlist), 'different length of alist and vlist'
    try:
        return vlist[alist.index(a)]
    except:
        return 0



# Chooses the attribute and its threshold with the highest info gain
# from the set of attributes
def choose_attr(df_train, df_val, attributes, attr_label, type):
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
                coef_l = model_l.coef_ if pcs_l else [[]]
                pcs_h, model_h, auc_h = logit_PC(df_train_h, df_val_h, attr_label)
                coef_h = model_h.coef_ if pcs_h else [[]]
                coeff_diff = 0
                for pc in set(pcs_l).union(set(pcs_h)):
                    coeff_diff += abs(value_in_index(pcs_l, pc, coef_l[0]) -
                                      value_in_index(pcs_h, pc, coef_h[0]))
                if coeff_diff > max_coeff_diff:
                    max_coeff_diff = coeff_diff
                    best_attr = attr
                    threshold = thres
    return best_attr, threshold





def logit_PC(df_train, df_test, attr_label):
    pcs = RF.learnPC_R(df_train, attr_label)
    if pcs:
        model = LogisticRegression().fit(df_train[pcs], df_train[attr_label])
        pred = model.predict(df_test[pcs])

        auc = evaluate_auc(df_test[attr_label].values.tolist(), pred)
        # fpr, tpr, thresholds = metrics.roc_curve(df_test[attr_label].values.tolist(), pred, pos_label=2)
        # print(metrics.auc(fpr, tpr))
        return pcs, model, auc
    else:
        return pcs, None, None



# Returns the number of positive and negative data
def num_values(df, attr):
    p_df = df[df[attr] == 1]
    n_df = df[df[attr] == 0]
    return p_df.shape[0], n_df.shape[0]

# Builds the Decision Tree based on training data, attributes to train on,
# and a prediction attribute
def build_tree(now_tree, df_train, df_val, df_test, cols, attr_label, type, max_level=5, level=0):
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
        best_attr, threshold = choose_attr(df_train, df_val, cols, attr_label, type)
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
                left = build_tree(tree, df_train_l, df_val_l, df_test_l, cols, attr_label, type, max_level, level + 1)

            if not now_tree.leaf:
                tree = Node()
                tree.parent = now_tree
                now_tree.right = tree
                right = build_tree(tree, df_train_h, df_val_h, df_test_h, cols, attr_label, type, max_level, level + 1)
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
        pred = root.model.predict(df_test[root.predict_attr])
        if len(np.unique(pred)) == 1 or len(
                np.unique(df_test[attr_label].values.tolist())) == 1:  # bug in roc_auc_score
            auc = accuracy_score(df_test[attr_label].values.tolist(), pred)
        else:
            auc = roc_auc_score(df_test[attr_label].values.tolist(), pred)
        print('leaf', ';', root.predict_attr, ';', root.auc, ';', root.size, ';', auc)
        avg_auc += auc * root.size
    else:
        pred = root.model.predict(df_test[root.predict_attr])
        if len(np.unique(pred)) == 1 or len(
                np.unique(df_test[attr_label].values.tolist())) == 1:  # bug in roc_auc_score
            auc = accuracy_score(df_test[attr_label].values.tolist(), pred)
        else:
            auc = roc_auc_score(df_test[attr_label].values.tolist(), pred)
        print(root.attr, ';', root.predict_attr, ';', root.auc, ';', root.size, ';', auc)
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

    from os.path import isfile, join
    mypath = 'C:/Users/admin-mas/Documents/Causality/data/binary_data/'
    # filenames = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    filenames = ['CollegeDistanceData-binary.csv']

    max_level = 2
    test_size = .2
    methods = ['logit', 'DT', 'LMT', 'MT-PC']
    types = ['Gini', 'knockout', 'coeff']  #
    # filename = join(mypath, 'CollegeDistanceData-binary.csv')
    for filename in filenames:
        print('=================================', filename)

        df = pd.read_csv(join(mypath, filename))
        attributes = df.columns[:-1]
        label = df.columns[-1]
        df_train, df_test = train_test_split(df, test_size=test_size, random_state=1)

        # baseline = Baeseline(df_train, df_test, attributes, label)
        # if 'logit' in methods:
        #     print('logit - AUC:', baseline.logit())
        # if 'DT' in methods:
        #     print('DT - AUC:', baseline.DT())
        # if 'LMT' in methods:
        #     print('LMT - AUC:', baseline.LMT())

        df_train, df_val = train_test_split(df_train, test_size=test_size, random_state=1)
        if 'MT-PC' in methods:
            for type in types:
                global avg_auc
                avg_auc = 0
                print('\n------------ MT-PC', type, '----------------------')
                root = Node()
                build_tree(root, df_train, df_val, df_test, attributes, label, type, max_level)
                print('attr;\t\tPC;\t\tInSample AUC;\t\tOutOfSample Size;\t\tOutOfSample AUC')
                print_tree(root, df_test, label)
                print(avg_auc, df_test.shape)
                print('MT_PC', type, '- AUC:', avg_auc/len(df_test))


def initialize():
    global avg_auc
    avg_auc = 0


# if you have global variables, you must
# initialize them outside the main block
initialize()

if __name__ == '__main__':
    main()
