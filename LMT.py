import math
import numpy as np
from sklearn.linear_model import Ridge


class LinearModelTree:
    def __init__(self, min_node_size, node_model_fit_func, min_split_improvement=0):
        self.min_node_size = min_node_size
        self.node_model_fit_func = node_model_fit_func
        self.min_split_improvement = min_split_improvement

    def lm_predictions(self, x, y):
        lm = self.node_model_fit_func(x, y)
        predictions = lm.predict(x)
        if np.isnan(predictions).any():
            print('got nan prediction')
            print(predictions)
            print(x)
            print(lm.miles_model.coef_)
            print(lm.metro_model.coef_)
            1/0
        return predictions, lm

    def convert_df_to_ndarray(self, X):
        if type(X) == 'pandas.core.frame.DataFrame' or type(X) == 'pandas.core.series.Series':
            return X.values
        return X

    def build_tree(self, X, lm_X, y):
        X = self.convert_df_to_ndarray(X)
        y = self.convert_df_to_ndarray(y)
        self.root = Node.build_node_recursive(self, X, lm_X, y)

    def predict(self, X, lm_X):
        X = self.convert_df_to_ndarray(X)
        # y = self.convert_df_to_ndarray(y)
        data = []
        for i in range(X.shape[0]):
            data.append(self.predict_one(X[i, :], lm_X.iloc[[i]]))
        return np.array(data)

    def predict_one(self, X, lm_X):
        X = self.convert_df_to_ndarray(X)
        # y = self.convert_df_to_ndarray(y)
        return self.root.predict_one(X, lm_X)

    # 2-d array of [node_id, prediction]
    def predict_full(self, X, lm_X):
        X = self.convert_df_to_ndarray(X)
        # y = self.convert_df_to_ndarray(y)
        data = []
        for i in range(X.shape[0]):
            data.append(self.predict_full_one(X[i, :], lm_X.iloc[[i]]))
        return np.array(data)

    def predict_full_one(self, X, lm_X):
        X = self.convert_df_to_ndarray(X)
        # y = self.convert_df_to_ndarray(y)
        return self.root.predict_full_one(X, lm_X)

    def node_count(self):
        return self.root.node_count()

    def serialize(self):
        return self.root.serialize()


class Node:
    def __init__(self, feature_idx, pivot_value, lm):
        self.feature_idx = feature_idx
        self.pivot_value = pivot_value
        self.lm = lm
        self.row_count = 0
        self.left = None
        self.right = None

    def node_count(self):
        if self.feature_idx is not None:
            return 1 + self.left.node_count() + self.right.node_count()
        else:
            return 1

    def predict_one(self, x, lm_x):
        local_value = self.lm.predict(lm_x)[0]
        if self.feature_idx is not None:
            child_value = 0
            if x[self.feature_idx] < self.pivot_value:
                child_value = self.left.predict_one(x, lm_x)
            else:
                child_value = self.right.predict_one(x, lm_x)

            return child_value + local_value
        else:
            return local_value

    def predict_full_one(self, x, lm_x, prefix='T'):
        local_value = self.lm.predict(lm_x)[0]
        if self.feature_idx is not None:
            result = None
            if x[self.feature_idx] < self.pivot_value:
                result = self.left.predict_full_one(x, lm_x, prefix + 'L')
            else:
                result = self.right.predict_full_one(x, lm_x, prefix + 'R')
            result[1] += local_value
            return result
        else:
            return np.array([prefix, local_value + self.lm.predict(lm_x)[0]])  # convert to 2-d array, then back

    @staticmethod
    def build_node_recursive(tree, X, lm_X, y):
        (feature_idx, pivot_value, lm, residuals) = Node.find_best_split(tree, X, lm_X, y)
        node = Node(feature_idx, pivot_value, lm)
        node.row_count = X.shape[0]

        if feature_idx is not None:
            left_X, left_lm_X, left_residuals, right_X, right_lm_X, right_residuals = Node.split_on_pivot(
              X, lm_X, residuals, feature_idx, pivot_value)
            node.left = Node.build_node_recursive(tree, left_X, left_lm_X, left_residuals)
            node.right = Node.build_node_recursive(tree, right_X, right_lm_X, right_residuals)

        return node

    @staticmethod
    def split_on_pivot(X, lm_X, y, feature_idx, pivot_value):
        sorting_indices = X[:, feature_idx].argsort()  # sort by column feature_idx
        sorted_X = X[sorting_indices]
        pivot_idx = np.argmax(sorted_X[:, feature_idx] >= pivot_value)
        sorted_lm_X = lm_X.iloc[sorting_indices, :]  # lm_X.[sorting_indices]
        sorted_y = y[sorting_indices]

        return (sorted_X[:pivot_idx, :],
                sorted_lm_X.iloc[:pivot_idx, :],
                sorted_y[:pivot_idx],
                sorted_X[pivot_idx:, :],
                sorted_lm_X.iloc[pivot_idx:, :],
                sorted_y[pivot_idx:])

    @staticmethod
    def find_best_split(tree, X, lm_X, y):
        predictions, lm = tree.lm_predictions(lm_X, y)
        residuals = y - predictions
        mean_value = residuals.mean()
        row_count = X.shape[0]
        sse = (residuals**2).sum()
        resid_sum = residuals.sum()
        best_sse = sse
        best_feature = None
        best_feature_pivot = None

        for feature_idx in range(X.shape[1]):
            sorting_indices = X[:, feature_idx].argsort()  # sort by column feature_idx
            sorted_X = X[sorting_indices]
            sorted_resid = residuals[sorting_indices]
            sum_left = 0
            sum_right = resid_sum
            sum_squared_left = 0
            sum_squared_right = sse
            count_left = 0
            count_right = row_count
            pivot_idx = 0

            while count_right >= tree.min_node_size:
                # advance our pivot
                row_y = sorted_resid[pivot_idx]
                sum_left += row_y
                sum_right -= row_y
                sum_squared_left += row_y*row_y
                sum_squared_right -= row_y*row_y
                count_left += 1
                count_right -= 1
                pivot_idx += 1

                if count_left >= tree.min_node_size and count_right >= tree.min_node_size:
                    # consider a split

                    # compute rmse from sum, sum_squared, and n: rmse = 1/n * sqrt ( n*sum_x2 - (sum_x)^2 )
                    rmse_left = math.sqrt((count_left * sum_squared_left) - (sum_left * sum_left)) / count_left
                    sse_left = rmse_left * rmse_left * count_left
                    rmse_right = math.sqrt((count_right * sum_squared_right) - (sum_right * sum_right)) / count_right
                    sse_right = rmse_right * rmse_right * count_right
                    split_sse = sse_left + sse_right

                    if (split_sse < best_sse and sse - split_sse > tree.min_split_improvement and
                          # only if the value is different than the last value
                          (count_left <= 1 or sorted_X[pivot_idx, feature_idx] != sorted_X[pivot_idx - 1, feature_idx])):
                        best_sse = split_sse
                        best_feature = feature_idx
                        best_feature_pivot = sorted_X[pivot_idx, feature_idx]

        return (best_feature, best_feature_pivot, lm, residuals)

    def serialize(self, prefix='T'):
        if self.feature_idx is not None:
            self_str = ',rc:%i,f:%i,v:%s' % (
                self.row_count, self.feature_idx, str(self.pivot_value))
            return "\n" + prefix + (self_str +
                                    self.left.serialize(prefix + 'L') +
                                    self.right.serialize(prefix + 'R')
                                    )
        else:
            self_str = (',rc:%i,f:_,v:_,int:%f,coef:%s' % (self.row_count, self.lm.intercept_, str(self.lm.coef_)))
            return "\n" + prefix + self_str