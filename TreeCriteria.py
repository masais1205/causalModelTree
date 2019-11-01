import math

class TreeCriteria(object):
    def __init__(self, df, attribute, attr_label):
        self.df = df
        self.attribute = attribute
        self.attr_label = attr_label


    # Returns the number of positive and negative data
    def num_values(self):
        p_df = self.df[self.df[self.attribute] == 1]
        n_df = self.df[self.df[self.attribute] == 0]
        return p_df.shape[0], n_df.shape[0]

    # First select the threshold of the attribute to split set of test data on
    # The threshold chosen splits the test data such that information gain is maximized
    def select_threshold(self):
        # Convert dataframe column to a list and round each value
        values = self.df[self.attribute].tolist()
        values = [float(x) for x in values]
        # Remove duplicate values by converting the list to a set, then sort the set
        values = set(values)
        values = list(values)
        values.sort()
        max_ig = float("-inf")
        thres_val = 0
        # try all threshold values that are half-way between successive values in this sorted list
        for i in range(0, len(values) - 1):
            thres = (values[i] + values[i + 1]) / 2
            ig = self.info_gain(thres)
            if ig > max_ig:
                max_ig = ig
                thres_val = thres
        # Return the threshold value that maximizes information gained
        return thres_val


    # Calculate info content (entropy) of the test data
    def info_entropy(self, df):
        # Dataframe and number of positive/negatives examples in the data
        p_df = df[df[self.attr_label] == 1]
        n_df = df[df[self.attr_label] == 0]
        p = float(p_df.shape[0])
        n = float(n_df.shape[0])
        # Calculate entropy
        if p == 0 or n == 0:
            I = 0
        else:
            I = ((-1 * p) / (p + n)) * math.log(p / (p + n), 2) + ((-1 * n) / (p + n)) * math.log(n / (p + n), 2)
        return I


    # Calculates the weighted average of the entropy after an attribute test
    def remainder(self, df_subsets):
        # number of test data
        num_data = self.df.shape[0]
        remainder = float(0)
        for df_sub in df_subsets:
            if df_sub.shape[0] > 1:
                remainder += float(df_sub.shape[0] / num_data) * self.info_entropy(df_sub)
        return remainder


    # Calculates the information gain from the attribute test based on a given threshold
    # Note: thresholds can change for the same attribute over time
    def info_gain(self, threshold):
        sub_l = self.df[self.df[self.attribute] < threshold]
        sub_h = self.df[self.df[self.attribute] > threshold]
        # Determine information content, and subract remainder of attributes from it
        ig = self.info_entropy(self.df) - self.remainder([sub_l, sub_h])
        return ig


    # Calculates the Gini index from the attribute test based on a given threshold
    # Note: thresholds can change for the same attribute over time
    def Gini_index(self, threshold):
        sub_l = self.df[self.df[self.attribute] < threshold]
        sub_h = self.df[self.df[self.attribute] >= threshold]

        prob_ll = len(sub_l[sub_l[self.attr_label] < threshold]) / len(sub_l)
        prob_lh = len(sub_l[sub_l[self.attr_label] >= threshold]) / len(sub_l)
        prob_hl = len(sub_h[sub_h[self.attr_label] < threshold]) / len(sub_h)
        prob_hh = len(sub_h[sub_h[self.attr_label] >= threshold]) / len(sub_h)

        Gini_l = 1 - (prob_ll ** 2 + prob_lh ** 2)
        Gini_h = 1 - (prob_hl ** 2 + prob_hh ** 2)

        return (len(sub_l) * Gini_l + len(sub_h) * Gini_l) / len(self.df)