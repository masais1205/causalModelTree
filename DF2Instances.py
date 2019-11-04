from weka.core.dataset import Instances, Instance, Attribute


class DF2Instances(object):
    def __init__(self, df, relation, attr_label):
        self.df = df
        self.relation = relation
        self.attr_label = attr_label

    def df_to_instances(self):
        '''
        transform pandas data frame to arff style data
        :param df:              panda data frame
        :param relation:        relation, string
        :param attr_label:      label attribute, string
        :return:                arff style data
        '''

        atts = []
        for col in self.df.columns:
            if col != self.attr_label:
                att = Attribute.create_numeric(col)
            else:
                att = Attribute.create_nominal(col, ['0', '1'])
            atts.append(att)
        nrow = len(self.df)
        result = Instances.create_instances(self.relation, atts, nrow)
        # data
        for i in range(nrow):
            inst = Instance.create_instance(self.df.iloc[i].astype('float64').to_numpy().copy(order='C'))
            result.add_instance(inst)

        return result