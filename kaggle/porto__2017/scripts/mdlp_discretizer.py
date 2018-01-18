from math import log
import random
import sys
import getopt
import re
import pandas as pd
import numpy as np

def entropy(data_classes, base=2):
    """
    Computes the entropy of a set of labels (class instantiations)
    :param base: logarithm base for computation
    :param data_classes: Series with labels of examples in a dataset
    :return: value of entropy
    """

    if not isinstance(data_classes, pd.core.series.Series):
        raise AttributeError('input array should be a pandas series')
    classes = data_classes.unique()
    N = len(data_classes)
    ent = 0  # initialize entropy

    # iterate over classes
    for c in classes:
        partition = data_classes[data_classes == c]  # data with class = c
        proportion = len(partition) / N
        #update entropy
        ent -= proportion * log(proportion, base)

    return ent

def cut_point_information_gain(dataset, cut_point, feature_label, class_label):
    """
    Return de information gain obtained by splitting a numeric attribute in two according to cut_point
    :param dataset: pandas dataframe with a column for attribute values and a column for class
    :param cut_point: threshold at which to partition the numeric attribute
    :param feature_label: column label of the numeric attribute values in data
    :param class_label: column label of the array of instance classes
    :return: information gain of partition obtained by threshold cut_point
    """
    if not isinstance(dataset, pd.core.frame.DataFrame):
        raise AttributeError('input dataset should be a pandas data frame')

    entropy_full = entropy(dataset[class_label])  # compute entropy of full dataset (w/o split)

    #split data at cut_point
    data_left = dataset[dataset[feature_label] <= cut_point]
    data_right = dataset[dataset[feature_label] > cut_point]
    (N, N_left, N_right) = (len(dataset), len(data_left), len(data_right))

    gain = entropy_full - (N_left / N) * entropy(data_left[class_label]) - \
        (N_right / N) * entropy(data_right[class_label])

    return gain



class MDLP_Discretizer(object):
    def __init__(self, dataset, testset, class_label, out_path_data, out_test_path_data, out_path_bins, features=None):
        """
        initializes discretizer object:
            saves raw copy of data and creates self._data with only features to discretize and class
            computes initial entropy (before any splitting)
            self._features = features to be discretized
            self._classes = unique classes in raw_data
            self._class_name = label of class in pandas dataframe
            self._data = partition of data with only features of interest and class
            self._cuts = dictionary with cut points for each feature
        :param dataset: pandas dataframe with data to discretize
        :param class_label: name of the column containing class in input dataframe
        :param features: if !None, features that the user wants to discretize specifically
        :return:
        """

        if not isinstance(dataset, pd.core.frame.DataFrame):  # class needs a pandas dataframe
            raise AttributeError('Input dataset should be a pandas data frame')

        if not isinstance(testset, pd.core.frame.DataFrame):  # class needs a pandas dataframe
            raise AttributeError('Test dataset should be a pandas data frame')

        self._data_raw = dataset #copy or original input data
        self._test_raw = testset #copy or original test data

        self._class_name = class_label

        self._classes = self._data_raw[self._class_name].unique()

        #if user specifies which attributes to discretize
        if features:
            self._features = [f for f in features if f in self._data_raw.columns]  # check if features in dataframe
            missing = set(features) - set(self._features)  # specified columns not in dataframe
            if missing:
                print ('WARNING: user-specified features %s not in input dataframe' % str(missing))
        else:  # then we need to recognize which features are numeric
            numeric_cols = self._data_raw._data.get_numeric_data().items
            self._features = [f for f in numeric_cols if f != class_label]
        #other features that won't be discretized
        self._ignored_features = set(self._data_raw.columns) - set(self._features)
        self._ignored_features_t = set(self._test_raw.columns) - set(self._features)

        #create copy of data only including features to discretize and class
        self._data = self._data_raw.loc[:, self._features + [class_label]]
        self._test = self._test_raw.loc[:, self._features]
        #pre-compute all boundary points in dataset
        self._boundaries = self.compute_boundary_points_all_features()
        #initialize feature bins with empty arrays
        self._cuts = {f: [] for f in self._features}
        #get cuts for all features
        self.all_features_accepted_cutpoints()
        #discretize self._data
        self.apply_cutpoints(out_data_path=out_path_data, out_test_path=out_test_path_data, out_bins_path=out_path_bins)

    def MDLPC_criterion(self, data, feature, cut_point):
        """
        Determines whether a partition is accepted according to the MDLPC criterion
        :param feature: feature of interest
        :param cut_point: proposed cut_point
        :param partition_index: index of the sample (dataframe partition) in the interval of interest
        :return: True/False, whether to accept the partition
        """
        #get dataframe only with desired attribute and class columns, and split by cut_point
        data_partition = data.copy(deep=True)
        data_left = data_partition[data_partition[feature] <= cut_point]
        data_right = data_partition[data_partition[feature] > cut_point]

        #compute information gain obtained when splitting data at cut_point
        cut_point_gain = cut_point_information_gain(dataset=data_partition, cut_point=cut_point,
                                                    feature_label=feature, class_label=self._class_name)
        #compute delta term in MDLPC criterion
        N = len(data_partition) # number of examples in current partition
        partition_entropy = entropy(data_partition[self._class_name])
        k = len(data_partition[self._class_name].unique())
        k_left = len(data_left[self._class_name].unique())
        k_right = len(data_right[self._class_name].unique())
        entropy_left = entropy(data_left[self._class_name])  # entropy of partition
        entropy_right = entropy(data_right[self._class_name])
        delta = log(3 ** k, 2) - (k * partition_entropy) + (k_left * entropy_left) + (k_right * entropy_right)

        #to split or not to split
        gain_threshold = (log(N - 1, 2) + delta) / N

        if cut_point_gain > gain_threshold:
            return True
        else:
            return False

    def feature_boundary_points(self, data, feature):
        """
        Given an attribute, find all potential cut_points (boundary points)
        :param feature: feature of interest
        :param partition_index: indices of rows for which feature value falls whithin interval of interest
        :return: array with potential cut_points
        """
        #get dataframe with only rows of interest, and feature and class columns
        data_partition = data.copy(deep=True)
        data_partition.sort_values(feature, ascending=True, inplace=True)

        boundary_points = []

        #add temporary columns
        data_partition['class_offset'] = data_partition[self._class_name].shift(1)  # column where first value is now second, and so forth
        data_partition['feature_offset'] = data_partition[feature].shift(1)  # column where first value is now second, and so forth
        data_partition['feature_change'] = (data_partition[feature] != data_partition['feature_offset'])
        data_partition['mid_points'] = data_partition.loc[:, [feature, 'feature_offset']].mean(axis=1)

        potential_cuts = data_partition[data_partition['feature_change'] == True].index[1:]
        sorted_index = data_partition.index.tolist()

        for row in potential_cuts:
            old_value = data_partition.loc[sorted_index[sorted_index.index(row) - 1]][feature]
            new_value = data_partition.loc[row][feature]
            old_classes = data_partition[data_partition[feature] == old_value][self._class_name].unique()
            new_classes = data_partition[data_partition[feature] == new_value][self._class_name].unique()
            if len(set.union(set(old_classes), set(new_classes))) > 1:
                boundary_points += [data_partition.loc[row]['mid_points']]

        return set(boundary_points)

    def compute_boundary_points_all_features(self):
        """
        Computes all possible boundary points for each attribute in self._features (features to discretize)
        :return:
        """
        boundaries = {}
        for attr in self._features:
            data_partition = self._data.loc[:, [attr, self._class_name]]
            boundaries[attr] = self.feature_boundary_points(data=data_partition, feature=attr)
        return boundaries

    def boundaries_in_partition(self, data, feature):
        """
        From the collection of all cut points for all features, find cut points that fall within a feature-partition's
        attribute-values' range
        :param data: data partition (pandas dataframe)
        :param feature: attribute of interest
        :return: points within feature's range
        """
        range_min, range_max = (data[feature].min(), data[feature].max())
        return set([x for x in self._boundaries[feature] if (x > range_min) and (x < range_max)])

    def best_cut_point(self, data, feature):
        """
        Selects the best cut point for a feature in a data partition based on information gain
        :param data: data partition (pandas dataframe)
        :param feature: target attribute
        :return: value of cut point with highest information gain (if many, picks first). None if no candidates
        """
        candidates = self.boundaries_in_partition(data=data, feature=feature)
        # candidates = self.feature_boundary_points(data=data, feature=feature)
        if not candidates:
            return None
        gains = [(cut, cut_point_information_gain(dataset=data, cut_point=cut, feature_label=feature,
                                                  class_label=self._class_name)) for cut in candidates]
        gains = sorted(gains, key=lambda x: x[1], reverse=True)

        return gains[0][0] #return cut point

    def single_feature_accepted_cutpoints(self, feature, partition_index=pd.DataFrame().index):
        """
        Computes the cuts for binning a feature according to the MDLP criterion
        :param feature: attribute of interest
        :param partition_index: index of examples in data partition for which cuts are required
        :return: list of cuts for binning feature in partition covered by partition_index
        """
        if partition_index.size == 0:
            partition_index = self._data.index  # if not specified, full sample to be considered for partition

        data_partition = self._data.loc[partition_index, [feature, self._class_name]]

        #exclude missing data:
        if data_partition[feature].isnull().values.any:
            data_partition = data_partition[~data_partition[feature].isnull()]

        #stop if constant or null feature values
        if len(data_partition[feature].unique()) < 2:
            return
        #determine whether to cut and where
        cut_candidate = self.best_cut_point(data=data_partition, feature=feature)
        if cut_candidate == None:
            return
        decision = self.MDLPC_criterion(data=data_partition, feature=feature, cut_point=cut_candidate)

        #apply decision
        if not decision:
            return  # if partition wasn't accepted, there's nothing else to do
        if decision:
            # try:
            #now we have two new partitions that need to be examined
            left_partition = data_partition[data_partition[feature] <= cut_candidate]
            right_partition = data_partition[data_partition[feature] > cut_candidate]
            if left_partition.empty or right_partition.empty:
                return #extreme point selected, don't partition
            self._cuts[feature] += [cut_candidate]  # accept partition
            self.single_feature_accepted_cutpoints(feature=feature, partition_index=left_partition.index)
            self.single_feature_accepted_cutpoints(feature=feature, partition_index=right_partition.index)
            #order cutpoints in ascending order
            self._cuts[feature] = sorted(self._cuts[feature])
            return

    def all_features_accepted_cutpoints(self):
        """
        Computes cut points for all numeric features (the ones in self._features)
        :return:
        """
        for attr in self._features:
            self.single_feature_accepted_cutpoints(feature=attr)
        return

    def apply_cutpoints(self, out_data_path=None, out_test_path=None, out_bins_path=None):
        """
        Discretizes data by applying bins according to self._cuts. Saves a new, discretized file, and a description of
        the bins
        :param out_data_path: path to save discretized data
        :param out_test_path: path to save discretized test data
        :param out_bins_path: path to save bins description
        :return:
        """
        pbin_label_collection = {}
        bin_label_collection = {}
        for attr in self._features:
            if len(self._cuts[attr]) == 0:
#                self._data[attr] = 'All'
                self._data[attr] = self._data[attr].values
                self._test[attr] = self._test[attr].values
                pbin_label_collection[attr] = ['No binning']
                bin_label_collection[attr] = ['All']
            else:
                cuts = [-np.inf] + self._cuts[attr] + [np.inf]
                print(attr, cuts)
                start_bin_indices = range(0, len(cuts) - 1)
                pbin_labels = ['%s_to_%s' % (str(cuts[i]), str(cuts[i+1])) for i in start_bin_indices]
                bin_labels = ['%d' % (i+1) for i in start_bin_indices]
                pbin_label_collection[attr] = pbin_labels
                bin_label_collection[attr] = bin_labels
                self._data[attr] = pd.cut(x=self._data[attr].values, bins=cuts, right=False, labels=bin_labels,
                                          precision=6, include_lowest=True)
                self._test[attr] = pd.cut(x=self._test[attr].values, bins=cuts, right=False, labels=bin_labels,
                                          precision=6, include_lowest=True)

        #reconstitute full data, now discretized
        if self._ignored_features:
        #the line below may help in removing double class column ; looks like it works
            self._data = self._data.loc[:, self._features]
            to_return_train = pd.concat([self._data, self._data_raw[list(self._ignored_features)]], axis=1)
            to_return_train = to_return_train[self._data_raw.columns] #sort columns so they have the original order
        else:
        #the line below may help in removing double class column ; looks like it works
            self._data = self._data.loc[:, self._features]
            to_return_train = self._data

        #save data as csv
        if out_data_path:
            to_return_train.to_csv(out_data_path, index=False)

        #reconstitute test data, now discretized
        if self._ignored_features:
        #the line below may help in removing double class column ; looks like it works
        #    self._test = self._test.loc[:, self._features]
            to_return_test = pd.concat([self._test, self._test_raw[list(self._ignored_features_t)]], axis=1)
            to_return_test = to_return_test[self._test_raw.columns] #sort columns so they have the original order
        else:
        #the line below may help in removing double class column ; looks like it works
        #    self._data = self._data.loc[:, self._features]
            to_return_test = self._test

        #save data as csv
        if out_test_path:
            to_return_test.to_csv(out_test_path, index=False)

        #save bins description
        if out_bins_path:
            with open(out_bins_path, 'w') as bins_file:
                print>>bins_file, 'Description of bins in file: %s' % out_data_path
                for attr in self._features:
                    print>>bins_file, 'attr: %s\n\t%s' % (attr, ', '.join([pbin_label for pbin_label in pbin_label_collection[attr]]))


def load_discretized():
    mdlp_train = pd.read_csv('./train-mdlp.csv')
    mdlp_train = mdlp_train.replace(np.nan, -1)
    train_mdlp = mdlp_train.drop('target', axis=1)
    mdlp_test = pd.read_csv('./test-mdlp.csv', dtype={'id': np.int32})
    test_mdlp = mdlp_test.replace(np.nan, -1)
    X, X_test = train_mdlp, test_mdlp
    return X, X_test

# example of usage
# X=train, y=labels, X_test=test

to_bin = X.loc[:, X.apply(lambda x: len(x.value_counts())) > 50].columns
X = X.replace(-1, np.nan)
X['target'] = y
X_test = X_test.replace(-1, np.nan)
class_label='target'
features = list(to_bin)

# define discretizer
discretizer = MDLP_Discretizer(dataset=X, testset=X_test, class_label=class_label, features=features,
                               out_path_data='./train-mdlp.csv', out_test_path_data='./test-mdlp.csv', out_path_bins=None)
X_d, X_test_d = load_discretized()

