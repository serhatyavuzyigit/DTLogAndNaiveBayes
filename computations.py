
import math


class Computations:
    __data = None
    __train_data = None
    __target = None
    __data_set_entropy = None
    __features = None

    def __init__(self, data):
        self.__data = data

    def take_features(self):
        column_size = len(self.__data.columns)
        _features = []
        for i in range(1, column_size - 1):
            _features.append(self.__data.columns[i])

        self.__features = _features
        return _features

    def take_target(self):
        column_size = len(self.__data.columns)
        self.__target = self.__data.columns[column_size - 1]
        return self.__data.columns[column_size - 1]

    def take_train_data(self):
        _TRAIN_DATA = {}
        features = self.take_features()
        data = self.__data
        size = len(self.__data)
        target = self.take_target()
        for x in features:
            _TRAIN_DATA[x] = data.loc[0:size, x].values
        _TRAIN_DATA[target] = data.loc[0:size, target].values
        self.__train_data = _TRAIN_DATA
        return _TRAIN_DATA

    def calculate_data_set_entropy(self, target):
        total = len(target)
        num_of_positive = 0
        num_of_negative = 0
        for i in target:
            if i == 0:
                num_of_negative = num_of_negative + 1
            if i == 1:
                num_of_positive = num_of_positive + 1

        entropy = -(num_of_positive / total) * math.log2(num_of_positive / total) - (
                num_of_negative / total) * math.log2(num_of_negative / total)
        self.__data_set_entropy = entropy
        return entropy

    def calculate_average_information_entropy(self, feature):
        negative_res_features = {}
        positive_res_features = {}
        values = self.__train_data[feature]
        target = self.__target
        for val in values:
            negative_res_features[val] = 0
            positive_res_features[val] = 0
        train_data = self.__train_data

        total_size = len(train_data[target])
        for i in range(total_size):
            result = train_data[target][i]
            if result == 0:
                negative_res_features[train_data[feature][i]] = negative_res_features[train_data[feature][i]] + 1
            if result == 1:
                positive_res_features[train_data[feature][i]] = positive_res_features[train_data[feature][i]] + 1

        entropies = {}
        for val in values:
            num_of_positive = positive_res_features[val]
            num_of_negative = negative_res_features[val]
            total = num_of_negative + num_of_positive
            if num_of_negative == 0 or num_of_positive == 0:
                entropies[val] = 0
            else:
                entropies[val] = -(num_of_positive / total) * math.log2(num_of_positive / total) - (
                        num_of_negative / total) * math.log2(num_of_negative / total)

        avg_inf_entropy = 0
        for val in positive_res_features:
            pos = positive_res_features[val]
            neg = negative_res_features[val]
            ent = entropies[val]
            avg_inf_entropy = avg_inf_entropy + ((pos + neg) / total_size) * ent

        return avg_inf_entropy

    def calculate_subset_data_average_information_entropy(self, feature, subset_data):
        negative_res_features = {}
        positive_res_features = {}
        values = subset_data[feature]
        target = self.__target
        for val in values:
            negative_res_features[val] = 0
            positive_res_features[val] = 0
        train_data = subset_data

        total_size = len(train_data[target])
        for i in range(total_size):
            result = train_data[target][i]
            if result == 0:
                negative_res_features[train_data[feature][i]] = negative_res_features[train_data[feature][i]] + 1
            if result == 1:
                positive_res_features[train_data[feature][i]] = positive_res_features[train_data[feature][i]] + 1

        entropies = {}
        for val in values:
            num_of_positive = positive_res_features[val]
            num_of_negative = negative_res_features[val]
            total = num_of_negative + num_of_positive
            if num_of_negative == 0 or num_of_positive == 0:
                entropies[val] = 0
            else:
                entropies[val] = -(num_of_positive / total) * math.log2(num_of_positive / total) - (
                        num_of_negative / total) * math.log2(num_of_negative / total)

        avg_inf_entropy = 0
        for val in positive_res_features:
            pos = positive_res_features[val]
            neg = negative_res_features[val]
            ent = entropies[val]
            avg_inf_entropy = avg_inf_entropy + ((pos + neg) / total_size) * ent

        return avg_inf_entropy

    def decide_root(self):
        data_set_entropy = self.__data_set_entropy
        features = self.__features
        _max = -1
        max_feature = features[0]
        for feature in features:
            avg_ent = self.calculate_average_information_entropy(feature)
            gain = data_set_entropy - avg_ent
            if _max < gain:
                max_feature = feature
                _max = gain

        return max_feature

    def decide_root_in_subset_data(self, feature, subset_data, entropy):
        train_data = subset_data
        subset_data_set_entropy = entropy
        features = self.__features
        _max = -1
        max_feature = ''
        for fea in features:
            if fea != feature:
                avg_ent = self.calculate_subset_data_average_information_entropy(fea, train_data)
                gain = subset_data_set_entropy - avg_ent
                if _max < gain:
                    max_feature = fea
                    _max = gain
        return max_feature



    def get_values(self, feature):
        res = {}
        values = self.__train_data[feature]
        for val in values:
            res[val] = 0
        ans = []
        for x in res:
            ans.append(x)
        return ans

    def take_subset_data(self, train_data, feature, value):
        subset_data = {}
        data = train_data
        target = self.__target
        features = self.__features
        for fea in features:
            subset_data[fea] = []
        subset_data[target] = []

        for i in range(len(data[target])):
            if data[feature][i] == value:
                for fea in features:
                    subset_data[fea].append(data[fea][i])
                subset_data[target].append(data[target][i])
        return subset_data

    def calculate_subset_entropy(self, train_data):
        subset_data = {}
        data = train_data
        target = self.__target
        subset_data[target] = []

        for i in range(len(data[target])):
            subset_data[target].append(data[target][i])

        num_of_positive = 0
        num_of_negative = 0
        for i in range(len(subset_data[target])):
            if subset_data[target][i] == 1:
                num_of_positive = num_of_positive + 1
            if subset_data[target][i] == 0:
                num_of_negative = num_of_negative + 1
        total = num_of_positive + num_of_negative

        entropy = 0
        if num_of_negative != 0 and num_of_positive != 0:
            entropy = -(num_of_positive / total) * math.log2(num_of_positive / total) - (
                num_of_negative / total) * math.log2(num_of_negative / total)

        return entropy


    def get_probs(self):
        train_data = self.__train_data
        target = self.__target
        pos = 0
        neg = 0

        for x in train_data[target]:
            if x == 0:
                neg = neg + 1
            if x == 1:
                pos = pos + 1

        return (pos/(pos+neg)), (neg/(pos+neg))

    def get_pos_and_neg(self):
        train_data = self.__train_data
        target = self.__target
        pos = 0
        neg = 0

        for x in train_data[target]:
            if x == 0:
                neg = neg + 1
            if x == 1:
                pos = pos + 1

        return pos,neg

    def calculate_feature_probabilites(self):
        train_data = self.__train_data
        target = self.__target
        features = self.__features
        feature_yes_probs = {}
        feature_no_probs = {}
        total_pos, total_neg = self.get_pos_and_neg()
        for feature in features:
            yes_results = {}
            no_results = {}
            values = self.get_values(feature)
            for val in values:
                pos = 0
                neg = 0
                for i in range(len(train_data[target])):
                    if train_data[feature][i] == val:
                        if train_data[target][i] == 0:
                            neg = neg + 1
                        if train_data[target][i] == 1:
                            pos = pos + 1

                yes_results[val] = pos / total_pos
                no_results[val] = neg / total_neg

            feature_yes_probs[feature] = yes_results
            feature_no_probs[feature] = no_results

        return feature_yes_probs, feature_no_probs

    def get_value_probability(self, feature, value):
        train_data = self.__train_data
        total = len(train_data[feature])
        occr = 0
        for i in range(total):
            if train_data[feature][i] == value:
                occr = occr + 1

        return occr / total



