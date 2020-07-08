import math


class NaiveBayes:
    __comp = None
    __prob_yes = None
    __prob_no = None
    __feature_yes_probs = None
    __feature_no_probs = None
    __train_data = None
    __target = None
    __features = None

    def __init__(self, prob_yes, prob_no, comp):
        self.__prob_yes = prob_yes
        self.__prob_no = prob_no
        self.__comp = comp
        self.__features = comp.take_features()
        self.__target = comp.take_target()

    def calculate_feature_probabilities(self):
        self.__feature_yes_probs, self.__feature_no_probs = self.__comp.calculate_feature_probabilites()

    def set_train_data(self, train_data):
        self.__train_data = train_data

    def set_target(self, target):
        self.__target = target

    def set_features(self, features):
        self.__features = features

    def predict_targets(self):
        # 5 fold cross validation
        train_data = self.__train_data
        target = self.__target
        length = len(train_data[target])
        size = int(math.ceil(length / 5))
        times = int(len(train_data[target]) / size)
        features = self.__features
        yes_probs = self.__feature_yes_probs
        no_probs = self.__feature_no_probs
        performance_results = []
        start = 0
        end = size
        for j in range(times):
            correct_results = []
            false_results = []
            for i in range(start, end):
                yes_result = 1
                no_result = 1
                evidence = 1
                for feature in features:
                    val = train_data[feature][i]
                    val_prob = self.__comp.get_value_probability(feature, val)
                    yes_prob = yes_probs[feature][val]
                    no_prob = no_probs[feature][val]
                    yes_result = yes_result * yes_prob
                    no_result = no_result * no_prob
                    evidence = evidence * val_prob

                yes_result = yes_result * self.__prob_yes
                no_result = no_result * self.__prob_no

                real_yes = yes_result / evidence
                real_no = no_result / evidence

                if real_yes > real_no and train_data[target][i] == 1:
                    correct_results.append(1)
                elif real_no > real_yes and train_data[target][i] == 0:
                    correct_results.append(1)
                else:
                    false_results.append(0)

            trues = len(correct_results)
            falses = len(false_results)
            performance_results.append(trues / (trues + falses))

            start = start + size
            end = end + size

        if size * times < length:
            correct_results = []
            false_results = []
            for i in range(start, length):
                yes_result = 1
                no_result = 1
                evidence = 1
                for feature in features:
                    val = train_data[feature][i]
                    val_prob = self.__comp.get_value_probability(feature, val)
                    yes_prob = yes_probs[feature][val]
                    no_prob = no_probs[feature][val]
                    yes_result = yes_result * yes_prob
                    no_result = no_result * no_prob
                    evidence = evidence * val_prob

                yes_result = yes_result * self.__prob_yes
                no_result = no_result * self.__prob_no

                real_yes = yes_result / evidence
                real_no = no_result / evidence

                if real_yes > real_no and train_data[target][i] == 1:
                    correct_results.append(1)
                elif real_no > real_yes and train_data[target][i] == 0:
                    correct_results.append(1)
                else:
                    false_results.append(0)

                trues = len(correct_results)
                falses = len(false_results)
                performance_results.append(trues / (trues + falses))

        performance_sum = 0
        for x in performance_results:
            performance_sum = performance_sum + x

        result = performance_sum / len(performance_results)
        return result
