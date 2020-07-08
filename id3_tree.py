from node import *
import math
from sklearn.linear_model import LogisticRegression

class ID3Tree:
    __tree_depth = None
    __root_node = None
    __train_data = None
    __comp = None

    def __init__(self, depth, comp):
        self.__tree_depth = depth
        self.__comp = comp

    def set_train_data(self, train_data):
        self.__train_data = train_data

    def set_root(self, root, children):
        self.__root_node = Node(root, None, root, children)
        self.__root_node.set_depth(0)

    def construct_tree(self, children_nodes, _train_data):
        for node in children_nodes:
            if node.get_data() != 'YES' and node.get_data() != 'NO' and node.get_depth() < self.__tree_depth:
                subset_data = self.__comp.take_subset_data(_train_data, node.get_feature(), node.get_value())
                entropy = self.__comp.calculate_subset_entropy(subset_data)
                root_value = self.__comp.decide_root_in_subset_data(node.get_feature(), subset_data, entropy)
                node.set_feature(root_value)
                feature = root_value
                node.set_children_values(self.__comp.get_values(feature))
                children_nodes = []
                target = self.__comp.take_target()
                for child in node.get_children_values():
                    results = []
                    for i in range(len(subset_data[target])):
                        if subset_data[feature][i] == child:
                            results.append(subset_data[target][i])
                    counter = 0
                    for x in results:
                        counter = counter + x

                    if counter == len(results):
                        # yes
                        yes_node = Node(child, 'YES', feature, None)

                        yes_node.set_depth(node.get_depth()+1)
                        yes_node.set_parent(node)
                        children_nodes.append(yes_node)

                    elif counter == 0:
                        # no
                        no_node = Node(child, 'NO', feature, None)
                        no_node.set_depth(node.get_depth()+1)
                        no_node.set_parent(node)
                        children_nodes.append(no_node)

                    else:
                        node = Node(child, None, feature, None)
                        node.set_depth(node.get_depth()+1)
                        node.set_parent(node)
                        children_nodes.append(node)

                node.set_children_nodes(children_nodes)
                self.construct_tree(children_nodes, subset_data)


    def construct_tree_from_root(self):
        target = self.__comp.take_target()
        root = self.__root_node
        train_data = self.__train_data
        feature = root.get_feature()
        children_nodes = []
        for child in root.get_children_values():
            results = []
            for i in range(len(train_data[target])):
                if train_data[feature][i] == child:
                    results.append(train_data[target][i])
            counter = 0
            for x in results:
                counter = counter + x

            if counter == len(results):
                # yes
                yes_node = Node(child, 'YES', feature, None)

                yes_node.set_depth(1)
                yes_node.set_parent(root)
                children_nodes.append(yes_node)

            elif counter == 0:
                # no
                no_node = Node(child, 'NO', feature, None)
                no_node.set_depth(1)
                no_node.set_parent(root)
                children_nodes.append(no_node)

            else:
                node = Node(child, None, feature, None)
                node.set_depth(1)
                node.set_parent(root)
                children_nodes.append(node)

        root.set_children_nodes(children_nodes)
        self.construct_tree(children_nodes, train_data)

    def predict_with_regressor(self, subset_data, feature_values):
        target = self.__comp.take_target()
        subset_target = subset_data[target]
        del subset_data[target]

        c = 0
        for x in subset_target:
            c = c + x
        if c == len(subset_target) or c == 0:
            if c == len(subset_target):
                return 'YES'
            else:
                return 'NO'

        h = len(subset_target)
        w = 0
        for x in subset_data:
            w = w + 1
        features = self.__comp.take_features()
        data = [[0 for x in range(w)] for y in range(h)]
        for i in range(h):
            for j in range(w):
                for fea in features:
                    data[i][j] = subset_data[fea][i]

        logistic_regression = LogisticRegression(solver='newton-cg')
        logistic_regression.fit(data, subset_target)

        query = []
        for x in features:
            query.append(feature_values[x])

        result = logistic_regression.predict([query])[0]
        if result == 1:
            return 'YES'
        else :
            return 'NO'


    def predict_query(self, feature_values):
        subset_data = self.__train_data
        localroot = self.__root_node
        while localroot.get_children_nodes() != None:
            val = feature_values[localroot.get_feature()]
            for child in localroot.get_children_nodes():
                if child.get_value() == val:
                    localroot = child
                    subset_data = self.__comp.take_subset_data(subset_data, localroot.get_feature(), child.get_value())
            if localroot.get_data() == 'YES' or localroot.get_data() == 'NO':
                return localroot.get_data()

        return self.predict_with_regressor(subset_data, feature_values)

    def predict_targets(self):
        # 5 fold cross validation
        train_data = self.__train_data
        target = self.__comp.take_target()
        length = len(train_data[target])
        size = int(math.ceil(length / 5))
        times = int(len(train_data[target]) / size)
        features = self.__comp.take_features()
        performance_results = []
        start = 0
        end = size
        for j in range(times):
            correct_results = []
            false_results = []
            for i in range(start, end):
                feature_values = {}
                for feature in features:
                    feature_values[feature] = train_data[feature][i]
                real_target = train_data[target][i]
                result = self.predict_query(feature_values)
                if result == 'YES' and real_target == 1:
                    correct_results.append(1)
                elif result == 'NO' and real_target == 0:
                    correct_results.append(1)
                else :
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
                feature_values = {}
                for feature in features:
                    feature_values[feature] = train_data[feature][i]
                real_target = train_data[target][i]
                result = self.predict_query(feature_values)
                if result == 'YES' and real_target == 1:
                    correct_results.append(1)
                elif result == 'NO' and real_target == 0:
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


    def print_tree_aux(self, children_nodes):
        if children_nodes != None:
            for child in children_nodes:
                if child.get_data() == 'YES' or child.get_data() == 'NO':
                    print(child.get_value())
                    print(child.get_data())
                else :
                    print(child.get_value())
                    print(child.get_feature())
                    self.print_tree_aux(child.get_children_nodes())

    def print_tree(self):
        print(self.__root_node.get_feature())
        self.print_tree_aux(self.__root_node.get_children_nodes())
