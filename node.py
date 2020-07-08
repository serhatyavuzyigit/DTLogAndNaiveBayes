class Node:

    __parent = None
    __depth = None
    __value = None
    __data = None
    __feature = None
    __children_values = None
    __children_nodes = None

    def __init__(self, value, data, feature, children):
        self.__value = value
        self.__data = data
        self.__children_values = children
        self.__feature = feature


    def set_feature(self, feature):
        self.__feature = feature

    def get_feature(self):
        return self.__feature

    def set_parent(self, parent):
        self.__parent = parent

    def get_parent(self):
        return self.__parent

    def set_depth(self, depth):
        self.__depth = depth

    def get_depth(self):
        return self.__depth

    def set_children_values(self, children_values):
        self.__children_values = children_values

    def get_children_values(self):
        return self.__children_values

    def set_children_nodes(self, children_nodes):
        self.__children_nodes = children_nodes

    def get_children_nodes(self):
        return self.__children_nodes

    def get_value(self):
        return self.__value

    def get_data(self):
        return self.__data

    def __str__(self):
        return "value: {} data: {} children_values: {}".format(
            self.__value,
            self.__data,
            self.__children_values,)
