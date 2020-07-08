import sys
import pandas as pd
from id3_tree import *
from computations import *
from naive_bayes import *

data_file = sys.argv[1]
depth = int(sys.argv[3])

data = pd.read_csv(data_file)

comp = Computations(data)
id3 = ID3Tree(depth, comp)

target = comp.take_target()
TRAIN_DATA = comp.take_train_data()
id3.set_train_data(TRAIN_DATA)

data_set_entropy = comp.calculate_data_set_entropy(TRAIN_DATA[target])

root = comp.decide_root()
root_values = comp.get_values(root)
id3.set_root(root, root_values)
id3.construct_tree_from_root()
id3_performance = id3.predict_targets()

prob_yes, prob_no = comp.get_probs()
naive_bayes = NaiveBayes(prob_yes, prob_no, comp)
naive_bayes.set_train_data(TRAIN_DATA)
naive_bayes.calculate_feature_probabilities()
nb_performance = naive_bayes.predict_targets()

print('DTLog: ', id3_performance, ' NB: ', nb_performance)
