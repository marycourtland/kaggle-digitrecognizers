# Random forest classifier for Kaggle digit recognizer data
#
# Created 2/5/2014
#
# References used:
# (1) http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm
# (2) http://oz.berkeley.edu/users/breiman/Using_random_forests_V3.1.pdf

import csv
import copy
import time
from collections import Counter
import random
random.seed(42)

# Parameters and settings
train_data_file = "../train.csv"
test_data_file = "../test.csv"

num_trees_in_forest = 1000
num_vars_tried_per_node = 28 # Source (2) claims that optimal value is sqrt(num_total_vars)

num_pixels = 28 * 28
digit_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
pixel_value_binsize = 32 # this should be a divisor of 256
pixel_value_bins = [(i, i*pixel_value_binsize, (i+1)*pixel_value_binsize) for i in range(256/pixel_value_binsize)]

dbg = True # this will print each classification tree in tree-line format

# Method to read CSV data
def read_csv(file_path):
  array = []
  with open(file_path, 'rb') as openfile:
    reader = csv.reader(openfile)
    for line in reader:
      array.append(line)
  return array


# A Variable object represents one particular feature of an observation
# TODO: make this general, not just for the digit recognizer
# For the digit recognizer, the variables are the image's pixels.
# Each pixel is a continuous value and should be binned.
class Variable:
  def __init__(self, cases, px_index):
    # For this specific problem, each case is a tuple of (id, min, max).
    # A pixel value falls into a case if it's >= the min and < the max.
    self.cases = cases
    
    # This is unique to this specific problem.
    # It refers to the index of the relevant pixel in the image.
    self.px_index = px_index 
  
  def getCase(self, observation):
    # Determine which case is the observation
    # TODO: make this general, not just for quantitative vars
    for (id, min, max) in self.cases:
      if getPixel(observation, self.px_index) < min: continue
      if getPixel(observation, self.px_index) >= max: continue
      # TODO: I'm pretty sure that only an ID is needed to be returned,
      # but check this.
      return id
    # If code reaches here, then the observation doesn't fit any case
    return None
      
  def partition(self, data):
    # The data should be a list of observations. No side effects will occur on it.
    # Partition it into sublists, each corresponding to each of this
    # variable's cases.
    # TODO: abstract the case (id,min,max) away from specific problem
    
    # Setup a dict mapping case IDs to subsets of the data
    partitions = {}
    for (id, min, max) in self.cases: partitions[id] = []
    
    # TODO: should None be a category? If not, then error should be thrown
    # when an observation doesn't fit any of the cases.
    partitions[None] = []
    
    for observation in data:
      partitions[self.getCase(observation)].append(observation)
    
    return partitions



# A single classification tree
# A node is represented by a simple list. The first element of this list
# is the variable object; the rest of the elements are nodes or leaves.
class ClassifierTree:
  def __init__(self, labels, variables, min_nodesize=100):
    self.labels = labels
    self.variables = variables
    self.first_node = []
    self.train_time = -1 # Time to train, in seconds
    
    # This is the fewest number of observations in a leaf node
    self.min_nodesize = min_nodesize
  
    
  def train(self, data):
    # Data should be a list of observations WITH LABELS INSIDE
    # Labels should be a list of labels corresponding to the observations
    if dbg: print 'Training tree'
    t0 = time.time()
    self.first_node = self.propagateTraining([None, data], copy.deepcopy(self.variables))
    self.train_time = time.time() - t0
    if dbg: print ''
    if dbg: print 'Done training tree. Time (secs):', self.train_time
    
  def propagateTraining(self, node, variables, prefix=""):
    # This recursive function takes a list of form:
    #   node = [None, [d, a, t, a]]
    # And it also takes a list of available variables.
    # It will transform the node into this form:
    #   node = [var, {var_case1:[node_recursed_on_datasubset], ...}]
    
    data = node[1]
    
    # If the data list is the min nodesize, or if we've run out of vars, then this is a leaf.
    # - Find the labels associated with each datum and aggregate them
    # - Replace the datalist with the aggregated label
    # - Return that label
    if (len(data) <= self.min_nodesize):
      if dbg: print prefix, '_'
      return aggregate(getAllLabels(data))
    
    # Otherwise: find the best-splitting variable and split the data by it
    # node = [var, [d, a, t, a]]
    v, split = self.getBestSplit(data, variables)
    node[0] = v
    variables.remove(v)
    
    if dbg: print prefix, v.px_index
    
    # Recurse on each partition
    # node := (var, recurse_on_first_split, recurse_on_next_split, ...)
    
    if dbg:
      print prefix,
      for case in split:
        print "{" + str(case) + ":" + str(len(split[case])) + "}", 
      print ''
      
    for case in split:
      # It is unfortunate that we have to do a deepcopy of the vars list each time,
      # but i don't see a way around it so that we don't repeat or exclude variables
      split[case] = self.propagateTraining([None, split[case]], copy.deepcopy(variables), prefix+"  ")
    node[1] = split
    return node
    
  def getBestSplit(self, data, available_variables):
    # Returns the variable & split which maximizes the gini impurity
    # No side effects occur on the list of variables
    
    # Choose subset of variables to try
    vars_to_try = random.sample(available_variables, num_vars_tried_per_node)
    
    # Try splitting the data by each variable;
    # find the split w/ maximum gini impurity
    max_gini = 0
    max_gini_split = None
    max_gini_var = None
    for variable in vars_to_try:
      split = variable.partition(data)
      gini = giniImpurity(split)
      if gini > max_gini:
        max_gini = gini
        max_gini_split = split
        max_gini_var = variable
    
    return [max_gini_var, max_gini_split]
    
  
  # Classify an observation!
  def classify(self, observation):
    if len(self.first_node) is 0:
      print "Cannot classify with an empty tree"
      return None
    return self.propagateClassifiction(observation, self.first_node)
    
  def propagateClassification(self, observation, node):
    # If the node isn't a list, then it's a leaf label. Return that.
    if type(node) != list: return node
    
    var = node[0] # the variable that is being split
    casenodes = node[1] # dict mapping cases -> more nodes in the tree
    case = var.getCase(observation)
    return propagateClassification(observation, casenodes[case])

class RandomForest:
  def __init__(self): pass
    
    

# Misc methods
def popRandomChoice(_list):
  # Remove a random variable from the list (as a side effect) and return it.
  v = random.choice(_list)
  _list.remove(v)
  return v
  
# Compute gini impurity of a list of observations
# Ig = 1 - (sum of squares of fractions of labels)
def giniImpurity(grouped_data):
  # The grouped_data should be a dict mapping categories to subsets of
  # data which fit those category
  N = 0.0
  for group in grouped_data:
    N += len(grouped_data[group])
  
  gini_sum = 0
  # Count the data in each label group, and normalize the count
  # Square it and add it to the summation
  for group in grouped_data:
    gini_sum += (len(grouped_data[group]) / N) ** 2
  
  return 1 - gini_sum

# Methods to manipulate the digit recognizer's data
def getPixel(digit_observation, i):
  return int(digit_observation[i+1])

def getLabel(observation):
  # The digit label is prepended at the beginning of each digit observation
  return observation[0]

def getAllLabels(training_data):
  # Returns a list of the labels of the training data
  return [getLabel(observation) for observation in training_data]

def groupByLabel(training_data):
  # Returns a dict mapping label --> list of observations with that label
  labelled_data = {l:[] for l in digit_labels}
  for observation in data:
    labelled_data[getLabel(observation)].append(observation)
  return labelled_data

def aggregate(labels):
  # 'Vote' on one out of many labels.
  # For the specific digit recognizer case, there are 10 independent labels.
  # Just return the most common.
  return Counter(labels).most_common(1)

def sampleTrainingData(data):
  # If there are N training observations, sample N from them (but with replacement)
  # This is bagging
  sample = []
  for i in range(len(data)):
    d = random.choice(range(len(data)))
    sample.append(data[d])
    if (d not in train_sample_counts): train_sample_counts[d] = 0
    train_sample_counts[d] += 1
  return sample

# If 3 is mapped to 40 in the following dict, then the training  
# observation at index 3 was used to train 40 trees.
# So essentially, it keeps track of which observations have been used.
# This will get initialized when the training data is read.
train_sample_counts = {}


def main():
  data_train = read_csv(train_data_file)
  
  # The column headings are in the first row, so remove them
  headings = data_train.pop(0)
  
  train_sample_counts = {i:0 for i in range(len(data_train))}
  
  # Create variables representing each pixel
  digit_vars = []
  
  for i in range(num_pixels):
    digit_vars.append(Variable(pixel_value_bins, i))
    
  # Create a classification tree!
  tree1 = ClassifierTree(digit_labels, digit_vars, 100)
  
  # Train it!
  tree1.train(sampleTrainingData(data_train))
  

if __name__ == '__main__': main()
