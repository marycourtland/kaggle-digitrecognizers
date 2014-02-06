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
    self.train_time = time.time() - t1
    if dbg: print ''
    if dbg: print 'Done training tree'
    
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
      return aggregate(extractLabels(data))
    
    # Otherwise: first, remove a variable from the list inserts it in the None's postion:
    # node = [var, [d, a, t, a]]
    v = popRandomChoice(variables) # this will remove the variable as a side-effect
    node[0] = v
    
    if dbg: print prefix, v.px_index
    
    # Partition the data and recurse on each partition
    # node := (var, recurse_on_first_partition, recurse_on_next_partition, ...)
    partitions = v.partition(data) # This comes out as a dict mapping cases to data subsets
    
    if dbg:
      print prefix,
      for case in partitions:
        print "{" + str(len(partitions[case])) + "}", 
      print ''
      
    for case in partitions:
      # It is unfortunate that we have to do a deepcopy of the vars list each time,
      # but i don't see a way around it so that we don't repeat or exclude variables
      partitions[case] = self.propagateTraining([None, partitions[case]], copy.deepcopy(variables), prefix+"  ")
    node[1] = partitions
    return node
    
  
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
  def __init__(self, num):
    
    

# Misc methods
def popRandomChoice(_list):
  # Remove a random variable from the list (as a side effect) and return it.
  v = random.choice(_list)
  _list.remove(v)
  return v

# Methods to manipulate the digit recognizer's data
def getPixel(digit_observation, i):
  return int(digit_observation[i+1])

def extractLabels(training_data):
  # The digit label is prepended at the beginning of each digit observation
  return [datum[0] for datum in training_data]
  
def aggregate(labels):
  # For the specific digit recognizer case, there are 10 independent labels.
  # Just return the most common.
  return Counter(labels).most_common(1)

def sampleTrainingData(data):
  # If there are N training observations, sample N from them (but with replacement)
  sample = []
  for i in range(len(data)):
    d = random.choice(range(len(data)))
    sample.append(data[d])
    train_sample_counts[d] += 1
  return sample

# If 3 is mapped to 40 in the following dict, then the training  
# observation at index 3 was used to train 40 trees.
# So essentially, it keeps track of which observations have been used.
# This will get initialized when the training data is read.
train_sample_counts = {}

if __name__ == '__main__':
  data_train = read_csv(test_data_file)
  train_sample_counts = {i:0 for i in range(len(data_train))}
  
  # The column headings are in the first row, so remove them
  headings = data_train.pop(0)
  
  # Create variables representing each pixel
  digit_vars = []
  
  for i in range(num_pixels):
    digit_vars.append(Variable(pixel_value_bins, i))
    
  # Create a classification tree!
  tree1 = ClassifierTree(digit_labels, digit_vars, 100)
  
  # Train it!
  tree1.train(sampleTrainingData(data_train))
  
