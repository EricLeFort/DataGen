#!/usr/local/bin/python2
# This script handles training a couple models on the various datasets to measure their efficacy.
# For the sake of this experiment we'll try a neural net, SVM, and a decision tree

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle

#Pseudo-random seed made by mashing my keyboard, used for repeatability
np.random.seed(971381386)
dataset = "plateaus"
verbose = False

#Model definitions
if(dataset == "full"):
	nn = MLPClassifier(
		alpha=0.1,
		learning_rate_init=0.1,
		batch_size=128,
		hidden_layer_sizes=(25, 25, 25),
		verbose=verbose)
	svm = SVC(
		kernel='rbf',
		C=5,
		gamma=0.001,
		max_iter=2000)
	tree = DecisionTreeClassifier(
		splitter='best',
		min_impurity_decrease=0.01,
		max_leaf_nodes=10,
		min_weight_fraction_leaf=0.01,
		max_features=6,
		class_weight='balanced')

elif(dataset == "real"):
	nn = MLPClassifier(
		alpha=0.0001,
		learning_rate_init=0.001,
		batch_size=64,
		hidden_layer_sizes=(25, 25, 25),
		verbose=verbose)
	svm = SVC(
		kernel='rbf',
		C=5,
		gamma=0.1,
		max_iter=2000)
	tree = DecisionTreeClassifier(
		splitter='random',
		min_impurity_decrease=0.02,
		max_leaf_nodes=6,
		min_weight_fraction_leaf=0.05,
		max_features=10,
		class_weight='balanced')

elif(dataset == "spikes"):
	nn = MLPClassifier(
		alpha=0.1,
		learning_rate_init=0.01,
		batch_size=32,
		hidden_layer_sizes=(25, 25, 25),
		verbose=verbose)
	svm = SVC(
		kernel='rbf',
		C=0.5,
		gamma=0.1,
		max_iter=2000)
	tree = DecisionTreeClassifier(
		splitter='best',
		min_impurity_decrease=0.01,
		max_leaf_nodes=10,
		min_weight_fraction_leaf=0,
		max_features=6,
		class_weight='balanced')

else:
	nn = MLPClassifier(
		alpha=0.0001,
		learning_rate_init=0.01,
		batch_size=64,
		hidden_layer_sizes=(25, 25, 25),
		verbose=verbose)
	svm = SVC(
		kernel='rbf',
		C=0.5,
		gamma=0.1,
		max_iter=2000)
	tree = DecisionTreeClassifier(
		splitter='best',
		min_impurity_decrease=0.01,
		max_leaf_nodes=6,
		min_weight_fraction_leaf=0,
		max_features=8,
		class_weight='balanced')

# Load train/test/validate data and split into x and y
# dataset - string dictating which dataset to use
#	"full" - Returns the full "real data" dataset
#	"real" - Returns 10% of the "real data" dataset
#	"spikes" - Returns the spikes dataset along with 10% of the "real data" dataset mixed in
#	"plateaus" - Returns the plateaus dataset along with 10% of the "real data" dataset mixed in
# num_spikes - int dictating the number of spikes to use. Can be 3, 5, or 9.
# scale - Whether to scale the data
def load_data(dataset, num_spikes, scale):
	if(dataset == "full"):
		data = pd.read_csv("../data/realData.csv")
	elif(dataset == "real"):
		data = pd.read_csv("../data/realData.csv")
		data = data.sample(int(data.shape[0]*0.1))
	elif(dataset == "spikes"):
		data = pd.read_csv("../data/realData.csv")
		data = data.sample(int(data.shape[0]*0.1))
		data = data.append(pd.read_csv("../data/spikes" + str(num_spikes) + ".csv"))
	else:			# "plateaus"
		data = pd.read_csv("../data/realData.csv")
		data = data.sample(int(data.shape[0]*0.1))
		data = data.append(pd.read_csv("../data/plateaus" + str(num_spikes) + ".csv"))

	data = shuffle(data)
	val = pd.read_csv("../data/validate.csv")
	train = data[:int(data.shape[0]*0.8)].reset_index(drop=True)
	test = data[int(data.shape[0]*0.8):].reset_index(drop=True)

	x_train = train.drop(["class", "type"], axis=1)
	y_train = train["class"]
	x_test = test.drop(["class", "type"], axis=1)
	y_test = test["class"]
	x_val = val.drop(["class", "type"], axis=1)
	y_val = val["class"]

	if scale:
		scaler = StandardScaler().fit(x_train)
		x_train = scaler.transform(x_train)
		x_test = scaler.transform(x_test)
		x_val = scaler.transform(x_val)

	return x_train, y_train, x_test, y_test, x_val, y_val

def perform_grid_search(estimator, param_grid, detailed=True):
	scores = ['precision', 'recall']
	for score in scores:
		print("# Tuning hyper-parameters for %s" % score)
		print("")

		clf = GridSearchCV(
				estimator,
				param_grid,
				cv=5,
				scoring='%s_macro' % score
		)
		clf.fit(x_train, y_train)

		print("Best parameters set found on development set:")
		print("")
		print(clf.best_params_)
		if detailed:
			print("")
			print("Grid scores on development set:")
			print("")
			means = clf.cv_results_['mean_test_score']
			stds = clf.cv_results_['std_test_score']
			for mean, std, params in zip(means, stds, clf.cv_results_['params']):
				print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
			print("")

		print("Detailed classification report:")
		print("")
		print("The model is trained on the full development set.")
		print("The scores are computed on the full evaluation set.")
		print("")
		y_true, y_pred = y_test, clf.predict(x_test)
		print(classification_report(y_true, y_pred))

def compute_accuracy_measures(confusion_matrix):
	acc = (0.0 + confusion[0][0] + confusion[1][1]) / (confusion[0][0] + confusion[0][1] + confusion[1][0] + confusion[1][1]) * 100
	precision = confusion[0][0] / (0.0 + confusion[0][0] + confusion[0][1]) * 100
	recall = confusion[1][1] / (0.0 + confusion[1][0] + confusion[1][1]) * 100

	return acc, precision, recall

def train_and_test(dataset):
	x_train, y_train, x_test, y_test, x_val, y_val = load_data(dataset, 9)

#Load and prepare train/test/validate datasets
x_train, y_train, x_test, y_test, x_val, y_val = load_data(dataset, 9, True)

#Hyperparameter grids to search
nn_param_grid = [
	{
		'hidden_layer_sizes': [(25, 25, 25)],
		'alpha': [1e-4, 1e-3, 1e-2, 1e-1],
		'batch_size': [32, 64, 128],
		'learning_rate_init': [1e-3, 1e-2, 1e-1]
	}
]
svm_param_grid = [
	{
		'kernel': ['linear'],
  		'C': [0.1, 0.5, 1, 5, 10]
	}, {
		'kernel': ['poly'],
		'C': [0.1, 0.5, 1, 5, 10],
		'degree': [3, 4, 5]
	}, {
		'kernel': ['rbf'],
		'C': [0.1, 0.5, 1, 5, 10],
		'gamma': [0.1, 0.01, 0.001, 0.0001]
	}, {
		'kernel': ['sigmoid'],
		'C': [0.1, 0.5, 1, 5, 10],
		'gamma': [0.1, 0.01, 0.001, 0.0001]
	}
]
tree_param_grid = [
	{
		'splitter': ['best', 'random'],
		'class_weight': ['balanced'],
		'min_weight_fraction_leaf': [0, 0.01, 0.05],
		'max_features': [6, 8, 10],
		'max_leaf_nodes': [6, 8, 10],
		'min_impurity_decrease': [0.01, 0.02, 0.03, 0.04, 0.05]
	}
]

#Tuning hyperparameters
#perform_grid_search(MLPClassifier(), nn_param_grid, False)
#perform_grid_search(SVC(), svm_param_grid, False)
#perform_grid_search(DecisionTreeClassifier(), tree_param_grid, False)
#import sys
#sys.exit()

#Train and test
nn.fit(x_train, y_train)
print("\nNeural Network")
confusion = confusion_matrix(y_train, nn.predict(x_train))
print("Train accuracy: \t" + str(compute_accuracy_measures(confusion)[0]) + "%")
print(confusion)
confusion = confusion_matrix(y_test, nn.predict(x_test))
print("Test accuracy: \t\t" + str(compute_accuracy_measures(confusion)[0]) + "%")
print(confusion)
confusion = confusion_matrix(y_val, nn.predict(x_val))
print("Validate")
print("\taccuracy: \t" + str(compute_accuracy_measures(confusion)[0]) + "%")
print("\tprecision: \t" + str(compute_accuracy_measures(confusion)[1]) + "%")
print("\trecall: \t" + str(compute_accuracy_measures(confusion)[2]) + "%")
print(confusion)
print("\n\n")

svm.fit(x_train, y_train)
print("SVM")
confusion = confusion_matrix(y_train, svm.predict(x_train))
print("Train accuracy: \t" + str(compute_accuracy_measures(confusion)[0]) + "%")
print(confusion)
confusion = confusion_matrix(y_test, svm.predict(x_test))
print("Test accuracy: \t\t" + str(compute_accuracy_measures(confusion)[0]) + "%")
print(confusion)
confusion = confusion_matrix(y_val, svm.predict(x_val))
print("Validate")
print("\taccuracy: \t" + str(compute_accuracy_measures(confusion)[0]) + "%")
print("\tprecision: \t" + str(compute_accuracy_measures(confusion)[1]) + "%")
print("\trecall: \t" + str(compute_accuracy_measures(confusion)[2]) + "%")
print(confusion)
print("\n\n")

tree.fit(x_train, y_train)
print("Decision Tree")
confusion = confusion_matrix(y_train, tree.predict(x_train))
print("Train accuracy: \t" + str(compute_accuracy_measures(confusion)[0]) + "%")
print(confusion)
confusion = confusion_matrix(y_test, tree.predict(x_test))
print("Test accuracy: \t\t" + str(compute_accuracy_measures(confusion)[0]) + "%")
print(confusion)
confusion = confusion_matrix(y_val, tree.predict(x_val))
print("Validate")
print("\taccuracy: \t" + str(compute_accuracy_measures(confusion)[0]) + "%")
print("\tprecision: \t" + str(compute_accuracy_measures(confusion)[1]) + "%")
print("\trecall: \t" + str(compute_accuracy_measures(confusion)[2]) + "%")
print(confusion)