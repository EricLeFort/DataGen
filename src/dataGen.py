#!/usr/local/bin/python2

# This script handles generating the data files.
# Aim for a dataset of 100k samples of dimension 10
# The samples are split evenly between classes and then evenly between types within those classes

import numpy as np
import pandas as pd
import math

#Pseudo-random seed made by mashing my keyboard, used for repeatability
np.random.seed(2975482625)

# Mean/stddev of each feature for the various types of each class
# The arrays are of the form: [class][type][feature]
# Features 1, 4, 6, and 9 should be virtually indistiguishable between classes
# Features 0, 2, 5, and 8 will have significant overlap but should be distinguishable in some cases
# Features 3 and 7  will have just some overlap and should usually be distinguishable
mean = [
	[
		[1500, 5924, 9.2, 29, 18.0, 62.4, 0.42, 635, 0.052, 87.01]
	],
	[
		[1300, 5924, 12.1, 68, 18.5, 73.2, 0.42, 840, 0.063, 87.03]
	]
]
stddev = [
	[
		[360, 1200, 3.0, 12, 4.2, 11.1, 0.02, 76, 0.021, 25]
	],
	[
		[290, 1500, 3.8, 15, 4.0, 13.0, 0.03, 89, 0.027, 25]
	]
]

samples = 1000
columns = ['class', 'type', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
cols = len(columns)

# Computes the value of the pdf of a normal distribution with a given mean and standard deviation
# at position x.
# This value can be anywhere in the range of (0, infinity)
def normal_pdf(x, mean, stddev):
	return math.exp(-(x - mean)**2 / (2*stddev**2)) / math.sqrt(2*math.pi*stddev**2)

# Generates the "real" dataset
def gen_real_data(samples, columns, mean, stddev):
	data = pd.DataFrame(columns=columns)

	for i in range(0, len(mean)):					# Each class
		for j in range(0, len(mean[i])):			# Each type
			subsamples = samples/len(mean)/len(mean[i])
			temp = pd.DataFrame(np.zeros(shape=(subsamples, cols)), columns=columns)
			temp['class'] = i
			temp['type'] = j
			for k in range(0, len(mean[i][j])):		# Each feature
				temp.iloc[:, k+2] = np.random.normal(mean[i][j][k], stddev[i][j][k], subsamples)
			data = data.append(temp, ignore_index=True)

	return data

# Generates the artificially generated dataset of either spikes or plateaus
# The plateaus will have a width equal to the spacing of the spikes
def gen_artificial_data(num_spikes, samples, columns, mean, stddev, is_plateau, deviations=3):
	data = pd.DataFrame(columns=columns)

	# The probabilities for each spike will be the same regardless of the specific mean/stddev
	# This is because the spike positioning is relative to the stddev and mean
	probabilities = []
	start = 1 - deviations
	spacing = deviations /((num_spikes - 1) / 2.0)
	for i in range (0, num_spikes):
		probabilities.append(normal_pdf(start + i*spacing, 1, 1))

	# Scale the probabilities to add to 1
	total = sum(probabilities)
	for i in range (0, num_spikes):
		probabilities[i] /= total

	# Convert into cumulative probabilities
	for i in range (1, num_spikes):
		probabilities[i] += probabilities[i-1]
	
	for i in range(0, len(mean)):					# Each class
		for j in range(0, len(mean[i])):			# Each type
			subsamples = samples/len(mean)/len(mean[i])
			temp = pd.DataFrame(np.zeros(shape=(subsamples, cols)), columns=columns)
			temp['class'] = i
			temp['type'] = j

			for k in range(0, len(mean[i][j])):		# Each feature
				start = mean[i][j][k] - deviations * stddev[i][j][k]
				spacing = (deviations*stddev[i][j][k]) / ((num_spikes - 1) / 2.0)
				values = []
				for l in range (0, num_spikes):
					values.append(start + l*spacing*1)
				temp.iloc[:, k+2] = np.random.uniform(0, 1, subsamples)

				# Assign the appropriate values depending on the random variable
				width = spacing / 2
				def temp_fun(x):
					# The shift allows sampling along plateaus of values with the same likelihood
					if(is_plateau):
						shift = np.random.uniform(-width, width)
					else:
						shift = 0

					for i in range(len(probabilities) - 1, -1, -1):
						if(x >= probabilities[i]):
							return values[i+1] + shift
					return values[0] + shift

				temp.iloc[:, k+2] = temp.iloc[:, k+2].apply(temp_fun)

			# Append this class/type combo to the dataset
			data = data.append(temp, ignore_index=True)

	return data

gen_real_data(samples, columns, mean, stddev).to_csv(path_or_buf="../data/validate.csv", index=False)
gen_real_data(samples, columns, mean, stddev).to_csv(path_or_buf="../data/realData.csv", index=False)
gen_artificial_data(3, int(0.9*samples), columns, mean, stddev, False).to_csv(path_or_buf="../data/spikes3.csv", index=False)
gen_artificial_data(5, int(0.9*samples), columns, mean, stddev, False).to_csv(path_or_buf="../data/spikes5.csv", index=False)
gen_artificial_data(9, int(0.9*samples), columns, mean, stddev, False).to_csv(path_or_buf="../data/spikes9.csv", index=False)
gen_artificial_data(3, int(0.9*samples), columns, mean, stddev, True).to_csv(path_or_buf="../data/plateaus3.csv", index=False)
gen_artificial_data(5, int(0.9*samples), columns, mean, stddev, True).to_csv(path_or_buf="../data/plateaus5.csv", index=False)
gen_artificial_data(9, int(0.9*samples), columns, mean, stddev, True).to_csv(path_or_buf="../data/plateaus9.csv", index=False)
