from scipy.spatial.distance import euclidean
from scipy.stats import mode
from sklearn.model_selection import train_test_split
import numpy as np
import time


neighbors = 8
training_rounds = 10


# Locate the most similar neighbors
def get_neighbor_labels(train_x, train_y, test_row):
	distances = list()
	# use i to mark the index of the train_row within train_x
	i = 0
	for train_row in train_x:
		dist = euclidean(test_row, train_row)
		distances.append((i, dist))
		i += 1
	distances.sort(key=lambda tup: tup[1])
	neighbor_labels = list()
	for i in range(neighbors):
		# get the index of the neighbors from the training data, then look up their label and add it
		location = distances[i][0]
		neighbor_labels.append(train_y[location])
	return neighbor_labels


def classify_one_row(train_x, train_y, test_row):
	start_time = time.time()
	nearest_labels = get_neighbor_labels(train_x, train_y, test_row)
	# ties are settled by selecting the smallest value
	mode_results = mode(nearest_labels)
	# if all values tie at 1, select the closest label
	if mode_results.count[0] == 1:
		result = nearest_labels[0]
	else:
		result = mode_results.mode[0]
	elapsed_time = time.time() - start_time
	# print(elapsed_time)
	return result


def classify_test_data(train_x, train_y, test_x):
	predictions = list()
	for test_row in test_x:
		predictions.append(classify_one_row(train_x, train_y, test_row))
	return predictions


def accuracy(predicted_labels, test_y):
	correct = 0
	for i in range(len(predicted_labels)):
		if predicted_labels[i] == test_y[i]:
			correct += 1
	score = correct / float(len(test_y)) * 100
	print('{} correct predictions were made for a score of {}%'.format(correct, score))


def evaluate(features, labels):
	for _ in range(training_rounds):
		x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
		predictions = classify_test_data(x_train, y_train, x_test)
		accuracy(predictions, y_test)


if __name__ == '__main__':
	dataset = np.array([[2.7810836, 2.550537003],[1.465489372, 2.362125076],[3.396561688, 4.400293529],[1.38807019, 1.850220317],[3.06407232, 3.005305973],[7.627531214, 2.759262235],[5.332441248, 2.088626775],[6.922596716, 1.77106367],[8.675418651, -0.242068655],[7.673756466, 3.508563011]])
	label = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
	evaluate(dataset, label)
