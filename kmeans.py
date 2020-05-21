from scipy.spatial.distance import euclidean
import numpy as np
import _pickle as pickle
import time


class Kmeans:
    def __init__(self, num_clusters, feature_matrix):
        self.k = num_clusters
        # get the shape of the matrix [# of docs, # of features (max vocab size)]
        dimensionality = feature_matrix.shape
        # calculate the min and max value for each feature, to be used in centroid initialization (earlier runs showed
        # min to always be 0)
        ranges = np.zeros((dimensionality[1]-1, 2))
        for dim in range(dimensionality[1]-1):
            ranges[dim, 0] = 0
            ranges[dim, 1] = np.max(feature_matrix[:, dim])
        # create k centroids, matching # of features stored for each doc, ignorining the UNK token b/c of its effect
        # on skewing the ranges and resulting centroids
        self.centroids = np.zeros((self.k, dimensionality[1]-1))
        for i in range(self.k):
            for dim in range(dimensionality[1]-1):
                # random, uniform initialization
                self.centroids[i, dim] = np.random.uniform(ranges[dim, 0], ranges[dim, 1], 1)
        # initialize class variables in the __init__ function
        self.cluster = []

    def convergeClusters(self, features):
        start_time = time.time()
        dimensions = features.shape

        # track episodes till converge and prevent too long of running times
        episodes = 0
        not_converged = True
        while episodes < 10000 and not_converged:
            episodes += 1

            # calculate the distance from each feature to the corresponding feature in every centroid
            distances = np.zeros((dimensions[0],self.k))
            for f_index, f_val in enumerate(features):
                for c_index, c_val in enumerate(self.centroids):
                    distances[f_index, c_index] = euclidean(f_val[:-1], c_val)
            # assign each point to the minimum distance (closest) cluster
            self.cluster = np.argmin(distances, axis = 1)

            # gather all the points in the cluster.  if no points, do nothing for now, else compute the mean
            # of that feature for that cluster
            updated_centroids = np.zeros_like(self.centroids)
            for centroid in range(self.k):
                temp = features[self.cluster == centroid]
                if len(temp) != 0:
                    for dim in range(dimensions[1]-1):
                        updated_centroids[centroid, dim] = np.mean(temp[:,dim])

            # checks if the distance between centroids is smaller than the system eps
            # (The smallest representable positive number such that 1.0 + eps != 1.0.) 2.22e-16 on my local machine
            if np.linalg.norm(updated_centroids - self.centroids) < np.finfo(float).eps:
                print("converged in {} episodes".format(episodes))
                not_converged = False

            self.centroids = updated_centroids
        print("converged in {} episodes".format(episodes))
        elapsed_time = time.time() - start_time
        print(f'cluster convergence took {elapsed_time} seconds')


    def evaluate(self, test_features):
        # essentially the cluster identifying logic of the converge function, just not in a loop and returns the
        # labels.
        dimensions = test_features.shape

        distances = np.zeros((dimensions[0], self.k))
        for f_index, f_val in enumerate(test_features):
            start_time = time.time()
            for c_index, c_val in enumerate(self.centroids):
                distances[f_index, c_index] = euclidean(f_val[:-1], c_val)
                elapsed_time = time.time() - start_time
                print(f'one classification took {elapsed_time} seconds')

        # assign each point to the minimum distance (closest) cluster
        test_labels = np.argmin(distances, axis=1)
        return test_labels

if __name__ == '__main__':
    # params = [[[0, 1], [0, 1]],
    #           [[5, 1], [5, 1]],
    #           [[-2, 5], [2, 5]],
    #           [[2, 1], [2, 1]],
    #           [[-5, 1], [-5, 1]]]
    #
    # n = 300
    # dims = len(params[0])
    #
    # feature_matrix = []
    # y = []
    # for ix, i in enumerate(params):
    #     inst = np.random.randn(n, dims)
    #     for dim in range(dims):
    #         inst[:, dim] = params[ix][dim][0] + params[ix][dim][1] * inst[:, dim]
    #         label = ix + np.zeros(n)
    #
    #     if len(feature_matrix) == 0:
    #         feature_matrix = inst
    #     else:
    #         feature_matrix = np.append(feature_matrix, inst, axis=0)
    #     if len(y) == 0:
    #         y = label
    #     else:
    #         y = np.append(y, label)
    #
    # num_clusters = len(params)
    #
    # print(y.shape)
    # print(feature_matrix.shape)
    file = open('features.p', 'rb')
    feats = pickle.load(file)
    kmeans = Kmeans(20, feats)
    kmeans.convergeClusters(feats)
    kmeans.evaluate(feats)