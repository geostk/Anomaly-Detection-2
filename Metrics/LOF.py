# Local Outlier Factor Implementation
# For more information visit https://en.wikipedia.org/wiki/Local_outlier_factor


import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbours
from sklearn.neighbors import LocalOutlierFactor
import statsmodel.api as sm
import scipy.stats as st


#Finding distance
def knn(data, k=20):
	knn_clf = NearestNeighbours(n_neighbors =k)
	knn_clf.fit(data)
	dists, indices = knn_clf.kneighbours(data)

	return dists, indices

#Reachability Distance
def reachability_distance(data, min_points, knn_dist):
	distance_min_points, indices_min_point = knn(data, min_points)
	distance_mat = np.zeros((len(distance_min_points), 3))
	distance_mat[:,0] = np.amax(distance_min_points)
	distance_mat[:,1] = np.amax(distance_min_points)
	distance_mat[:,2] = np.amax(distance_min_points)

	return distance_mat, indices_min_point

# Local Reachability Distance
def LRD(min_points, knn_dist):
	lrd = min_points/np.sum(knn_dist, axis =1)
	return lrd

# Local Outlier factor
def lof(lrd, min_points, data):
	lof_anom = []
	for item in data:
		temp_lrd = np.divide(lrd[item[1:]], lrd[item[0]])
		lof_anom.append(temp_lrd.sum()/min_points)
	return lof_anom


# For directly calculating LOF using SKlearn Library
def lof_direct(data, k= 20, outlier_ratio = 0.15):

	clf = LocalOutlierFactor(n_neighbors= 20)
	pred = clf.fit_predict(data)
	n_outliers = int(len(data)*outlier_ratio)
	outlier_pred = pred[-1: n_outliers]


	return outlier_pred


