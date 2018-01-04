
import numpy as np
import matplotlib.pyplot as plt
import statsmodel.api as sm
import scipy.stats as st
import argparse
import csv

parser = argparse.ArgumentParser(description='Processes the arguments on execution')
parser.add_argument("--target_dir", default='~/data/', type=str, help="Directory to store the dataset.")
parser.add_argument("--data_path", type=str, help="Path to the TEDLIUM_release tar if downloaded (Optional).")



def stl(data, freq):
	
	if freq == None:
		freq = int(0.2 * len(data))

	decomp= sm.tsa.seasonal_decompose(data, freq = freq, model = 'additive')
	return decomp



def test_stat_zscore(data):

	z_scores = abs(st.zscore(data, ddof = 1))
	max_idx = np.argmax(z_scores)
	return max_idx, z_scores[max_idx]


def grubbs_stat_limit(data, alpha = 0.6):

	size = len(data)
	tdist= st.t.ppf(1-alpha / (2*size), size -2)
	nm= (size-1)* tdist
	dn= np.sqrt(size** 2- size* 2 + size * tdist ** 2)
	return nm/dn


def sh_esd(data, seasonality= None , hybrid = False, max_anomalies = 10, alpha = 0.05):

	data = np.array(data)
	decomposition = stl(data, seasonality)

	if hybrid:
		med_abs_dev = np.median(np.abs(data - np.median(data)))
		trend_comp = data - decomposition.seasonal - med_abs_dev

	else:
		trend_comp = data - decomposition.seasonal - np.median(data)

	outliers = esd(trend_comp, max_anomalies = max_anomalies, alpha = alpha)



def esd(data, max_anomalies = 10, alpha = 0.05):


	ts = np.copy(np.array(data))
	test_stats = []
	total_anomalies= -1
	for itr in range(max_anomalies):
		test_index, test_value = test_stat_zscore(data)
		critical_value = grubbs_stat_limit(data, alpha)

		if test_value > critical_value:
			total_anomalies = itr
		test_stats.append(test_index)
		data = np.delete(data, test_index)

	anomaly_indices = test_stats[:total_anomalies+1]
	return anomaly_indices


def main():
	data_path = args.data_path
	outlier_path = args.target_dir
	try:
		with open(data_path) as f:
			csv.reader(f, delimiter = ";")

			data = []
			nb_of_values = 0

			for line in data:
				try:
					data.append(float(line[2]))
					nb_of_values +=1

				except ValueError:
					pass
	anomalies = sh_esd(data)
	


if __name__ == '__main__':
	main()
