import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics

def main():
	pandas.set_option('display.max_columns', 81) # Для отображения 81 столбца

	data_path = "../train.csv"
	data_raw = pandas.read_csv(data_path, sep=',', index_col="Id")

	data_label_column = data_raw["SalePrice"]
	data_ready = preprocess_data(data_raw.drop("SalePrice", axis=1), exc_types=["object"])
	x_train, x_test, y_train, y_test = train_test_split(data_ready,
														data_label_column,
														test_size=0.2,
														random_state=0)
	data_labels_train, data_labels_test = y_train[:], y_test[:]
	del y_train, y_test
	data_features_train, data_features_test = x_train[:], x_test[:]
	del x_train, x_test

	completing_lab_part1(data_features_train, data_features_test, data_labels_train, data_labels_test)

	plt.show()
	pass

def completing_lab_part1(X_train, X_test, Y_train, Y_test):
	hypothesis = linear_model.LinearRegression()
	print("\nHypothesis:", hypothesis)
	hypothesis.fit(X_train, Y_train)
	prediction = hypothesis.predict(X_test)
	print_error(prediction, Y_test)

	print("w0 =", hypothesis.intercept_)
	features_coefficients = pandas.DataFrame(X_test.columns, columns=["Feature"])
	features_coefficients["Weight"] = hypothesis.coef_
	print(features_coefficients)

	features_coefficients.plot(kind="bar")
	pass

def calculate_error(pred, actual, type='rmsle'):
	if type == 'rmsle':
		return np.sqrt(np.mean(np.power(np.log1p(pred) - np.log1p(actual), 2))) #log1p(x) == log(x + 1)
	elif type == 'rmse':
		return np.mean(np.square(np.subtract(pred, actual)))/2
	pass

def print_error(predicted, actual):
	predicted = predicted.astype(np.int64)
	print("RMSLE = {error}".format(error=calculate_error(predicted, actual)))
	print("Accuracy =", metrics.accuracy_score(actual, predicted))

	predicted_actual = pandas.DataFrame({"Predic" : predicted[:8], "Actual" : actual[:8]})
	predicted_actual["Differ"] = predicted_actual["Predic"] - predicted_actual["Actual"]
	print(predicted_actual, end="\n\n\n")
	pass

def preprocess_data(data_raw, min_uniq=0, exc_types=None):
	data_processing = data_raw.copy(deep=True)
	if exc_types is not None:
		data_processing = data_raw.dropna(axis=1).select_dtypes(exclude=exc_types)

	uniq_column_names = (data_processing.nunique() >= min_uniq).index
	data_processing = data_processing[uniq_column_names]

	data_processing = pandas.DataFrame(preprocessing.minmax_scale(data_processing), columns=data_processing.columns)
	return data_processing

def plot_graph_data(feature, label):
	plt.xlabel(feature.name)
	plt.ylabel(label.name)
	plt.scatter(feature, label, marker='.', color='red', s=10)
	pass

main()
