import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split

def main():
	pandas.set_option('display.max_columns', 81) # Для отображения 81 столбца

	data_path = "../train.csv"
	data_raw = pandas.read_csv(data_path, sep=',', index_col="Id")

	#print(dataFrameRaw.info()) # Получение инфорации о столбцах
	#print(dataFrameRaw.describe(include="object")) # Получение информации о всех строках с типом object, чтобы понять уникальных значений какого параметра больше, чтобы использовать это параметр как признак
	#print(dataFrameRaw["LotConfig"].value_counts()) # Получение количества уникальных значений для указанного столбца

	uniqueuness = 50
	data_ready = data_raw.dropna(axis=1).select_dtypes(exclude=["bool", "object"])
	uniqueuness_column = (data_ready.nunique() >= uniqueuness).index
	data_ready = data_ready[uniqueuness_column]
	x_train, x_test, y_train, y_test = train_test_split(data_ready.drop("SalePrice", axis=1),
														data_ready["SalePrice"],
														test_size=0.2,
														random_state=0)

	data_labels_train, data_labels_test = y_train[:], y_test[:]
	del y_train, y_test
	data_features_train, data_features_test = x_train[:], x_test[:]
	del x_train, x_test

	hypothesis = linear_model.LinearRegression()
	hypothesis.fit(data_features_train, data_labels_train)
	prediction = hypothesis.predict(data_features_test)
	print("RMSLE = {error}".format(error=calculate_error(prediction, data_labels_test)))

	print("w0 =", hypothesis.intercept_)
	features_coefficients = pandas.DataFrame(data_features_train.columns, columns=["Feature"])
	features_coefficients["Weight"] = hypothesis.coef_
	print(features_coefficients)

	plot_graph_data(data_features_train["LotArea"], data_labels_train)
	plot_graph_hypothesis(hypothesis, data_features_train)
	plot_bargraph_weights(features_coefficients)
	plt.show()
	pass

def calculate_error(pred, actual, type='rmsle'):
	if type == 'rmsle':
		return np.sqrt(np.mean(np.power(np.log1p(pred) - np.log1p(actual), 2))) #log1p(x) == log(x + 1)
	elif type == 'rmse':
		return np.mean(np.square(np.subtract(pred, actual)))/2

def plot_graph_data(feature, label):
	plt.xlabel(feature.name)
	plt.ylabel(label.name)
	plt.scatter(feature, label, marker='.', color='red', s=10)
	pass

def plot_graph_hypothesis(hypo, feature):
	y = [x * hypo.coef_[0] + hypo.intercept_ for x in feature.values]
	plt.plot(feature, y)
	pass

def plot_bargraph_weights(features_weights):
	plt.figure(2)
	plt.bar(features_weights["Feature"], features_weights["Weight"])
	pass

main()
