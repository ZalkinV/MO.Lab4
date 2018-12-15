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

	x_train, x_test, y_train, y_test = train_test_split(data_raw.drop("SalePrice", axis=1),
														data_raw["SalePrice"],
														test_size=0.2,
														random_state=0)
	feautures_names = ["LotArea"]#["Neighborhood", "LotArea", "YearBuilt"]
	data_labels_train, data_labels_test = y_train[:], y_test[:]
	del y_train, y_test 
	data_features_train = get_features(x_train, names=feautures_names)
	data_features_test = get_features(x_test, names=feautures_names)
	del x_train, x_test

	hypothesis = linear_model.LinearRegression()
	hypothesis.fit(data_features_train, data_labels_train)
	print("{w0} + {w1}*x".format(w0=hypothesis.intercept_, w1=hypothesis.coef_[0]))

	show_graph(data_features_train["LotArea"], data_labels_train)
	pass

def get_features(data_frame, min_unique= None, names= None):
	features_names = []

	if names == None and min_unique != None:
		for col_name in data_frame:
			unique_count = data_frame[col_name].nunique()
			if unique_count >= min_unique:
				features_names.append(col_name)
	elif names != None and min_unique == None:
		features_names = names

	return data_frame[features_names]

def calculate_error(pred, actual, type='rmsle'):
	if type == 'rmsle':
		return np.sqrt(np.mean(np.power(np.log1p(pred) - np.log1p(actual), 2))) #log1p(x) == log(x + 1)
	elif type == 'rmse':
		return np.mean(np.square(np.subtract(pred, actual)))/2

def show_graph(feature, label):
	plt.xlabel(feature.name)
	plt.ylabel(label.name)
	plt.scatter(feature, label, marker='.', color='red', s=10)
	plt.show()
	pass

main()
