import numpy as np
import pandas
import matplotlib.pyplot as plt

def main():
	pandas.set_option('display.max_columns', 81) # Для отображения 81 столбца

	data_path = "../train.csv"
	data_raw = pandas.read_csv(data_path, sep=',', index_col="Id")

	#print(dataFrameRaw.info()) # Получение инфорации о столбцах
	#print(dataFrameRaw.describe(include="object")) # Получение информации о всех строках с типом object, чтобы понять уникальных значений какого параметра больше, чтобы использовать это параметр как признак
	#print(dataFrameRaw["LotConfig"].value_counts()) # Получение количества уникальных значений для указанного столбца

	feautures_names = ["LotArea"]#["Neighborhood", "LotArea", "YearBuilt"]
	data_train, data_test = get_data_parts(data_raw.sample(frac=1), 0.6, 0.4) # frac - доля от всего набора данных в случайном порядке
	data_labels_train = data_train["SalePrice"]
	data_labels_test = data_test["SalePrice"]
	data_features_train = get_features(data_train.drop("SalePrice", axis=1), names=feautures_names)
	data_features_test = get_features(data_test.drop("SalePrice", axis=1), names=feautures_names)
		
	show_graph(data_features_train["LotArea"], data_labels_train)

	#print(dataFrame)
	pass

def get_data_parts(data, *args):
	parts = []
	prev_last_row = 0
	for frac in args:
		curr_last_row = int(prev_last_row + len(data) * frac)
		parts.append(data[prev_last_row:curr_last_row])
		prev_last_row = curr_last_row
	return parts

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
