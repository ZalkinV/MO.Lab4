import numpy
import pandas
import matplotlib.pyplot as plt

def Main():
	pandas.set_option('display.max_columns', 81) # Для отображения 81 столбца

	dataPath = "../train.csv"
	dataFrameRaw = pandas.read_csv(dataPath, sep=',', index_col="Id")

	#print(dataFrameRaw.info()) # Получение инфорации о столбцах
	#print(dataFrameRaw.describe(include="object")) # Получение информации о всех строках с типом object, чтобы понять уникальных значений какого параметра больше, чтобы использовать это параметр как признак
	#print(dataFrameRaw["LotConfig"].value_counts()) # Получение количества уникальных значений для указанного столбца

	feauturesNames = ["Neighborhood", "LotArea", "YearBuilt"]
	dataFrame = GetFeatures(dataFrameRaw.drop("SalePrice", axis=1), names=feauturesNames)
	dataFrameShuff = dataFrame.sample(frac=1) # frac - доля от всех набора данных в случайном порядке
	dataTrain, dataCross, dataTest = GetDataParts(dataFrameShuff, 0.5, 0.2, 0.3)

	print(dataFrame)
	pass

def GetDataParts(data, *args):
	parts = []
	prevLastRow = 0
	for frac in args:
		currLastRow = int(prevLastRow + len(data) * frac)
		parts.append(data[prevLastRow:currLastRow])
		prevLastRow = currLastRow
	return parts

def GetFeatures(dataFrame, minUnique= None, names= None):
	featuresNames = []

	if names == None and minUnique != None:
		for columnName in dataFrame:
			uniqueCount = dataFrame[columnName].nunique()
			if uniqueCount >= minUnique:
				featuresNames.append(columnName)
	elif names != None and minUnique == None:
		featuresNames = names

	return dataFrame[featuresNames]

def ShowGraph(dataFrame):
	plt.plot(dataFrame.index, dataFrame["SalePrice"])
	plt.show()
	pass
Main()
