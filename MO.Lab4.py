import numpy
import pandas

def Main():
	dataPath = "../train.csv"
	data = pandas.read_csv(dataPath, delimiter=',')
	print(data)
	pass

Main()
