# import tensorflow as tf
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 

def import_hypothesis(file_location):
	df=pd.read_csv(file_location)
	
	data_size=df.shape[0]
	number_of_features = df.shape[1]-1
	data_size = df.shape[0]
	y=np.array(list(df.ix[:,1]))
	x = np.ones((data_size,number_of_features))

	for i in range(3,number_of_features+2):
		x[:,i-2]=df[df.columns[i-1]]
	return x,y,data_size,number_of_features

x_train,y_train,data_size_train,number_of_features = import_hypothesis("30_train_features.csv")
x_test,y_test,data_size_test,number_of_features = import_hypothesis("30_test_features.csv")

