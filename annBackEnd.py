import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import os
dataframe = pd.read_csv("D:/Projects/FY Project/masterFile.txt", header=None)
dataset=dataframe.values
dataset=dataset[0:,0:15]
trainingInput=dataset[:,0:12]
trainingOutputAngle=dataset[:,14:15]
trainingOutputSpeed=dataset[:,13:14]
prevAngle=0
prevSpeed=2
# define base mode
def baseline_model_angle():
	# create model
	model = Sequential()
	model.add(Dense(18, input_dim=12, init='normal', activation='tanh'))
	model.add(Dense(7,init='normal',activation='relu'))
	model.add(Dense(5,init='normal',activation='relu'))
	model.add(Dense(3,init='normal',activation='relu'))
	model.add(Dense(1, init='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
# define base mode
def baseline_model_speed():
	# create model
	model = Sequential()
	model.add(Dense(18, input_dim=12, init='normal', activation='relu'))
	model.add(Dense(1, init='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
seed=7
np.random.seed(seed)
estimators_angle = []
estimators_angle.append(('standardize', StandardScaler()))
estimators_angle.append(('mlp', KerasRegressor(build_fn=baseline_model_angle, nb_epoch=50, batch_size=5, verbose=0)))
pipeline_angle = Pipeline(estimators_angle)
kfold = KFold(n_splits=10, random_state=seed)
pipeline_angle.fit(trainingInput,trainingOutputAngle)
estimators_speed = []
estimators_speed.append(('standardize', StandardScaler()))
estimators_speed.append(('mlp', KerasRegressor(build_fn=baseline_model_speed, nb_epoch=50, batch_size=5, verbose=0)))
pipeline_speed = Pipeline(estimators_speed)
pipeline_speed.fit(trainingInput,trainingOutputSpeed)
prevAngle=-180
prevSpeed=-100	
endless=1
counter=1
with open("forUnity.txt", "r+") as f:
	 	f.seek(0)
	 	f.write(str(counter))
	 	f.write(',')
	 	f.write(str(2))
	 	f.write(',')
	 	f.write(str(0))
	 	counter+=1
while endless==1:
	 #if os.stat("forANN.txt").st_size == 0:
	 #	print('file empty')
	 #	continue
	 #print('file has values')
	 try:
	 	data=pd.read_csv('forANN.txt',header=None)
	 except:
	 	print('Empty') 
	 	continue
	 print('Not Empty')

	 #print(data)
	 #print(data)
	 inputData=data.values
	# print(prev)
	 #print(inputData)
	 #current=inputData[0][0]
	 #if current==prev:
	 	#continue	 	
	 #prev=inputData
	 #print(inputData)
	 #print(inputData)
	 outputAngle=pipeline_angle.predict(inputData)
	 outputSpeed=pipeline_speed.predict(inputData)
	 if(outputAngle == prevAngle and outputSpeed == prevSpeed):
	 	continue
	 try:
	 	with open("forUnity.txt", "r+") as f:
	 	 f.seek(0)
	 	 f.write(str(counter))
	 	 f.write(',')
	 	 f.write(str(outputSpeed))
	 	 f.write(',')
	 	 f.write(str(outputAngle))
	 except:
	 	continue
	 counter+=1
	 prevAngle=outputAngle
	 prevSpeed=outputSpeed
	 



	 



     	