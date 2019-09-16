import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import tensorflow as tf

from sklearn.svm import LinearSVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error as msle

import keras
import keras.layers as layers

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor

from xgboost import XGBRegressor

from pyramid.arima import auto_arima


dftr=pd.read_csv('Train.csv',parse_dates=['date_time'],index_col='date_time')

dfte=pd.read_csv('Test.csv',parse_dates=['date_time'],index_col='date_time')

dfss=pd.read_csv('sample_submission.csv')

dftr.wind_direction=(3.14/180)*(dftr.wind_direction)  #wind_direction cardinal degrees to radians

print(dftr.shape)
date=dftr.index

#Feature Engineering

ms=np.array([],dtype=str)	#boolean features
me=np.array([],dtype=str)
ys=np.array([],dtype=str)
ye=np.array([],dtype=str)
qs=np.array([],dtype=str)
qe=np.array([],dtype=str)

for i in range(33750):
  aaa=pd.Timestamp(date[i])
  me=np.append(me,aaa.is_month_end)
  ms=np.append(ms,aaa.is_month_start)
  ye=np.append(ye,aaa.is_year_end)
  ys=np.append(ys,aaa.is_year_start)
  qe=np.append(qe,aaa.is_quarter_end)
  qs=np.append(qs,aaa.is_quarter_start)
  

time=np.array([],dtype=int)		# Features extracted from date_time
woy=np.array([],dtype=int)
dow=np.array([],dtype=int)
month=np.array([],dtype=str)
week=np.array([],dtype=float)
days=np.array([],dtype=str)
year=np.array([],dtype=float)

for i in range(33750):
  aaa=pd.Timestamp(date[i])
  days=np.append(days,aaa.day_name())
  month=np.append(month,aaa.month_name())
  number_dec = str((aaa.year/100)-int(aaa.year/100)).split('.')[1]
  year=np.append(year,int(number_dec[0:2])/10)
  time=np.append(time,aaa.hour) 
  dow=np.append(dow,aaa.dayofweek)  
  woy=np.append(woy,aaa.weekofyear)  
  week=np.append(week,((aaa.day-1) // 7 + 1))



tr_x=dftr.iloc[:,0:13]

#One Hot Encoding
tr_x=pd.get_dummies(tr_x)    

tr_y=dftr.iloc[:,13]

xtr,xte,ytr,yte=train_test_split(tr_x,tr_y,test_size=0.15)

def scr(yact,ypre):
  return (100-mean_squared_error(yact,ypre))


#Models Tried
model1=LinearRegression(n_jobs=-1,normalize=True)

model1.fit(xtr,ytr)

pr=model1.predict(xtr)


100-msle(np.absolute(ytr),np.absolute(pr))

scr(ytr,pr)

model2=LinearSVR()

model2.fit(xtr,ytr)

pr=model2.predict(xtr)

scr(ytr,pr)



model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(1, activation='linear')
])

model.compile(optimizer='adam',
              loss='mse',
              metrics=['mse'])

model.evaluate(xte,yte)


#Shape change to 3 dimensional for lstm 
xtrcnn=np.reshape(xtr.values,(xtr.values.shape[0],1,xtr.values.shape[1]))
ytrcnn=np.reshape(ytr.shape[0],1,ytr.shape[1])

model = keras.Sequential([
    layers.LSTM(64,input_shape=(xtrcnn.shape[1],xtrcnn.shape[2])),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(32, activation=tf.nn.relu),
    layers.Dropout(0.10),
    layers.Dense(32, activation=tf.nn.relu),
    layers.Dense(32, activation=tf.nn.relu),
    layers.Dense(12, activation=tf.nn.relu),
    
    layers.Dense(12, activation=tf.nn.relu),
    layers.Dense(1)
  ])

#optimizer = tf.keras.optimizers.RMSprop(0.001)

model.compile(loss='mean_squared_error',
                optimizer='adam',
                metrics=['mean_absolute_error', 'mean_squared_error'])

model.fit(xtrcnn, ytr.values, epochs=50)

model.evaluate(xte,yte)




modelGB = ensemble.GradientBoostingRegressor(
max_features=75,
max_depth=20,
n_estimators=1000)

model= ensemble.AdaBoostRegressor(
        n_estimators=100,
    base_estimator=modelGB
)

# ExtraTreesRegressor (one of final model used)

modelETR = ensemble.ExtraTreesRegressor(
    
      max_depth=24,
      random_state=42,
#     min_samples_split=7,
#     min_samples_leaf=7,
      n_estimators=500,
      max_features=60,
#     oob_score=True,
#     bootstrap=True,
      n_jobs=-1
)

#Model fitted for supervised regression

modelETR.fit(tr_x,tr_y)   

etp=modelETR.predict(xte)

100-msle(np.absolute(yte),np.absolute(model.predict(xte)))

scr(yte,etp)


# train set preparation for time series forecasting:

#Dropping duplicates
tr_x=tr_x[~tr_x.index.duplicated(keep='first')] 
tr_y=tr_y[~tr_y.index.duplicated(keep='first')]


#Interpolating missing data
tr_x=tr_x.resample('H').interpolate(method='linear')
tr_y=tr_y.resample('H').interpolate(method='linear')


#Use of auto_arima to get best hyperparameters (order for SARIMAX) 
modelar = auto_arima(tr_y, trace=True, error_action='ignore', suppress_warnings=True)
# modelar.fit()

modelar.predict(10000)[-1000]

#order=(p,d,q) (5,0,5) detected from auto_arima above

#basic ARIMA Model
tsmodel0=sm.tsa.ARIMA(tr_y,exog=tr_x.values,order=(5,0,5),freq='H')


#SARIMAX our final model
#Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors model

tsmodel=sm.tsa.statespace.SARIMAX(tr_y,exog=tr_x.values,order=(5,0,5),seasonal_order=(0,0,0,0),freq='H')

ts=tsmodel.fit()

ts.predict(tr_y.index.values[0],tr_y.index.values[0],exog=tr_x.iloc[0].values)



#Other Tried models
########################
reg1 = GradientBoostingRegressor(random_state=1, n_estimators=300,max_depth=23)
reg2 = RandomForestRegressor(random_state=1, n_estimators=300,max_depth=23)
#reg3 = LinearRegression()
reg3=ensemble.ExtraTreesRegressor(
    max_depth=23,
    random_state=42,
    n_estimators=300,
    bootstrap=True,
    n_jobs=-1)
ereg = VotingRegressor(estimators=[('gb', reg1), ('rf', reg2), ('er', reg3)])
ereg = ereg.fit(tr_x, tr_y)

scr(yte,ereg.predict(xte))


model = XGBRegressor(
     n_estimators=500)
model.fit(tr_x,tr_y)
xgp=model.predict(xte)
scr(yte,xgp)
########################



#Preparation of Test set to prepare final submission file
te=dfte

datete=dfte.index

te.wind_direction=(3.14/180)*(te.wind_direction)


mste=np.array([],dtype=str)
mete=np.array([],dtype=str)
yste=np.array([],dtype=str)
yete=np.array([],dtype=str)
qste=np.array([],dtype=str)
qete=np.array([],dtype=str)

for i in range(14454):
  aaa=pd.Timestamp(datete[i])
  mete=np.append(mete,aaa.is_month_end)
  mste=np.append(mste,aaa.is_month_start)
  yete=np.append(yete,aaa.is_year_end)
  yste=np.append(yste,aaa.is_year_start)
  qete=np.append(qete,aaa.is_quarter_end)
  qste=np.append(qste,aaa.is_quarter_start)

timete=np.array([],dtype=int)
dowte=np.array([],dtype=int)
woyte=np.array([],dtype=int)
monthte=np.array([],dtype=str)
weekte=np.array([],dtype=float)
dayste=np.array([],dtype=str)
yearte=np.array([],dtype=float)

for i in range(14454):
  aaa=pd.Timestamp(datete[i])
  dayste=np.append(dayste,aaa.day_name())
  monthte=np.append(monthte,aaa.month_name())
  number_dec = str((aaa.year/100)-int(aaa.year/100)).split('.')[1]
  yearte=np.append(yearte,int(number_dec[0:2])/10)
  timete=np.append(timete,aaa.hour)  
  dowte=np.append(dowte,aaa.dayofweek)  
  woyte=np.append(woyte,aaa.weekofyear)  
  weekte=np.append(weekte,((aaa.day-1) // 7 + 1))


te['weekday']=dayste
te['year']=yearte
te['month']=monthte
te['week']=weekte
te['hour']=timete
te['ms']=mste
te['me']=mete
te['ys']=yste
te['ye']=yete
te['woy']=woyte
te['dow']=dowte
te['qs']=qste
te['qe']=qete

#dummy variables
te_x=pd.get_dummies(te)



#matching columns of test set with the final train set
missing_cols = set( tr_x.columns ) - set( te_x.columns )

for c in missing_cols:
       te_x[c] = 0

te_x=te_x[tr_x.columns]

te_x.columns==tr_x.columns

#test to check if the columns match
assert( set( tr_x.columns ) - set( te_x.columns ) == set())


#ExtraTreesRegressor prediction 
te_pr=modelETR.predict(te_x)


#preparing test data for time series forecasting with SARIMAX

#dropping duplicates
tte=te_x[~te_x.index.duplicated(keep='first')]

#interpolating missing values
tte=tte.resample('H').interpolate(method='linear')


#SARIMAX Predictions
tspre=np.absolute(ts.predict(tte.index.values[0],tte.index.values[12023],exog=tte.values))

#ETRegressor sub
sub1=pd.DataFrame({'date_time':dfte.index.values,'traffic_volume':te_pr.ravel()})

#SARIMAX sub 
sub2=pd.DataFrame({'date_time':dfte.index.values,'traffic_volume':tspre[dfte.index].values.ravel()})

#Final sub
final_prediction= 0.6 * te_pr.ravel() + 0.4 * tspre[dfte.index].values.ravel()

#Genrating submission file
sub=pd.DataFrame({'date_time':dfte.index.values,'traffic_volume':final_prediction})

sub.to_csv('submission.csv',index=False)
