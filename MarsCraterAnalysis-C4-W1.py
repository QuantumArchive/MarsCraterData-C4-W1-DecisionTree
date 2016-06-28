# coding: utf-8

"""
Created on Tue June 28 11:42:08 2016

@author: Chris
"""
import pandas
import numpy
import matplotlib.pyplot as plt
import matplotlib.pylab as pll
import sklearn.metrics
import scipy.stats
import pydotplus
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from io import BytesIO
from IPython.display import Image


#from IPython.display import display
#get_ipython().magic(u'matplotlib inline')

#bug fix for display formats to avoid run time errors
pandas.set_option('display.float_format', lambda x:'%f'%x)

#Set Pandas to show all columns in DataFrame
pandas.set_option('display.max_columns', None)
#Set Pandas to show all rows in DataFrame
pandas.set_option('display.max_rows', None)

#data here will act as the data frame containing the Mars crater data
data = pandas.read_csv('D:\\Coursera\\marscrater_pds.csv', low_memory=False)

#convert the latitude and diameter columns to numeric and ejecta morphology is categorical
data['LATITUDE_CIRCLE_IMAGE'] = pandas.to_numeric(data['LATITUDE_CIRCLE_IMAGE'])
data['DIAM_CIRCLE_IMAGE'] = pandas.to_numeric(data['DIAM_CIRCLE_IMAGE'])
data['MORPHOLOGY_EJECTA_1'] = data['MORPHOLOGY_EJECTA_1'].astype('category')

#Any crater with no designated morphology will be replaced with NaN
data['MORPHOLOGY_EJECTA_1'] = data['MORPHOLOGY_EJECTA_1'].replace(' ',numpy.NaN)

#Remove any data with NaN values
data2 = data.dropna()

#Only keep craters with morphology of interest
morphofinterest = ['Rd','SLEPS','SLERS']
data2 = data2.loc[data['MORPHOLOGY_EJECTA_1'].isin(morphofinterest)]
data2.describe()

#The original inquiry led to a hypothesis that the latitude of a crater (explanatory) shows association with its diameter (response).
#Because we are interested in a categorical response variable with two levels, we split the crater size into two groups, small
#and large. From an earlier analysis we found that the distribution of craters with diameters smaller than 1.53 km were 
#statistically significant when compared to groups of larger craters when looking at the their distribution by latitude.
#Because the distribution of craters across latitude tends to follow a more normal distribution, we also decide to split
#craters based on whether they are closer to the equator or closer to one of the poles. Also, to because our decision tree
#won't accept strings, we'll recode the Ejecta morphology of interest.
    
def cratersize(x):
    if x <= 1.53:
        return 'SMALL'
    else:
        return 'LARGE'
    
def georegion(x):
    if x <= -30:
        return 0
    elif x <= 30:
        return 1
    else:
        return 0
    
def morph(x):
    if x == 'Rd':
        return 0
    elif x == 'SLEPS':
        return 1
    elif x == 'SLERS':
        return 2
    
data2['CRATER_BIN'] = data2['DIAM_CIRCLE_IMAGE'].apply(lambda x: cratersize(x))
data2['LATITUDE_BIN'] = data2['LATITUDE_CIRCLE_IMAGE'].apply(lambda x: georegion(x))
data2['MORPH_BIN'] = data2['MORPHOLOGY_EJECTA_1'].apply(lambda x: morph(x))
data2['CRATER_BIN'] = data2['CRATER_BIN'].astype('category')
data2['LATITUDE_BIN'] = data2['LATITUDE_BIN'].astype('category')
data2['MORPH_BIN'] = data2['MORPH_BIN'].astype('category')

#Now we'll set up our predictors and target
predictors = data2[['DEPTH_RIMFLOOR_TOPOG','MORPH_BIN','NUMBER_LAYERS','LATITUDE_BIN']]
target = data2['CRATER_BIN']

#Looking at the datatypes
data2.dtypes

#We now create our training set and test set
pred_train, pred_test, tar_train, tar_test  =   train_test_split(predictors, target, test_size=.4)

pred_train.shape
pred_test.shape
tar_train.shape
tar_test.shape

#Build model on training data
classifier=DecisionTreeClassifier() #initialize Decision Tree Classifier
classifier=classifier.fit(pred_train,tar_train) #fit function, pass predictors and targets from training data to

#make predictions based on the test data
predictions=classifier.predict(pred_test)
#remember confusion matrix here is to determine the degree of misclassification
print(sklearn.metrics.confusion_matrix(tar_test,predictions))
print(sklearn.metrics.accuracy_score(tar_test, predictions))

#Create image
out = BytesIO()
tree.export_graphviz(classifier, out_file=out)
graph=pydotplus.graph_from_dot_data(out.getvalue())
Image(graph.create_png())
