#!/usr/bin/env python
# coding: utf-8

# In[49]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as mlt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# In[51]:


prediction= pd.read_csv('House_price.csv')
prediction


# In[74]:


#To check for duplicate; No Duplicates
prediction.duplicated().sum()


# In[76]:


#To check for missing values; No no missing values
prediction['SalePrice'].isna().sum()


# In[95]:


#Outlier analysis usin Interquartile Range
Q1= prediction['SalePrice'].quantile(0.25)
Q2= prediction['SalePrice'].quantile(0.5)
Q3= prediction['SalePrice'].quantile(0.75)
IQR= Q3-Q1

upper_boundary= Q3 + (1.5*IQR)
lower_boundary= Q1 - (1.5*IQR)
print(upper_boundary , lower_boundary)


# In[90]:


# Outlier analysis using Standard Deviation
upper_limit= prediction['SalePrice'].mean() + (2*prediction['SalePrice'].std())
lower_limit=  prediction['SalePrice'].mean() - (2*prediction['SalePrice'].std())
print( upper_limit , lower_limit)


# In[96]:


outliers = prediction[(prediction['SalePrice'] < lower_boundary) & (prediction['SalePrice'] > upper_boundary)]
outliers


# In[77]:


#Convertion of the categorical data 'Location' to numerical data
label_encoders={}
for column in ["Location"]:
    label_encoders[column] = LabelEncoder()
    prediction[column] = label_encoders[column].fit_transform(prediction[column])


# In[78]:


#Preparing the features for matrix and target vector

X= prediction.drop('SalePrice', axis=1)
y= prediction['SalePrice']
print("For X, the values are: \n", X,"\n\n","For y, the values are: \n",y)


# In[116]:


#Splitting the dataset into Training and Testing set

X_train, X_test, y_train, y_test =train_test_split(X,y, test_size=0.2, random_state=42)

#Transformation of our dataset
scaler= StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[117]:


#Training the model
model = GradientBoostingRegressor()
model.fit(X_train, y_train)


# In[118]:


#Make Prediction on the test set
y_pred= model.predict(X_test)


# In[119]:


#Evaluating the model using Mean Square Error, Mean Absolute Error, and R2 method
mse = mean_squared_error(y_test, y_pred) 
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score (y_test, y_pred)

print("The Mean Square Error = ", mse, "\nThe Mean Absolute Error = ", mae, "\nThe R2 Score = ", r2)


# In[121]:


import pickle
model_in= open("House_model1.pkl", 'rb')
classy = pickle.load(model_in)


# In[123]:


import pickle
model_out= open("House_model1.pkl", 'wb')
pickle.dump(model, model_out)
model_out.close


# In[122]:


input_data = (2003, 3, 2, 82450, 2)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped =input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)
print(prediction)


# In[ ]:





# In[ ]:




