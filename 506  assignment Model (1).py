#!/usr/bin/env python
# coding: utf-8




import pandas as pd
import numpy as np
import matplotlib.pyplot as mlt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import pickle





prediction= pd.read_csv('House_price.csv')
prediction





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


pred = model
loc_dic = {
    "urban": 2,
    "suburban" : 1,
    "rural"  : 0
    }

def predict_price(input_data):
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = pred.predict(input_data_reshaped)
    predicted_price = pred.predict(input_data_reshaped)
    return predicted_price[0]

    
def main():
    st.title("Welcome to the OOU Real Estate Platform")
    st.header('Be Ready to Get The Best!')
    global username, show_welcome_page
    username= st.text_input("Your Name")
    
    st.write("Wecome, ", username)
    
    st.write("LET US KNOW YOUR TASTE")
    global no_bedroom, no_bathroom, location, area, house_age
    house_age = (st.number_input("House Age"))
    no_bedroom = st.number_input(" Number of Bedroom")
    no_bathroom = st.number_input(" Number of Bathroom")
    area = st.number_input ("Area in Square ft ")
    location_word =st.number_input(" Location (Rural, SubUrban or Urban Area?)").lower()
    location = loc_dic.get(location_word, 0)
    
    if st.button("House Price"):
        input_data_reshaped=pd.DataFrame({
        "House Age" : [house_age],
        "Numbers of Bedrooms": [no_bedroom],
        "Number of Bathrooms" :[no_bathroom],
        "Square Feet" : [area], 
        "Location" : [location]
        })
    
        predicted_price = predict_price(input_data_reshaped)
        predicted_price_rounded = round(predicted_price, 2) #rounded to 2 decimal places
        
        
        st.write("Base on your requirements, the House price is: ", predicted_price_rounded)
     
    
    
        
    
    



if __name__ == "__main__":
    main() 










