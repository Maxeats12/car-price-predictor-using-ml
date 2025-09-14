#!/usr/bin/env python
# coding: utf-8

# #Importing the Dependencies

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics


# In[2]:


# loading the data from csv file to pandas dataframe
car_dataset = pd.read_csv("car data.csv")


# In[3]:


car_dataset.head()


# In[4]:


#checking the number of rows and columns
car_dataset.shape


# In[5]:


#getting some information about the dataset
car_dataset.info()


# In[6]:


#checking the null values
car_dataset.isnull().sum()


# In[7]:


#checking the distribution of categorical data
print(car_dataset.Fuel_Type.value_counts())
print(car_dataset.Seller_Type.value_counts())
print(car_dataset.Transmission.value_counts())


# #Encoding the categorical data 

# In[8]:


# encoding "Fuel_type" Column
car_dataset.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)

#encoding "Seller_Type" Column
car_dataset.replace({'Seller_Type':{'Dealer':0,'Individual':1}},inplace=True)

#encoding "Transmission" Column
car_dataset.replace({'Transmission':{'Manual':0,'Automatic':1}},inplace=True)


# In[9]:


car_dataset.head()


# In[ ]:





# #Splitting the data and Target

# In[10]:


X= car_dataset.drop(['Car_Name','Selling_Price'],axis=1)
Y= car_dataset['Selling_Price']


# In[11]:


print(X)
print(Y)


# #Splitting Training and Test data

# In[12]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1, random_state=2)


# #Model Training

# #1.Linear REgression

# In[13]:


#loading  the linear regression model
lin_reg_model= LinearRegression()


# In[14]:


lin_reg_model.fit(X_train,Y_train)


# In[15]:


#prediction on Training data
training_data_prediction=lin_reg_model.predict(X_train)


# In[19]:


#R Square Error
error_score = metrics.r2_score(Y_train,training_data_prediction)
print("R Square Error: ",error_score)


# In[20]:


#Visualize the actual Prices and Predicted Prices


# In[22]:


plt.scatter(Y_train,training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Prices vs Predicted Prices")


# In[27]:


#prediction on Test data
test_data_prediction=lin_reg_model.predict(X_test)


# In[28]:


#R square Error
error_score = metrics.r2_score(Y_test,test_data_prediction)
print("R squared Error: ",error_score)


# In[29]:


plt.scatter(Y_test,test_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Prices vs Predicted Prices")


# #Lasso Regression

# In[32]:


#loading  the lasso regression model
lass_reg_model= Lasso()


# In[33]:


lass_reg_model.fit(X_train,Y_train)


# In[34]:


#prediction on Training data
training_data_prediction=lass_reg_model.predict(X_train)


# In[36]:


#R Square Error
error_score = metrics.r2_score(Y_train,training_data_prediction)
print("R Square Error: ",error_score)


# In[37]:


#Visualize the actual Prices and Predicted Prices


# In[38]:


plt.scatter(Y_train,training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Prices vs Predicted Prices")


# In[39]:


#prediction on Training data
test_data_prediction=lass_reg_model.predict(X_test)


# In[40]:


#R square Error
error_score = metrics.r2_score(Y_test,test_data_prediction)
print("R squared Error: ",error_score)


# In[41]:


plt.scatter(Y_test,test_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Prices vs Predicted Prices")


# In[ ]:





# In[ ]:




