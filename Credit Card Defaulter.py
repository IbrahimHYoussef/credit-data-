import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import streamlit as st 


header=st.beta_container() 
the_data_set=st.beta_container()
data_analysis=st.beta_container()
model_work=st.beta_container()

with header:
    st.title('Credit Card defaulter')
    st.text('why people default on their credit cards and what can we learn from that')
    st.subheader('Content')
    
with the_data_set:
    st.header('The main data set')
    st.text('This data set is from https://www.kaggle.com/gauravtopre/credit-card-defaulter-prediction it was chosen as a quick demenstration of skill to show what can i do with streamlit')
    st.subheader('Column Information')
    st.text('1=payment delay for 1 month 9=payment delay for nine months and more)')
    st.markdown('PAY0: Repayment status in September 2005 (-1=pay duly)')             
    st.markdown('PAY2: Repayment status in August 2005 (scale same as above)')
    st.markdown('PAY3: Repayment status in July 2005 (scale same as above)')
    st.markdown('PAY4: Repayment status in June 2005 (scale same as above)')
    st.markdown('PAY5: Repayment status in May 2005 (scale same as above)')
    st.markdown('PAY6: Repayment status in April 2005 (scale same as above)')
    st.markdown('BILLAMT1: Amount of bill statement in September 2005 (NT dollar)')
    st.markdown('BILLAMT2: Amount of bill statement in August 2005 (NT dollar)')
    st.markdown('BILLAMT3: Amount of bill statement in July 2005 (NT dollar)')
    st.markdown('BILLAMT4: Amount of bill statement in June 2005 (NT dollar)')
    st.markdown('BILLAMT5: Amount of bill statement in May 2005 (NT dollar)')
    st.markdown('BILLAMT6: Amount of bill statement in April 2005 (NT dollar)')
    st.markdown('PAYAMT1: Amount of previous payment in September 2005 (NT dollar)')
    st.markdown('PAYAMT2: Amount of previous payment in August 2005 (NT dollar)')
    st.markdown('PAYAMT3: Amount of previous payment in July 2005 (NT dollar)')
    st.markdown('PAYAMT4: Amount of previous payment in June 2005 (NT dollar)')
    st.markdown('PAYAMT5: Amount of previous payment in May 2005 (NT dollar)')
    st.markdown('PAYAMT6: Amount of previous payment in April 2005 (NT dollar)')
    
    
    c_default=pd.read_csv('C:/Users/Ibrahim/Desktop/Data dem/Credit Card Defaulter Prediction.csv')
    c_default['EDUCATION']= c_default['EDUCATION'].str.replace('0','Unknown')
    c_default['MARRIAGE']=c_default['MARRIAGE'].str.replace('0','Other')
    c_default['default ']=c_default['default '].str.replace('y','True')
    c_default['default ']=c_default['default '].str.replace('Y','True')
    c_default['default ']=c_default['default '].str.replace('n','False')
    c_default['default ']=c_default['default '].str.replace('N','False')
    c_default['default '] = c_default['default '].map({'False':False, 'True':True})
    c_default_s=pd.get_dummies(c_default['SEX'])
    c_default_e=pd.get_dummies(c_default['EDUCATION'])
    c_default_m=pd.get_dummies(c_default['MARRIAGE'])
    
    st.write(c_default.head(20))
    st.write(c_default.info())
    
    st.text('this data set is ready to go with the initial data analysis to find the main trends')

with data_analysis:
    st.header('Initial data analysis')
    st.text('looking into more detales in the data set to find where the trends are this will help us better understand the data set we are working with thus helping us get the rigth model for the job at hand')
    st.text('to better understand our data set some Questions need to asked')
    st.subheader('what dose the Descriptives statistics of the data set looks?')
    c_default_des=c_default.describe()
    st.write(c_default_des)
    st.subheader('How are the Columns correlated to each other?')
    sns.set(rc={"figure.figsize":(50,18)})
    sns.set(font_scale=3)
    st.write(sns.heatmap(c_default.corr(),annot=True))
    st.text('we can see that the main factors that have a high correlation to each other are payments month differacne and the bill amount other than that only few have as high of a correlation to each other')
    
    st.subheader('how dose sex effect credit card defalut?')
    st.text('male and Female count in the data set')
    sel_col, disp_col=st.beta_columns(2)
    dex=sel_col.selectbox('select the columns to see values with in',options=('SEX','EDUCATION','MARRIAGE','LIMIT_BAL'))
    dex_1=sel_col.selectbox('select the comparison valuse',options=('SEX','EDUCATION','MARRIAGE'),index=2)
    
    
    st.bar_chart(pd.DataFrame(c_default[dex].value_counts()))
    st.text('male and female count on who defalted')
    
    st.bar_chart(pd.pivot_table(data=c_default,index=['default ','SEX'],
                         values=['LIMIT_BAL'],aggfunc='count'))
pd.pivot_table(data=c_default,index=['default '],values=['LIMIT_BAL','SEX'],aggfunc='count')    
#c_default.BILL_AMT2.value_counts()    
'''   
with model_work:
    from sklearn import *
    st.header('Model creating and testing')
    st.text('the model is already trained tested and ready to go you just need to fill in the') 
    y=c_default['default ']
    y=pd.DataFrame(y)
    col_x=['LIMIT_BAL','AGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']
    x_1=c_default[col_x]
    x=pd.DataFrame()
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=1,random_state=1)
    from sklearn.linear_model import LogisticRegression
    card_model=LogisticRegression()
    card_model.fit(x,y)
    st.header('Predict your credit card default')
    st.
    
    pred=[]
    y_pred=card_model.predict()
'''    
    

  
