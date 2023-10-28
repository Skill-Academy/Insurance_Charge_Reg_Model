import streamlit as st
import pickle
import numpy as np


lr = pickle.load(open('lr_model_28Oct.pkl','rb'))
dt = pickle.load(open('dt_model_28Oct.pkl','rb'))
rf = pickle.load(open('rf_model_28Oct.pkl','rb'))


st.title('Insurance Charge prediction Web App')
st.subheader('Fill the below Details to predict Insurance Charges')


model = st.sidebar.selectbox('Select the ML Model',['Lin_Reg','DT_Reg','RF_Reg'])


# age 	sex 	bmi 	children 	smoker 	charges 	
# region_northwest 	region_southeast 	region_southwest

age = st.slider('Age',18,64)
sex = st.selectbox('Sex',['Male','Female'])
bmi = st.slider('BMI',15,53)
children = st.selectbox('Children',[0,1,2,3,4,5])
smoker = st.selectbox('Smoker',['Yes','No'])
region = st.selectbox('Region',['NorthWest','NorthEast','SouthWest','SouthEast'])


if st.button('Predict Insurance Charges'):
    if sex=='Male':
        sex = 1
    else:
        sex = 0
    if smoker == "Yes":
        smoker = 1
    else:
        smoker = 0
    if region=="NorthWest":
        nwest = 1
        neast = 0
        swest = 0
        seast = 0
    elif region=="SouthWest":
        nwest = 0
        neast = 0
        swest = 1
        seast = 0
    elif region=="SouthEast":
        nwest = 0
        neast = 0
        swest = 0
        seast = 1
    else:
        nwest = 0
        neast = 1
        swest = 0
        seast = 0

    test = np.array([age,sex,bmi,children,smoker,nwest,swest,seast])
    test = test.reshape(1,8)
    if model == "Lin_Reg":
        st.success(lr.predict(test)[0])
    elif model == "DT_Reg":
        st.success(dt.predict(test)[0])
    else:
        st.success(rf.predict(test)[0])

















# To run Streamlit Web App
# streamlit run app.py





