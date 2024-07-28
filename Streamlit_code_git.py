import streamlit as st
import pandas as pd
import streamlit as st
import pickle

storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob(model_path)
blob.download_to_filename('local_model.pkl') 


model_path = './House_model1.pkl'
model_in = open(model_path, 'rb')
pred = pickle.load(model_in)

loc_dic = {
    "urban": 2,
    "suburban" : 1,
    "rural"  : 0
    }

def predict_price(data):
    predicted_price=pred.predict(data)[0]
    return predicted_price

    
def main():
    st.title("Welcome to the OOU Real Estate Platform")
    st.header('Be Ready to Get The Best!')
    global username, show_welcome_page
    username= st.text_input("Your Name")
    
    st.write("Wecome, ", username)
    
    st.write("LET US KNOW YOUR TASTE")
    global no_bedroom, no_bathroom, location, area, house_age
    house_age = int(st.number_input("House Age"))
    no_bedroom = st.number_input(" Number of Bedroom")
    no_bathroom = st.number_input(" Number of Bathroom")
    area = st.number_input ("Area in Square ft ")
    location_word =st.text_input(" Location (Rural, SubUrban or Urban Area?)").lower()
    location = loc_dic.get(location_word, 0)
    
    
    if st.button("House Price"):
        user_data=pd.DataFrame({
        "House Age" : [house_age],
        "Numbers of Bedrooms": [no_bedroom],
        "Number of Bathrooms" :[no_bathroom],
        "Square Feet" : [area], 
        "Location" : [location]
        })
    
        predicted_price = predict_price(user_data)
        predicted_price_rounded = round(predicted_price, 2) #rounded to 2 decimal places
        
        
        st.write("Base on your requirements, the House price is: ", predicted_price_rounded)
     
    
    
        
    
    



if __name__ == "__main__":
    main() 
