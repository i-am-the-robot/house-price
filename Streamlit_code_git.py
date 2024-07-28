himport streamlit as st
import pandas as pd
import pickle
pred = pickle.load(open('https://github.com/i-am-the-robot/house-price/blob/9053f5c503aa344cb558024300cffeb8999eed52/House_model1.pkl','rb'))


loc_dic = {
    "urban": 2,
    "suburban" : 1,
    "rural"  : 0
    }

def predict_price(data):
    data = data.values
    predicted_price = pred.predict(data)
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
