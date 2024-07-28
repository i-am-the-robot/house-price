import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the model
model_file = 'House_model.pkl'
with open(model_file, 'rb') as file:
    model = pickle.load(file)

# Location dictionary
loc_dic = {
    "urban": 2,
    "suburban": 1,
    "rural": 0
}

def predict_price(input_data):
    # Convert input data to NumPy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the array for prediction
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Make predictions
    predicted_price = model.predict(input_data_reshaped)

    return predicted_price[0]

def main():
    st.title("Welcome to the OOU Real Estate Platform")
    st.header('Be Ready to Get The Best!')

    # User input
    username = st.text_input("Your Name")
    st.write("Welcome,", username)

    st.write("LET US KNOW YOUR TASTE")
    house_age = st.number_input("House Age")
    no_bedroom = st.number_input("Number of Bedroom")
    no_bathroom = st.number_input("Number of Bathroom")
    area = st.number_input("Area in Square ft")
    location_word = st.text_input("Location (Rural, SubUrban or Urban Area?)").lower()
    location = loc_dic.get(location_word, 0)

    # Prediction button
    if st.button("House Price"):
        input_data = {
            "House Age": [house_age],
            "Numbers of Bedrooms": [no_bedroom],
            "Number of Bathrooms": [no_bathroom],
            "Square Feet": [area],
            "Location": [location]
        }
        input_data_df = pd.DataFrame(input_data)

        predicted_price = predict_price(input_data_df)
        predicted_price_rounded = round(predicted_price, 2)

        st.write("Based on your requirements, the House price is:", predicted_price_rounded)

if __name__ == "__main__":
    main()
