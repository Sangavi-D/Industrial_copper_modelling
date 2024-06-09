import pandas as pd
import numpy as np
import pickle
from math import exp 
import streamlit as st

st.title("Copper Price and Status Prediction")
# Define features for status prediction
status_features = ['quantity tons_log','selling_price_log','application','thickness_log','width','country','customer','product_ref']
# status_features.remove("selling_price_log")  # Exclude selling price for status prediction




def predict_selling_price(data, model_file="random_forest_model.pkl", scaler_file="scaler.pkl", ohe_file="ohe.pkl"):
  """
  Predicts selling price using a trained Random Forest model and saved scaler and encoder.

  Args:
      data (pandas.DataFrame): DataFrame containing features for prediction.
      model_file (str, optional): Path to the pickled Random Forest model file. Defaults to "random_forest_model.pkl".
      scaler_file (str, optional): Path to the pickled scaler file. Defaults to "scaler.pkl".
      ohe_file (str, optional): Path to the pickled OneHotEncoder file. Defaults to "ohe.pkl".

  Returns:
      float: Predicted selling price.
  """

  # Load the model, scaler, and encoder
  with open(model_file, "rb") as f:
      model = pickle.load(f)
  with open(scaler_file, "rb") as f:
      scaler = pickle.load(f)
  with open(ohe_file, "rb") as f:
      ohe = pickle.load(f)

  # Extract features from data
  selling_price_features = [
      "quantity tons_log",
      "application",
      "thickness_log",
      "width",
      "country",
      "customer",
      "product_ref",
  ]
  X = data[selling_price_features]

  # Encode categorical features
  X_ohe = ohe.transform(data[["item type", "status"]]).toarray()

  # Combine numerical and encoded features
  X = np.concatenate(
      (
          X[[
              "quantity tons_log",
              "application",
              "thickness_log",
              "width",
              "country",
              "customer",
              "product_ref",
          ]].values,
          X_ohe,
      ),
      axis=1,
  )

  # Scale the features
  X = scaler.transform(X)

  # Make prediction
  predicted_price = model.predict(X)[0]  # Assuming it returns an array, return the first element

   # Convert predicted price from log scale to actual price
  actual_selling_price = exp(predicted_price)

  return actual_selling_price

# # Example usage
# data = pd.DataFrame({
#   "quantity tons_log": [55],
#   "application": [10],
#   "thickness_log": [2],
#   "width": [1200],
#   "item type": ['W'],
#   "status": ['Won'],
#   "country": [28],
#   "customer": [30156308.00],
#   "product_ref": [1670798778],
# })



#Function to predict the status using ExtraTreesClassifier

def predict_status(data, model_file="extra_trees_model.pkl", label_encoder_target_file="label_encoder_target.pkl", label_encoders_file="label_encoders.pkl"):
  """
  Predicts the status ("Won" or "Lost") for a new data point using a trained ExtraTreesClassifier model and saved label encoders.

  Args:
      data (pandas.DataFrame): DataFrame containing features for prediction.
      model_file (str, optional): Path to the pickled ExtraTreesClassifier model file. Defaults to "extra_trees_model.pkl".
      label_encoder_target_file (str, optional): Path to the pickled LabelEncoder file for the target variable. Defaults to "label_encoder_target.pkl".
      label_encoders_file (str, optional): Path to the pickled dictionary containing LabelEncoders for categorical features. Defaults to "label_encoders.pkl".

  Returns:
      str: Predicted status ("Won" or "Lost").
  """

  # Load the model and label encoders
  with open(model_file, "rb") as f:
      model = pickle.load(f)
  with open(label_encoder_target_file, "rb") as f:
      label_encoder_target = pickle.load(f)
  with open(label_encoders_file, "rb") as f:
      label_encoders = pickle.load(f)

  # Extract features from data
  features = ['quantity tons_log', 'selling_price_log',  'application',
 'thickness_log', 'width', 'country', 'customer', 'product_ref','item type']
  data_processed = data[features]

  # Encode categorical features
  categorical_features = ['item type']
  for feature in categorical_features:
      data_processed[feature] = label_encoders[feature].transform(data_processed[feature])

  # Make prediction
  predict_status = model.predict(data_processed)
#   print(predict_status.shape)
  predicted_status_decoded = label_encoder_target.inverse_transform(predict_status)


  return predicted_status_decoded

# Example usage
# data = pd.DataFrame({
#   "quantity tons_log": [55],
#   "selling_price_log": [3.5],
#   "application": [10],
#   "thickness_log": [0.8],
#   "width": [1200],
#   "country": [28],
#   "customer": [30156308.00],
#   "product_ref": [1670798778],
#   "item type": ['IPL']
# })

# predicted_status = predict_status(data)


# Create tabs
tab1, tab2 = st.tabs(["Predict Selling Price", "Predict Status"])

with tab1:
    # Selling price prediction form
    st.header("Predict Copper Selling Price")
    quantity_tons_log = st.number_input("Quantity (tons, log-scaled)", min_value=0.0)
    thickness_log = st.number_input("Thickness (log-scaled)", min_value=0.0)
    width = st.number_input("Width")
    customer = st.number_input("Customer")

    # Dropdowns for categorical features
    item_type_options = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
    item_type = st.selectbox("Item Type", item_type_options)
    status_options = ['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM',
       'Wonderful', 'Revised', 'Offered', 'Offerable']
    status = st.selectbox("Status", status_options)
    application_options = [10., 41., 28., 59., 15.,  4., 38., 56., 42., 26., 27., 19., 20.,
       66., 29., 22., 40., 25., 67., 79.,  3., 99.,  2.,  5., 39., 69.,
       70., 65., 58., 68.]
    application = st.selectbox("Application", application_options)
    country_options = [28,  25,  30,  32,  38,  78,  27,  77, 113,  79,  26,  39,  40,
        84,  80, 107,  89]  # Assuming countries are not encoded
    country = st.selectbox("Country", country_options)
    
    product_ref_options = [1670798778, 1668701718,     628377,     640665,     611993,
       1668701376,  164141591, 1671863738, 1332077137,     640405,
       1693867550, 1665572374, 1282007633, 1668701698,     628117,
       1690738206,     628112,     640400, 1671876026,  164336407,
        164337175, 1668701725, 1665572032,     611728, 1721130331,
       1693867563,     611733, 1690738219, 1722207579,  929423819,
       1665584320, 1665584662, 1665584642]  # Assuming product_ref is not encoded
    product_ref = st.selectbox("Product Ref", product_ref_options)

    if st.button("Predict selling price"):
        data = pd.DataFrame({
            "quantity tons_log": [quantity_tons_log],
            "application": [application],
            "thickness_log": [thickness_log],
            "width": [width],
            "item type": [item_type],
            "status": [status],
            "country": [country],
            "customer": [customer],
            "product_ref": [product_ref],
        })
        predicted_price = predict_selling_price(data)
        st.success(f"**Predicted Selling Price: {predicted_price:.2f}**")


with tab2:
    # Status prediction form
    st.header("Predict Copper Status")
    quantity_tons_log = st.number_input("Quantity (tons, log-scaled)", min_value=0.0,key=1)
    thickness_log = st.number_input("Thickness (log-scaled)", min_value=0.0,key=2)
    selling_price_log = st.number_input("selling_price (log-scaled)", min_value=0.0,key=3)
    width = st.number_input("Width",key=4)
    customer = st.number_input("Customer",key=5)

    # Dropdowns for categorical features
    item_type_options = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
    item_type = st.selectbox("Item Type", item_type_options,key=6)
    application_options = [10., 41., 28., 59., 15.,  4., 38., 56., 42., 26., 27., 19., 20.,
       66., 29., 22., 40., 25., 67., 79.,  3., 99.,  2.,  5., 39., 69.,
       70., 65., 58., 68.]
    application = st.selectbox("Application", application_options,key=7)
    country_options = [28,  25,  30,  32,  38,  78,  27,  77, 113,  79,  26,  39,  40,
        84,  80, 107,  89]  # Assuming countries are not encoded
    country = st.selectbox("Country", country_options,key=8)
    
    product_ref_options = [1670798778, 1668701718,     628377,     640665,     611993,
       1668701376,  164141591, 1671863738, 1332077137,     640405,
       1693867550, 1665572374, 1282007633, 1668701698,     628117,
       1690738206,     628112,     640400, 1671876026,  164336407,
        164337175, 1668701725, 1665572032,     611728, 1721130331,
       1693867563,     611733, 1690738219, 1722207579,  929423819,
       1665584320, 1665584662, 1665584642]  # Assuming product_ref is not encoded
    product_ref = st.selectbox("Product Ref", product_ref_options,key=9)

    if st.button("Predict status"):
        data = pd.DataFrame({
            "quantity tons_log": [quantity_tons_log],
            "application": [application],
            "thickness_log": [thickness_log],
            "width": [width],
            "item type": [item_type],
            "country": [country],
            "customer": [customer],
            "product_ref": [product_ref],
            'selling_price_log':[selling_price_log]
        })
        predicted_status = predict_status(data)
        st.success(f"**Predicted Status: {predicted_status[0]}**")
        

