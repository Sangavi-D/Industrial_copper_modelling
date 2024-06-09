
# Industrial Copper Modeling

This project aims to develop an efficient machine learning regression model to predict the selling price and a classification model for lead classification.The models utilizes advanced techniques such as data normalization, outlier detection ,handling data in the wrong format, identifying the distribution of features, and leveraging tree-based models, to predict the selling price and leads accurately.



## Requirements
NumPy

Pandas

Scikit-learn

Streamlit
## Steps involved
1. Transform the data into a suitable format and perform any necessary cleaning and pre-processing steps.
2. Exploring skewness and outliers in the dataset.
3. ML Regression model which predicts continuous variable ‘Selling_Price’.
4. ML Classification model which predicts Status: WON or LOST.
5. Creating a streamlit page where column values can be inserted and you will get the Selling_Price predicted value or Status(Won/Lost).

## Getting started
Original uncleaned Dataset : https://docs.google.com/spreadsheets/d/18eR6DBe5TMWU9FnIewaGtsepDbV4BOyr/edit#gid=462557918
1. Start with data preprocessing steps(icm_data_preprocessing.ipynb) using the dataset provided and store the cleaned data seperately.
2. Choose the best regression and classification model by comparing the evaluation metrics of each model(selecting_model.ipynb)
3. Here for regression DecisionTreeRegressor,RandomForestRegressor and ExtraTreesRegressor were compared and RandomForestRegressor was found to be better.
4. For classification model ExtraTreesClassifier, XGBClassifier and LogisticRegression were compared and ExtraTreesClassifier was found to be better.
5. Use pickle module to dump and load models.
6. Create streamlit app to get user input and predict results.