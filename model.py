from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
#import numpy as np
import pickle


# load dataset
streeteasy = pd.read_csv(
    "https://raw.githubusercontent.com/sonnynomnom/Codecademy-Machine-Learning-Fundamentals/master/StreetEasy/manhattan.csv")

df = pd.DataFrame(streeteasy)

# independent varables
x = df[['bedrooms', 'bathrooms', 'size_sqft', 'min_to_subway', 'floor', 'building_age_yrs', 'no_fee',
        'has_roofdeck', 'has_washer_dryer', 'has_elevator', 'has_dishwasher', 'has_patio', 'has_gym']]

# dependent variable
y = df[['rent']]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, test_size=0.2, random_state=6)

# init model
lm = LinearRegression()

model = lm.fit(x_train, y_train)

# save the model to disk
filename = 'model.sav'
pickle.dump(model, open(filename, 'wb'))
