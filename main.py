import pandas as pd
from sklearn import linear_model

df = pd.read_csv("stars_competitor_test.csv")

# spec_class_dict = {'M':1, 'K':2, 'G':3, 'F':4, 'A':5, 'B':6, 'O':7}
# # color_dict = {'Red':12068133, 'Blue':2437560, 'Orange':15105042, 'Blue White':}

# let = []
# for i in range(len(df['Color'])):
#     if df['Color'][i] not in let:
#         let.append(df['Color'][i])

# print(let)

X = df[['Temperature', 'L', 'R', 'A_M']]
y = df['Type']

regr = linear_model.LinearRegression()
regr.fit(X, y)

predictedType = regr.predict([[3484, 0.000551, .0998, 16.67]])

print(predictedType)

