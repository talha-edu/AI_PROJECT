
from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import  mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
from xgboost import XGBRegressor

app = Flask(__name__)

# Load your dataset (replace 'your_dataset.csv' with the actual file path)
data = pd.read_csv(r'D:\PYTHON\AI\dataset\Property_with_Feature_Engineering_ORIGINAL.csv')

print(data)

print(data.describe())

print(data.info())

print(data.columns)


sns.barplot(data =data,
             x = data['property_type'],
             y = data['price']
            
           
             )
plt.show()


X= data.loc[:, ['property_type','location', 'city', 'province_name', 'area_sqft', 'baths',  'bedrooms']]

y_price = data['price']

print('BEFORE ENCODING: ')
print(X)

enc = OrdinalEncoder()
enc.fit(data.loc[:, ['property_type','location', 'city', 'province_name']])
X[['property_type','location', 'city', 'province_name']] = enc.transform(data.loc[:, ['property_type','location', 'city', 'province_name']])

print('AFTER ENCODING: ')
print(X)

X_train, X_test, y_price_train , y_price_test = train_test_split( X,y_price,test_size=0.4, random_state=42)


# Standardize the features (mean=0 and variance=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)




print(X_train_scaled)
print(y_price_train)


# Initialize the Random Forest Regressor for price prediction
price_regressor = RandomForestRegressor(random_state=42)
price_regressor.fit(X_train_scaled, y_price_train)

# Make predictions on the scaled test set
predicted_prices = price_regressor.predict(X_test_scaled)

# Calculate performance metrics for price prediction
mae_price = mean_absolute_error(y_price_test, predicted_prices)
mse_price = mean_squared_error(y_price_test, predicted_prices)
r2_price = r2_score(y_price_test, predicted_prices)


# Print the metrics
print('\n     ----||RANDOM FOREST REGRESSION||----\n___________________________________________')
print(f'Price - Mean Squared Error: {r2_price}')



plt.scatter(y_price_test,predicted_prices)

# Generating the parameters of the best fit line
m, c = np.polyfit(y_price_test,predicted_prices, 1)

# Plotting the straight line by using the generated parameters
plt.plot(y_price_test, m*y_price_test+c)

plt.show()




#FITTING THE DATA IN THE MODELS
degree = 2 #  the degree
poly_features = PolynomialFeatures(degree=degree)
X_train_scaled_poly = poly_features.fit_transform(X_train_scaled)
X_test_scaled_poly = poly_features.transform(X_test_scaled)

poly_reg = LinearRegression()
poly_reg.fit(X_train_scaled_poly, y_price_train)

# Ridge Regression
alpha_ridge = 1.0  # Set the regularization strength for Ridge Regression
ridge_reg = Ridge(alpha=alpha_ridge)
ridge_reg.fit(X_train_scaled, y_price_train)

# Lasso Regression
alpha_lasso = 1.0  # Set the regularization strength for Lasso Regression
lasso_reg = Lasso(alpha=alpha_lasso)
lasso_reg.fit(X_train_scaled, y_price_train)


#XGBOOST
xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)  # 'reg:squarederror' for regression tasks
xgb_model.fit(X_train_scaled, y_price_train)


#MAKING THE PREDICTIONS

y_pred_poly = poly_reg.predict(X_test_scaled_poly)
y_pred_ridge = ridge_reg.predict(X_test_scaled)
y_pred_lasso = lasso_reg.predict(X_test_scaled)
y_pred_XG = xgb_model.predict(X_test_scaled)




# performance metrics

r2_poly = r2_score(y_price_test, y_pred_poly)



r2_ridge = r2_score(y_price_test, y_pred_ridge)



r2_lasso = r2_score(y_price_test, y_pred_lasso)





r2_XG = r2_score(y_price_test, y_pred_XG)





 #Display
print('\n\n     --POLYNOMIAL REGRESSION--') 
print('_______________________________________\n')


print("R2 Score:", r2_poly)


print('\n\n     --RIDGE REGRESSION--')
print('_______________________________________\n')


print("R2 Score:", r2_ridge)


print('\n\n     --LASSO REGRESSION--')
print('_______________________________________\n')

print("R2 Score:", r2_lasso)


print('\n\n     --XGBOOST REGRESSION--')
print('_______________________________________\n')

print("R2 Score:", r2_XG)





# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction and visualization
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input values from the form
            user_input = {}

            user_input['property_type'] = request.form['property_type']
           
            user_input['location'] = request.form['location']

            user_input['city'] = request.form['city']

            user_input['province_name'] = request.form['province_name']

            user_input['area_sqft'] = request.form['area_sqft']
                        
            user_input['baths'] = request.form['baths']
            
            user_input['bedrooms'] = request.form['bedrooms']



            # Convert user input to a DataFrame
            user_input_df = pd.DataFrame([user_input])
            user_input_df[['property_type','location', 'city', 'province_name']] = enc.transform(user_input_df[['property_type','location', 'city', 'province_name']])
            p  = price_regressor.predict(user_input_df)
        

            return render_template('result.html', p=p)



if __name__ == '__main__':
    app.run()
