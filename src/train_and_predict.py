from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import  mean_squared_error
import matplotlib.pyplot as plt
import os
import pandas as pd

def  train_and_predict():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "..", "data", "sample_data.csv")

    # Load data from CSV
    df = pd.read_csv(csv_path)
    X = df[["area_sqft"]]
    y= df["price_lakh"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    #predictions
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("mean squared error",mse)

    #single prediction
    area = [[1800]]
    predicted_price = model.predict(area)
    print(f"Predicted price for {area[0][0]} sqft = ₹{predicted_price[0]} lakh")

    #plot
    plt.scatter(X, y, color='blue', label='Actual data')
    plt.plot(X, model.predict(X), color='red', label='Regression price')
    plt.xlabel('Area (sqft)')
    plt.ylabel('Price (lakh ₹)')
    plt.legend()
    plt.show()


