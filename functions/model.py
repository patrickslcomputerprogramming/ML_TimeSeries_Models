import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from sklearn.multioutput import MultiOutputRegressor  # For multi-output regression



# Data Preprocessing to create lag features
def preprocess_data(df):
    '''
    Select and transform the most appropriate columns 
    '''
    # Set the Number of decimal digits after the floating point.
    # pd.set_eng_float_format(accuracy=2, use_eng_prefix=True)
    
    # Add columns 'month' and 'days'
    df['month'] = pd.to_datetime(df['date']).dt.month.values
    df['day'] =  pd.to_datetime(df['date']).dt.day.values
    
    # Drop the column 'date'
    new_df= df.drop(columns= ['date'])

    # Group by the columns 'month' and 'month' and aggregate by calculating the mean of the other columns (different year, same month and day)
    new_df = new_df.groupby(['month', 'day']).agg({
        'purchases': 'median',
        'unique_invoices': 'median',
        'unique_streams': 'median',
        'total_views': 'median',
        'revenue': 'median'
    }).reset_index()
    
    #Rename columns 
    new_df = new_df.rename(columns={"revenue": "current_revenue"})
    new_df = new_df.rename(columns={"month": "current_month"})
    new_df = new_df.rename(columns={"day": "current_day"})

    # Calculate sum of revenue after 31 more days 
    n = len(new_df)  # number of rows in the dataframe
    period = 32  # number of days to sum
    new_df['next_month_revenue'] = 0.0
    new_df['next_month'] = 0  
    new_df['next_day'] = 0  
    for i in range(n):
        # Use modulo to wrap around for the last rows when not enough rows are left
        new_rev_value = 0
        for j in range(period):
            new_rev_value += new_df['current_revenue'].iloc[(i + j) % n]
        
        # Use .loc to avoid chained assignment warning
        new_df.loc[i, 'next_month_revenue'] = new_rev_value
        
        # Calculate the target date (the last date in the period)
        target_month = new_df['current_month'].iloc[(i + period - 1) % n]
        new_df.loc[i, 'next_month'] = target_month
        target_month = new_df['current_day'].iloc[(i + period - 1) % n]
        new_df.loc[i, 'next_day'] = target_month

       
    return new_df

# Split the data into features and target
def split_into_predictors_targets(df, Xsize=None):
    if Xsize == 1 :
        X = df[['day_of_year']]
    elif Xsize == 6:
        X = df[['day_of_year', 'current_revenue', 'purchases', 'unique_invoices', 'unique_streams', 'total_views']]
    else:
        X = df[['current_month', 'current_day']]
    
    if Xsize == 6:
        y = df[['next_month_revenue']]
    else:   
        y = df[['next_month_revenue', 'current_revenue', 'purchases', 'unique_invoices', 'unique_streams', 'total_views']]
    
    return X, y

# split the data into a training and test sets
def split_into_training_test(X, y):
    # Split into train and test sets (we'll use 80% for training and 20% for testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)
    return X_train, X_test, y_train, y_test


# Train and evaluate a model
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Compute evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    return mae, mse, rmse, r2, y_pred

# Train, predict, and evaluate each model
def train_predict_evaluate(X_train, y_train, X_test, y_test, nbr_of_y=None): 
    if nbr_of_y == 1 :
        # Linear Regression Model
        lr_model = LinearRegression()
  
        # Random Forest Model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                
        # XGBoost Model
        xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
               
    else:   
        # Linear Regression Model
        lr_model = MultiOutputRegressor(LinearRegression())
       
        # Random Forest Model
        rf_model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
          
        # XGBoost Model
        xgb_model = MultiOutputRegressor(xgb.XGBRegressor(n_estimators=100, random_state=42))
       
    #Calculate y (predicted data) and performance metrics
    mae_lr, mse_lr, rmse_lr, r2_lr, y_pred_lr = train_and_evaluate_model(lr_model, X_train, y_train, X_test, y_test)
    mae_rf, mse_rf, rmse_rf, r2_rf, y_pred_rf = train_and_evaluate_model(rf_model, X_train, y_train, X_test, y_test)
    mae_xgb, mse_xgb, rmse_xgb, r2_xgb, y_pred_xgb = train_and_evaluate_model(xgb_model, X_train, y_train, X_test, y_test)
    # Display lr performance metrics
    print("Linear Regression Model:")
    print("Mean Absolute Error (MAE): %.2f" % mae_lr)
    print("Mean Squared Error (MSE) : %.2f" % mse_lr)
    print("Root Mean-Square Error (RMSE) : %.2f" % rmse_lr)
    print("Coefficient of Determination (R2) : %.2f" % r2_lr)
    
    # Display rf performance metrics
    print("\nRandom Forest Model:")
    print("Mean Absolute Error (MAE): %.2f" % mae_rf)
    print("Mean Squared Error (MSE) : %.2f" % mse_rf)
    print("Root Mean-Square Error (RMSE) : %.2f" % rmse_rf)
    print("Coefficient of Determination (R2) : %.2f" % r2_rf)
    
    # Display xgb performance metrics
    print("\nXGBoost Model:")
    print("Mean Absolute Error (MAE): %.2f" % mae_xgb)
    print("Mean Squared Error (MSE) : %.2f" % mse_xgb)
    print("Root Mean-Square Error (RMSE) : %.2f" % rmse_xgb)
    print("Coefficient of Determination (R2) : %.2f" % r2_xgb)
    
    # Display evaluation metrics together
    d = {
    'MAE': [mae_lr, mae_rf, mae_xgb], 
    'MSE': [mse_lr, mse_rf, mse_xgb], 
    'RMSE': [rmse_lr, rmse_rf, rmse_xgb],
    'R2'  : [r2_lr, r2_rf, r2_xgb]  
    }
    report = pd.DataFrame(data=d, index=(['Linear', 'Random Forest', 'XGBoost']))
    print ("\nEvaluation Summary : \n",report)
    
    return y_pred_lr, y_pred_rf, y_pred_xgb, lr_model, rf_model, xgb_model 


def list_day_month_day_of_year():
    #Calculate day of the year (366 days) from month (1 to 12) and day (1 to 31 including exception) 
    day_of_year = {}
    count_day = 0
    for month in range (1,13):
        for day in range (1,32):
            #February have 29 days
            if (month==2 and day>29):
                break
            #April, June, September, November have 30 days 
            if ((month==4 or month==6 or month==9 or month==11) and day>30):
                break
            #Combine month and day (month-day) and calculate corresponding day of year
            month_day = str(month).zfill(2) + '-' + str(day).zfill(2)
            count_day += 1
            day_of_year[month_day] = count_day
    return day_of_year


def calculate_day_of_year(month, day):
    #Search day of the year (366 days) from month (1 to 12) and day (1 to 31 including exception)
    if (month<0) or (month>12) or (day<0) or (day>31) or (month==2 and day>29) or ((month==4 or month==6 or month==9 or month==11) and day>30):
        raise Exception("Something went wrong! Invalid date!")
    else :
        month_day = str(int(month)).zfill(2) + '-' + str(int(day)).zfill(2)
        corresponding_day_of_year = list_day_month_day_of_year()[month_day]
    return corresponding_day_of_year   


def to_day_of_year_df(df):
    #Calculate day of year
    day_range =[]
    nbr_rows = len(df)
    for i in range (nbr_rows):
        month=int(df['current_month'].iloc[i])
        day=int(df['current_day'].iloc[i])
        day_range.append(calculate_day_of_year(month, day))
        
    #Calculate indexes
    indexes=[]
    for item in df.index:
        indexes.append(item) 
        
    #Create a dataframe with the days of year
    df_day_of_year=pd.DataFrame()
    df_day_of_year['day_of_year'] = pd.DataFrame(data=day_range, index=(indexes))
    for column_name in df.columns:
        df_day_of_year[column_name] = df[column_name]
        #df_day_of_year['current_day'] = df['current_day']
  
    return df_day_of_year
    

# Example usage
if __name__ == "__main__":
    # Load the data (replace with your dataset)
    df = pd.read_csv('your_data.csv')  # Replace with your actual dataset path

    models = train_predict_evaluate(df)
      
