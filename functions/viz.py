#Import the required local files
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import matplotlib.ticker as ticker

from functions import  model  

def show_lineGraph(df, xAxisName, yAxisName):
    """
    Display a line plot 
    Line Plots display information as a series of data points called 'markers' connected by straight line segments
    """
    #Prepare the graph
    #sns.set_style("whitegrid")
    #plt.style.use(['ggplot'])
    fig, g = plt.subplots(figsize = (20,6))
    g = sns.lineplot(data=df, x=xAxisName, y=yAxisName)
    titleText='Line graph to understand the data distribution of '+str(yAxisName)+' by '+str(xAxisName)
    plt.title(titleText)
    plt.xticks(rotation=45)
    loc = mdates.MonthLocator(bymonth=range(1, 13))
    g.xaxis.set_major_locator(loc)
    #Display the graph
    plt.show(g)
    
def show_heatmap(df):
    """
    Display a heatmap including each column that includes numerical data
    Heatmaps can be used to check the correlation, dependence or relationship, if any, between each pair of columns
    """
    #Keep only continuous column values
    #Create an array with non-continuous datatypes columns like datetime, object 
    nan_dtype_column_name=[]
    for column_name in df.columns:
        if (df[column_name].dtypes!='int' and df.dtypes[column_name]!='float'):
            nan_dtype_column_name.append(column_name)
    #Drop non-continuous datatypes columns from the dataframe
    size_nan_dtype_column_name=len(nan_dtype_column_name)
    if (size_nan_dtype_column_name>0) :
        print("The following columns do not contain numerical data and then were removed:")
        for i in range(0, size_nan_dtype_column_name):
            print("#"+str(i+1)+" : "+str(nan_dtype_column_name[i]))
    df=df.drop(columns=nan_dtype_column_name)
    #Calculate correlation
    correlation = df.corr()
    #Display correlation
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation, annot=True, cmap="crest")
    plt.title('\nHeatmap to Show Correlations Between Continuous Columns Values')
    
# Plotting model test results (existing and predicted data)
def plot_test_data(X_test, y_test, y_pred_lr, y_pred_rf, y_pred_xgb, target_column):
    fig, ax1 = plt.subplots(figsize = (20,6))
    
    if target_column == 1:
        #Plot observed data used for test
        ax1.plot(X_test, y_test, label=f'Actual {y_test.columns}', color='black', marker='o', linestyle='-', linewidth=2)

        # Plot predictions from Linear Regression
        plt.plot(X_test, y_pred_lr, label=f'LR Predicted {y_test.columns}', color='blue', marker='o', linestyle='--', linewidth=2)
        
        # Plot predictions from Random Forest
        plt.plot(X_test, y_pred_rf, label=f'RF Predicted {y_test.columns}', color='red', marker='o', linestyle='--', linewidth=2)
        
        # Plot predictions from XGBoost
        plt.plot(X_test, y_pred_xgb, label=f'XGB Predicted {y_test.columns}', color='green', marker='o', linestyle='--', linewidth=2)
    
    else :
    # Plot actual values
        #Plot observed data used for test
        ax1.plot(X_test, y_test[target_column], label=f'Actual {target_column}', color='black', marker='o', linestyle='-', linewidth=2)

        # Plot predictions from Linear Regression
        plt.plot(X_test, y_pred_lr[:, list(y_test.columns).index(target_column)], label=f'LR Predicted {target_column}', color='blue', marker='o', linestyle='--', linewidth=2)
        
        # Plot predictions from Random Forest
        plt.plot(X_test, y_pred_rf[:, list(y_test.columns).index(target_column)], label=f'RF Predicted {target_column}', color='red', marker='o', linestyle='--', linewidth=2)
        
        # Plot predictions from XGBoost
        plt.plot(X_test, y_pred_xgb[:, list(y_test.columns).index(target_column)], label=f'XGB Predicted {target_column}', color='green', marker='o', linestyle='--', linewidth=2)
    
    
    # Add labels and title
    plt.xlabel('Month.Day')
    plt.ylabel(target_column)
    plt.title(f'{target_column} Predictions Comparison')
    plt.legend()
    plt.grid(True)
    ax1.set_xlim(330, 366)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()
    
    
def show_barChart2(df):
    fig, g = plt.subplots(figsize = (20,6))
    g = sns.barplot(data=df, x='number_of_transactions', y='country', orient='h')
    titleText='Bar chart - Top 10 countries with more transactions'
    plt.title(titleText)
    plt.xticks(rotation=45)
    plt.grid(True)
    #Display the graph
    plt.show(g)
    
def show_boxplot(df, col_name):
    """
    Display a boxplot of any column that includes numerical data
    Boxplots shows data points (outliers) that differ significantly from the rest of the data analyzed
    """
    #Prepare the graph
    sns.boxplot(data=df,x=col_name)
    titleText='Boxplot to identify outliers in the columns: '+ str(col_name)
    plt.title(titleText)
    plt.grid(True)
    #Display the graph
    plt.show()