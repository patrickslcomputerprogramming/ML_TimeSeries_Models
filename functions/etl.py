#To check whether the dataset was loaded or not
def get_df_shape(df):
    if (df.shape[0] == 0) :
        print("Something went wrong! No data were the dataframe.")
    else:
        # Display the shape of the data 
        print("SHAPE OF THE DATA: ")
        print("Number of rows: " + str(df.shape[0]))
        print("Number of columns: " + str(df.shape[1]))

#To get get exploratory data analysis data
def get_exploratory_data_analysis_data(df):
    # Check whether there is categorical data
    size = len (df)
    objectIn = numericalIn = 0
    for i in range (size):
        if df.iloc[i].dtypes=='object':
            objectIn += 1
            break
        else :
            numericalIn += 1
            break
    if numericalIn > 0 :
        #Display descriptive statistics 
        print("\nDESCRIPTIVE STATISTICS ABOUT NUMERICAL COLUMNS")
        print(df.describe().T)
    if objectIn > 0 :
        print("\nDESCRIPTIVE STATISTICS ABOUT CATEGORICAL OBJECT COLUMNS")
        print(df.describe(include="object"))

#To count column data categories   
def count_column_data_categories(df):
    #Calculate the number of categories of data included in each column
    #Display all columns with their number of categories of data when they are 
    uniqueCategoryColumns=[]
    for col in df.columns: 
        print(col, ":", df[col].unique().shape[0]) 
        if(df[col].unique().shape[0]==1):
            uniqueCategoryColumns.append(col)
    #Display only columns with unique category data when they are 
    sizeUniqueCategoryColumns=len(uniqueCategoryColumns)
    if(sizeUniqueCategoryColumns>0):
        print("The following columns includes a unique category of data: ")
        for i in range (0, sizeUniqueCategoryColumns):
            print(str(uniqueCategoryColumns)+" ")
    else:
        print("\nThere is no column that includes a unique category of data") 
    
#To ckeck for duplicates rows
def check_duplicate_rows(df):
    print("\nDUPLICATED DATA ROWS")
    #Calculate duplicate (True) and non duplicate (False) rows
    duplicateOrNot=df.duplicated()
    numberOfDuplicatedRows = duplicateOrNot.sum()
    if (numberOfDuplicatedRows > 0):
        print(str(numberOfDuplicatedRows)+" rows are duplicated")
        #Display first 10 dupplicate rows
        print("First 5 duplicated rows:")
        j=1 #Counter to display a specific number of duplicate rows
        #Identify and display only duplicate (True) rows 
        for i in df.index:
            if (duplicateOrNot[i]==True) :
                print (str(df.iloc[[i]])) 
                j = j + 1
                if j==5:
                    break #Stop identifying duplicate after displaying the tenth 
    else:
        print("There is no duplicated rows")


#To observe data
def get_data_info(df):
    """
    observe data to identify required tranforms
    """    
    # Display the first 10 rows of the dataframe
    print("\nFIRST 10 ROWS OF THE DATA")
    print(df.head(10))

    # Display the last 10 rows of the dataframe
    print("\nLAST 10 ROWS OF THE DATA")
    print(df.tail(10))

    #Display info about the DataFrame, including the index dtype and columns, non-null values and memory usage
    print("\nINFO ABOUT THE DATA")
    print(df.info())
    
    # Display basic information about the data
    print("\nBASIC INFO ABOUT THE DATA")
    print(df.info())

    # Check for missing values               
    column_names = []
    for i in df.columns:
        column_names.append(i)
    print("\nMISSING DATA")
    numberOfColumnsWithMissingData = 0
    for name in column_names:
        numberOfMissingData = df[name].isna().sum()
        if (numberOfMissingData > 0):
            print (str(numberOfMissingData) + " data are missing from the column named " + str(name))
            numberOfColumnsWithMissingData += 1
    if (numberOfColumnsWithMissingData == 0) :
            print ("There is no missing data")

    #Ckeck for duplicates rows
    check_duplicate_rows(df)
        
    #Calculate statistics
    get_exploratory_data_analysis_data(df)
    
    #Calculate the number of categories by column 
    count_column_data_categories(df)