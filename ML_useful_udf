# User defined functions
def get_percent_missing_values(dataframe):
    """
    Calculate percentage of missing values in each column of dataframe
    Columns with no missing values won't be displayed
    """
    percent_missing = (dataframe.isnull().sum()/len(dataframe))*100
    percent_missing = round(percent_missing, 2)
    percent_missing = pd.DataFrame({'Percent_missing': percent_missing})
    return percent_missing[percent_missing.Percent_missing != 0].sort_values(by = 'Percent_missing',axis = 0, ascending = False)


def get_categorical_columns(dataframe):
    """
    To get list of all the categorical columns
    """
    all_columns = dataframe.columns
    numeric_cols = dataframe._get_numeric_data().columns
    return list(set(all_columns) - set(numeric_cols))

def dummify_categorical_features(data, columns):
    dummified_data = pd.get_dummies(data, columns = columns, drop_first=True)
    return dummified_data

# from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score, f1_score,roc_auc_score, classification_report
def binary_classification_performance(y_test, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    accuracy = round(accuracy_score(y_pred = y_pred, y_true = y_test),2)
    precision = round(precision_score(y_pred = y_pred, y_true = y_test),2)
    recall = round(recall_score(y_pred = y_pred, y_true = y_test),2)
    f1_score = round(2*precision*recall/(precision + recall),2)
    specificity = round(tn/(tn+fp),2)
    npv = round(tn/(tn+fn),2)
    auc_roc = round(roc_auc_score(y_score = y_pred, y_true = y_test),2)


    result = pd.DataFrame({'Accuracy' : [accuracy],
                         'Precision or PPV' : [precision],
                         'Recall or senitivity or TPR' : [recall],
                         'f1 score' : [f1_score],
                         'AUC_ROC' : [auc_roc],
                         'Specificty or TNR': [specificity],
                         'NPV' : [npv],
                         'True Positive' : [tp],
                         'True Negative' : [tn],
                         'False Positive':[fp],
                         'False Negative':[fn]})
    return result
    

# from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
def regression_performace(y_true, y_pred):
    r2_score = r2_score(y_true = y_true, y_pred = y_pred)
    mean_absolute_error = mean_absolute_error(y_true = y_true, y_pred = y_pred)
    mean_squared_error = mean_squared_error(y_true = y_true, y_pred = y_pred)
    
    result =  pd.DataFrame({'r2_score': r2_score,
                           'Mean_Squared_Error': mean_squared_error,
                          'Mean_Absolute_Error': mean_absolute_error})
    return result
    
 
