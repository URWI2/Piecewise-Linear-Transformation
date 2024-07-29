
import numpy as np
import pandas as pd
import miceforest as mf
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from collections import OrderedDict
from copy import deepcopy 
import pickle 
import cloudpickle 


def delete_randomly_data(df, delete_percent, random_state=None):
    '''
    deletes a given percentage of values in the dataset randomly
    '''
    num_values_to_delete = int(df.size * delete_percent)
    np.random.seed(random_state)
    indices = np.random.choice(df.size, num_values_to_delete, replace=False)
    
    df_copy = df.copy(deep=True).values.flatten()
    if df_copy.dtype.kind == 'f':
        df_copy[indices] = np.nan
    else:
        df_copy = df_copy.astype(float)
        df_copy[indices] = np.nan
    
    return pd.DataFrame(df_copy.reshape(df.shape), columns=df.columns)
    
def distribution_mice(num_iter,kernel,column_pos,instance):
    
  #get the imputed values from the multiple imputed datasets
  imput_values = []
  for i in range(0,num_iter):
    imput_df = kernel.complete_data(dataset=i)
    try:
        column = imput_df.iloc[:,column_pos]
    except:
        column = imput_df[:,column_pos]
    row = column[instance]
    imput_values.append(row)
  return imput_values

def pdf_variance(points):
    
    # Calculate the PDF using a histogram with 100 bins 
    hist, bin_edges = np.histogram(points, bins=100, density=True)
    
    # Calculate the mean of the PDF
    mean = np.mean(hist)
    
    # Calculate the variance of the PDF
    variance = np.sum((hist - mean)**2 * np.diff(bin_edges))
    
    return variance

def fill_kde_dict(kde_dict,size):
    
    '''
    create ordered dict containing each instance number as key and either nan
    or the imputed kernel functions as values 
    '''
    
    sorted_dict = OrderedDict()

    for num in range(0, size-1):
        if num not in kde_dict:
            sorted_dict[num] = np.nan

    for key, value in kde_dict.items():
        sorted_dict[key] = value

    return sorted_dict

def KDE_evaluation(X, X_complete, imputer, num_samples_mice, num_samples_kde, filename):
    '''
    impute missing data with scipy KDE distributions obtained via mice imputation 
    '''
    np.set_printoptions(suppress=True)
    # calculate percentage of deleted values across all attributes

    variance = 0
    count_pdfs = 0
    df = X.copy()
    #columns = df.columns
    kde_models_dict = {}
    for index, x in df.iterrows():
        instance_kde = []
        missing_dims = np.where(np.isnan(x))[0]  # Get the indices of missing dimensions
        if len(missing_dims) > 0:
            for i in range(len(x)):
                if i in missing_dims:
                    # extract the estimated points form the imputation for every dimension
        
                    mice_imp_points = np.array(distribution_mice(num_samples_mice, imputer, i, index))
                    mice_array = mice_imp_points.reshape((num_samples_mice,))
                    instance_kde.append(mice_array)
                    # evaluate the variance of the pdf
                    variance += pdf_variance(mice_imp_points)
                    count_pdfs += 1

            # define multinomial kernel density estimation
            instance_kde = np.array(instance_kde).T
            instance_kde = instance_kde.T
           

            # create KDE
            kde = stats.gaussian_kde(dataset=instance_kde,bw_method=0.2)
            # save kde in dict 
            kde_models_dict[index] = (kde,0)

        else:
            # x containts no missing value 
            kde_models_dict[index] = np.nan
    
    # safe kdes as pkl file
    kde_models_dict = fill_kde_dict(kde_models_dict,len(X))
    kde_models_dict = dict(sorted(kde_models_dict.items(), key=lambda x: x[0]))
    
    return kde_models_dict

def feature_importance(X,y):
    '''
    calculates the feature importance for X based on Random Forest Classifier
    '''
    try:
        # Initialisation of Random Forest Classifier
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        importance = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_}).sort_values('Importance', ascending=False)
        plt.barh(importance['Feature'], importance['Importance'])
        plt.xlabel('Importance')
        plt.title('Feature Importance')
        plt.show()
    except:
        # Initialisation of Random Forest Regressor 
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        importance = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_}).sort_values('Importance', ascending=False)
        plt.barh(importance['Feature'], importance['Importance'])
        plt.xlabel('Importance')
        plt.title('Feature Importance')
        plt.show()
        
        

if __name__ == "__main__":
    
    dataset = None
    
    DATASET_NAME = 'iris'
    MISSING = 25
   
    with open(f"{DATASET_NAME}/data/{DATASET_NAME}_data_test.npy", 'rb') as f:
        X = np.load(f)
        y = np.load(f)
    
    X = pd.DataFrame(X)
    X_complete = deepcopy(X)
    
    filename=""
    
    # Feature imporance
    feature_importance(X, y)    
   
    # delete values from selected columns 
    cols_to_delete = [0,1,3]
   
    X[cols_to_delete] = delete_randomly_data(X[cols_to_delete], MISSING*0.01, 31)
    
    # create kernel on missing data
    d = 50
    size = X.shape[0]
    print("Imputing data...")
    
    imputer = mf.ImputationKernel(
      np.array(X), 
      datasets=d,
      save_all_iterations=True,
      random_state=1
    )
    imputer.mice(4)
   
    print("Generating KDEs...")
    
    kde_models_dict = KDE_evaluation(X[:size], X_complete[:size], imputer, d, d,filename)
    
    try:
        KDE1 = kde_models_dict[1][0]
        kernel_min = np.min(KDE1.dataset)
        kernel_max = np.max(KDE1.dataset)
        x1 = np.linspace(kernel_min, kernel_max, 1000)
        y1 = KDE1.pdf(x1)
        plt.plot(x1, y1)
    except:
        pass


    #save results
    with open(f"{DATASET_NAME}/data/{DATASET_NAME}_saved_pdfs_{MISSING}.cp.pkl", 'wb') as outp:
            cloudpickle.dump(kde_models_dict,outp)
       
    try:
        with open(f"{DATASET_NAME}/data/{DATASET_NAME}_data_{MISSING}.pkl", 'wb') as outp:
            pickle.dump(X, outp, pickle.HIGHEST_PROTOCOL)
    except:
        with open(f"{DATASET_NAME}/data/{DATASET_NAME}_data_{MISSING}.cp.pkl", 'wb') as outp:
            cloudpickle.dump(X, outp)

    print("KDEs saved.")
