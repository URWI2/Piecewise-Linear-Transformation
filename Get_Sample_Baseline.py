
import numpy as np
from distribution import Distribution
from Linear_Segments import loadModel, Cube, get_restricted_cpu
from Polytopes import CPU
import pickle
import cloudpickle

if __name__ == "__main__":
    
    DATASET_NAME = 'breast'
    MISSING = 25

    with open(f"{DATASET_NAME}/data/{DATASET_NAME}_data_{MISSING}.pkl",'rb') as handle:
        data=pickle.load(handle)
    with open(f"{DATASET_NAME}/data/{DATASET_NAME}_saved_pdfs_{MISSING}.cp.pkl",'rb') as handle:
        data_pdfs=cloudpickle.load(handle)
  
    #the desired thresholds for the convergence criterion
    thresholds={ins:0.01 for ins in list(data_pdfs.keys())}
   
    #bin numbers per dimension for start histogram 
    bins={ins:20 for ins in list(data_pdfs.keys())}

    for ins in list(data_pdfs.keys()):
    
        instance=ins
        certainty_vector = data.iloc[instance].values
        print(certainty_vector)
       
        #skip certain instances that do not require a baseline histogram 
        try:
            saved_pdf=data_pdfs[instance][0]
        except:
            continue 
        
        new_cv=list(np.zeros(certainty_vector.shape))
        for i in range(len(certainty_vector)):
            if np.isnan(certainty_vector[i])==True:
                new_cv[i] = None
            else:
                new_cv[i] = certainty_vector[i]
        
        certainty_vector=new_cv[:data.shape[1]]
        
       
        alpha = 0.0
      
        n = saved_pdf.d 
        model = loadModel(f"{DATASET_NAME}/model_{DATASET_NAME}.pt", alpha=alpha)
    
        cb = Cube(n)
        for i in range(n):
            cb.stretch(1.2*(data_pdfs[ins][0].dataset[i].max() - data_pdfs[ins][0].dataset[i].min()), i+1)
            cb.translate(1.1*data_pdfs[ins][0].dataset[i].min() - 0.1*data_pdfs[ins][0].dataset[i].max(), i+1)
           
    
        v = cb.points
        conn = cb.conn
        cpu_cb = CPU(v, conn)
    
        res_cpu = get_restricted_cpu(
            model, cpu=cpu_cb, cert_vec=certainty_vector, alpha=alpha)
    
        
        distribution = Distribution(res_cpu, pdf = saved_pdf, model = model)
        
        
        baseline = distribution.find_baseline_histo(cert_vec = certainty_vector, 
                                                    start_params= (10000 , bins[ins]), samp_nr_mult = 2, 
                                                    bin_nr_mult = 1, threshold = thresholds[ins] )
        
        histo = baseline[1]
        
        
            
            
        with open(f"{DATASET_NAME}/histos/baseline_{DATASET_NAME}_{MISSING}_{ins}.pkl", 'wb') as handle:
            pickle.dump(histo, handle)

