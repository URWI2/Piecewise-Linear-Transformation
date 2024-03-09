
import numpy as np
import copy
from distribution import Distribution, comparePDFs, hellinger, propagate_adf, global_method, make_histo_grid, fit_into_uncert_vec
from Linear_Segments import loadModel, Cube, get_restricted_cpu
from Polytopes import CPU
import pickle
from utils.PropagationMethods import ADF
from utils.adf import ReLU as ADFReLU
from scipy.stats import multivariate_normal
from utils.as_model import NNModel
from utils.as_dataset import Dataset
from utils.as_subspace import NNSubspace
import cloudpickle
from statistics import mean 
from filterpy.kalman import MerweScaledSigmaPoints, unscented_transform
import scipy


if __name__ == "__main__":

    ########################### PREPARATIONS #################################
    
    DATASET_NAME = 'housing'
    MISSING = 25

    with open(f"{DATASET_NAME}/data/{DATASET_NAME}_data_{MISSING}.pkl",'rb') as handle:
        data=pickle.load(handle)
    with open(f"{DATASET_NAME}/data/{DATASET_NAME}_saved_pdfs_{MISSING}.cp.pkl",'rb') as handle:
        data_pdfs=cloudpickle.load(handle)
    
    with open(f"{DATASET_NAME}/data/{DATASET_NAME}_data_test.npy",'rb') as handle:
        test_data_ = np.load(handle)
        y_true = np.load(handle)
        
    results=[]
    our_pdfs={}
    as_pdfs={}
    ut_pdfs={}
    adf_pdfs={}
    
    #parameters k for edgewise subdivision per uncertain input dimension 
    subdivs={1: 250, 2: 100, 3: 30, 4: 2, 5: 1, 6: 1, 7: 1, 8: 1}
  
    #parameter alpha for LeakyReLu function
    alpha = 0.0
   
    
  
    ############################### EVALUATION OF ALL METHODS #####################################
    for j,ins in enumerate(list(data_pdfs.keys())):
        
        print("Instance", ins)
        
        if type(data_pdfs[ins]) is not tuple:
            model = loadModel(f"{DATASET_NAME}/model_{DATASET_NAME}.pt", alpha=alpha)
            pred = model.propagate(data.to_numpy()[ins].reshape(1,-1))[0]
            continue
        
        with open(f"{DATASET_NAME}/histos/baseline_{DATASET_NAME}_{MISSING}_{ins}.pkl",'rb') as handle:
            baseline_histo=pickle.load(handle)
        histo=baseline_histo
        result=[]
        
        #sampling baseline histogram 
        sampling_pdf = histo[0]
        
        instance=ins
        certainty_vector = data.iloc[instance].values
        
        saved_pdf=data_pdfs[instance][0]
        
        new_cv=list(np.zeros(certainty_vector.shape))
        for i in range(len(certainty_vector)):
            if np.isnan(certainty_vector[i])==True:
                new_cv[i] = None
            else:
                new_cv[i] = certainty_vector[i]
        
        certainty_vector=new_cv
    
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
    
        ################# SUBDIVISION AND LINSEG PREPARATIONS #####################
    
        res_cpu.subdivide(k = subdivs[n], epsilon=None, mode="delaunay")
      
        res_cpu.prep_lin_seg_for_eval()
    
        distribution = Distribution(res_cpu, pdf = saved_pdf, model = model)
    
        #calculate bin volume
        vol_bin = 1  
        for i in range(0, len(histo[1])):
            vol_bin = vol_bin*(histo[1][i][1]-histo[1][i][0])
    
                
        ############################# OUR METHOD #################################

      
        our_pdf = global_method(distribution, model, histo, cert_vec=certainty_vector)    
      
        our_pdfs[j]=our_pdf
        
        ############################### ADF ######################################
    
        histo_grid = copy.deepcopy(histo[1])
    
        for i in range(len(histo[1])):
            histo_len_i = (histo[1][i][1]-histo[1][i][0])/2
            histo_grid[i] = histo_grid[i]-histo_len_i
            histo_grid[i] = histo_grid[i][1:]
    
        histo_grid_points = make_histo_grid(histo_grid)
    
        means = copy.deepcopy(certainty_vector)
        vars_vect = np.zeros(len(certainty_vector))
        kde_ct = 0
        for i, x in enumerate(certainty_vector):
            if x == None:

                means[i] = data_pdfs[j][0].dataset[kde_ct].mean()
                if np.min(data_pdfs[j][0].dataset) == np.max(data_pdfs[j][0].dataset):
                    vars_vect[i] = 0.01
                else:
                    vars_vect[i]=data_pdfs[j][0].dataset[kde_ct].var()
                kde_ct +=1
            else:
                vars_vect[i] = 0.0
            
        adf = ADF(means=means, vars=vars_vect)
        adf.act = ADFReLU().forward
        outs_adf = propagate_adf(adf, model)
        cov_matrix = np.zeros((len(outs_adf[1]), len(outs_adf[1])))
        for i in range(len(outs_adf[1])):
            cov_matrix[i, i] = outs_adf[1][i]
        nv_adf = multivariate_normal(mean=outs_adf[0], cov=cov_matrix, allow_singular=True)
        
        adf_output = nv_adf.pdf(histo_grid_points)
        adf_pdfs[j]= adf_output

        ########################### UT ###########################################
       
        mean_list = []
        var_list = []
        kde_ct = 0
        for i, val in enumerate(certainty_vector):
            if val == None:
                mean_list.append(data_pdfs[j][0].dataset[kde_ct].mean())
                var_list.append(data_pdfs[j][0].dataset[kde_ct].var())
                kde_ct +=1
            
        sigmas = MerweScaledSigmaPoints(n=n, alpha=.3, beta=2., kappa=.1)
        points = sigmas.sigma_points(x=mean_list,P=np.diag(var_list))#Aus var list eine Matrix machen?
        points = [fit_into_uncert_vec(x=x, cert_vec=certainty_vector) for x in points]
        points = np.array(model.propagate(points))
        ukf_mean, ukf_cov = unscented_transform(points, sigmas.Wm, sigmas.Wc)  
        ukf_pdf= scipy.stats.multivariate_normal(mean=ukf_mean, cov=ukf_cov, allow_singular=True).pdf
        
        ut_output = ukf_pdf(histo_grid_points)
        ut_pdfs[j]=ut_output

        ##################### ACTIVE SUBSPACE ####################################
        AS_dataset = Dataset(dataset_name = DATASET_NAME, missing = MISSING)
        AS_model = NNModel(dataset_name = DATASET_NAME, model_id='0')
        
        AS = NNSubspace(model = AS_model,
                        x = AS_dataset.x_test[ins],
                        inst_idx = ins,
                        x_train_mean = AS_dataset.x_train_mean,
                        dataset_name = DATASET_NAME,
                        dataset = AS_dataset)
    
    
        AS.sampling_setup(num_gradient_mc=1000,
                      num_rs_mc=1000,
                      seed=7,
                      bool_clip=False,
                      sigma=0.7,
                      num_eigenvalue = int(AS_dataset.img_rows/2))
        AS.run()
    
        as_output_samps = np.zeros((AS.y_samples_rs.shape))
        for d, dim_poly in enumerate(AS.poly1d):
            as_output_samps[:,d] = dim_poly(AS.xv[:,0])
        as_output = np.histogramdd(as_output_samps, bins = histo[1], density = True)[0]
        as_pdfs[j]=as_output
        
        
        prob_mass = np.sum(our_pdf)*(vol_bin)
    
        print("Prob Mass Output (PLT):", prob_mass)
        
        norm_1 = comparePDFs(pdfs=histo[0], our_pdfs=our_pdf, p=1, width=vol_bin)
        norm_2 = comparePDFs(pdfs=histo[0], our_pdfs=our_pdf, p=2, width=vol_bin)
        norm_5 = hellinger(pdfs=histo[0], our_pdfs=our_pdf, width=vol_bin)
       
        
        print()
        print("Norms for PLT")
        print("Estimation of L1-norm:", norm_1)
        print("Estimation of L2-norm:", norm_2)
        print("Estimation of Hellinger distance:", norm_5)
    
        norm_3 = comparePDFs(
            pdfs=histo[0], our_pdfs=adf_output, p=1, width=vol_bin)
        norm_4 = comparePDFs(
            pdfs=histo[0], our_pdfs=adf_output, p=2, width=vol_bin)
        norm_6 = hellinger(pdfs=histo[0], our_pdfs=adf_output, width=vol_bin)
        
        
        print()
        print("Norms for ADF")
        print("Estimation of L1-norm:", norm_3)
        print("Estimation of L2-norm:", norm_4)
        print("Estimation of Hellinger distance:", norm_6)
       
        norm_1_as = comparePDFs(pdfs = histo[0], our_pdfs = as_output, p = 1, width = vol_bin)
        norm_2_as = comparePDFs(pdfs = histo[0], our_pdfs = as_output, p = 2, width = vol_bin)
        
        print()
        print("Norms for AS")
        print("Estimation of L1-norm:", norm_1_as)
        print("Estimation of L2-norm:", norm_2_as)
        print("Estimation of Hellinger distance:", hellinger(pdfs = histo[0], our_pdfs = as_output, width = vol_bin))
        
        hellinger_as = (hellinger(pdfs = histo[0], our_pdfs = as_output, width = vol_bin))
        
        norm_1_ut = comparePDFs(pdfs = histo[0], our_pdfs = ut_output, p = 1, width = vol_bin)
        norm_2_ut = comparePDFs(pdfs = histo[0], our_pdfs = ut_output, p = 2, width = vol_bin)
        hellinger_ut = hellinger(pdfs = histo[0], our_pdfs = ut_output, width = vol_bin)
        
        print()
        print("Norms for UT")
        print("Estimation of L1-norm:", norm_1_ut)
        print("Estimation of L2-norm:", norm_2_ut)
        print("Estimation of Hellinger distance:", hellinger_ut)
        print()       
        
        result.append((prob_mass, norm_1, norm_2, norm_5, norm_3, norm_4, norm_6, norm_1_as, norm_2_as, hellinger_as, 
                       norm_1_ut, norm_2_ut, hellinger_ut))
        
        result.append(ins)
        results.append(result)
        
    with open(f"{DATASET_NAME}/results/save_run_{DATASET_NAME}_{MISSING}.pkl", 'wb') as handle:
        pickle.dump(results, handle)
    
   
    #Calculate the mean values of the metrics across all uncertain instances  
    print("Total results")
    print("Norms for PLT")
    print("Prob mass", mean([res[0][0] for res in results]))
    print("Estimation of L1-norm:", mean([res[0][1] for res in results]))
    print("Estimation of L2-norm:", mean([res[0][2] for res in results]))
    print("Estimation of Hellinger distance:", mean([res[0][3] for res in results]))
   
    print("Norms for ADF")
    print("Estimation of L1-norm:", mean([res[0][4] for res in results]))
    print("Estimation of L2-norm:", mean([res[0][5] for res in results]))
    print("Estimation of Hellinger distance:", mean([res[0][6] for res in results]))
   
    
    print("Norms for AS")
    print("Estimation of L1-norm:", mean([res[0][7] for res in results]))
    print("Estimation of L2-norm:", mean([res[0][8] for res in results]))
    print("Estimation of Hellinger distance:", mean([res[0][9] for res in results]))

    
    print("Norms for UT")
    print("Estimation of L1-norm:", mean([res[0][10] for res in results]))
    print("Estimation of L2-norm:", mean([res[0][11] for res in results]))
    print("Estimation of Hellinger distance:", mean([res[0][12] for res in results]))
   

    with open(f"{DATASET_NAME}/results/our_pdf_results{DATASET_NAME}_{MISSING}.pkl", 'wb') as handle:
        pickle.dump(our_pdfs, handle)
    
    with open(f"{DATASET_NAME}/results/adf_pdf_results{DATASET_NAME}_{MISSING}.pkl", 'wb') as handle:
        pickle.dump(adf_pdfs, handle)
    with open(f"{DATASET_NAME}/results/as_pdf_results{DATASET_NAME}_{MISSING}.pkl", 'wb') as handle:
        pickle.dump(as_pdfs, handle)
    with open(f"{DATASET_NAME}/results/ut_pdf_results{DATASET_NAME}_{MISSING}.pkl", 'wb') as handle:
        pickle.dump(ut_pdfs, handle)
    
    