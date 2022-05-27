import itertools

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

from basic_exploration_turbulence_from_h5_files import \
    get_instant_data_y_ct_from_h5_file
from SOM_wall import add_colorbar, plot_predictions
from susi_som import SOMClustering

'''
SOM for horizontal plane, 15 standarized features:
    (1-3) instantaneous velocity: u_s,  v_s, w_s
    (4-5) velocity fluctuations:  uf_s, vf_s
    (6-14) velocity gradients:    dudx_s, dudy_s, dudz_, dvdx_s, dvdy_s, dvdz_s, dwdx_s, dwdy_s, dwdz_s
    (15) x-coordinates:           x_s 
'''

def construct_ndarray_from_standarized_features_h5file(file_name, sampling):
    ''''
    Constructs the dataset of shape (N_samples, N_features) from a h5 file of 15 standarized features evaluated
    in an horizontal plane, for y_value and time chosen when creating the dataset.
    
    Parameters
    ----------
    file_name : str      : file name of the h5 file with the 15 standarized features of the 
                           simulation snapshot horizontal plane
                           i.e. JHTDB_standarized_features/JHTDB_time_10-0_n_831x512_y_0-03_standarized_features.h5
    sampling : int       : sampling on the original simulation data
                           i.e. sampling = 2 takes half of the data, by steps of 2 [::2]
    
    Values
    ------
    dataset : np.ndarray(N_samples,N_features) : dataset of standarized features, as np.ndarray
                                                 i.e. N_samples = 425472 (for Nx = 831 and Nz = 512) and
                                                 N_features = 16 for horizontal plane
                                                 *** maybe x_s and specially z_s will not be used for SOM clustering!
    df      : pd.DataFrame(n_keys = 16)        : contains the standarized features, with their names as .keys().
                                                 same content as dataset, DataFrame created for visualization and interpretability
    '''
    with h5py.File(file_name, "r") as f:
        u_s = f['u_s'][::sampling]
        v_s = f['v_s'][::sampling]
        w_s = f['w_s'][::sampling]
        uf_s = f['uf_s'][::sampling]
        vf_s = f['vf_s'][::sampling]
        dudx_s = f['dudx_s'][::sampling]
        dudy_s = f['dudy_s'][::sampling]
        dudz_s = f['dudz_s'][::sampling]
        dvdx_s = f['dvdx_s'][::sampling]
        dvdy_s = f['dvdy_s'][::sampling]
        dvdz_s = f['dvdz_s'][::sampling]
        dwdx_s = f['dwdx_s'][::sampling]
        dwdy_s = f['dwdy_s'][::sampling]
        dwdz_s = f['dwdz_s'][::sampling]
        x_s = f['x_s'][::sampling]
        z_s = f['z_s'][::sampling] 
    # Dataset for clustering: np.ndarray
    dataset = np.array([u_s, v_s, w_s, uf_s, vf_s, 
                        dudx_s, dudy_s, dudz_s, dvdx_s, dvdy_s, dvdz_s, dwdx_s, dwdy_s, dwdz_s, x_s, z_s])
    dataset = np.swapaxes(dataset,0,1)
    # Dataframe for visualization purposes: pandas.DataFrame object
    df = pd.DataFrame()
    df['u_s'] = u_s
    df['v_s'] = v_s
    df['w_s'] = w_s
    df['uf_s'] = uf_s
    df['vf_s'] = vf_s
    df['dudx_s'] = dudx_s
    df['dudy_s'] = dudy_s
    df['dudz_s'] = dudz_s
    df['dvdx_s'] = dvdx_s
    df['dvdy_s'] = dvdy_s
    df['dvdz_s'] = dvdz_s
    df['dwdx_s'] = dwdx_s
    df['dwdy_s'] = dwdy_s
    df['dwdz_s'] = dwdz_s
    df['x_s'] = x_s
    df['z_s'] = z_s
    return dataset, df

def single_SOMClustering(data,df,m,n,lr_start,lr_end,n_iter,
                         bmu_umatrix_storing=False, bmu_umatrix_storing_frequency=None, 
                         make_final_plots=True):
    
    # Create and fit the SOM clustering algorithm
    som = SOMClustering(n_rows = m, 
                        n_columns = n, 
                        init_mode_unsupervised = 'random_data', 
                        n_iter_unsupervised = n_iter, 
                        learning_rate_start = lr_start,
                        learning_rate_end = lr_end,
                        random_state = 1, 
                        verbose = True,
                        bmu_umatrix_storing=bmu_umatrix_storing,
                        bmu_umatrix_storing_frequency= bmu_umatrix_storing_frequency)
    som.fit(data)
    
    if make_final_plots:
        # Predict dataset (assign a cluster to each datapoint)
        data_transformed = som.transform(data)
        df['predictions'] = data_transformed[:,0]*10+data_transformed[:,1]
        plot_predictions(df,m,n,sampling=1,marker_size=2)
        # Best Matching Units - times a neuron is the winning neuron
        bmus = som.get_bmus(data)
        # U-Matrix
        umatrix = som.get_u_matrix()
        plot_bmu_and_umatrix(bmus,umatrix,m,n)
        
    if bmu_umatrix_storing:
        for key in som.bmu_history:
            num_iter_str = '# iteration: {}'.format(key*bmu_umatrix_storing_frequency)
            plot_bmu_and_umatrix(bmus=som.bmu_history[key],
                                 umatrix=som.umatrix_history[key], m=m, n=n,
                                 subtitles=['BMU at ' + num_iter_str + '\n',
                                            'U-matrix at ' + num_iter_str + '\n'])        
    
    return som, df

def perform_hac(som,data,m,n,num_merges = 10):
    w_cl_som = som.unsuper_som_
    w_cl = w_cl_som.reshape(m*n,-1)
    bmus = som.get_bmus(data)
    n_cl_som = np.zeros((m,n),dtype='int')
    for data_point in range(len(bmus)):
        n_cl_som[bmus[data_point][0],bmus[data_point][1]] += 1 
    n_cl = n_cl_som.reshape(m*n,-1)
    
    # index associated to each cluster. new index when 2 clusters are merged.
    cl_id = np.zeros((m*n,num_merges+1),dtype='int')
    cl_id[:,0] = np.arange(m*n)
    merge_id = m*n
    
    for merge_iter in range(num_merges):
        d_wc = np.zeros((m*n-merge_iter,m*n-merge_iter))
        for r_mn in range(m*n-merge_iter):
            for s_mn in range(m*n-merge_iter):
                d_wc[r_mn,s_mn] = ((n_cl[r_mn]*n_cl[s_mn])/(n_cl[r_mn]+n_cl[s_mn]))*(np.linalg.norm(w_cl[r_mn,:]-w_cl[s_mn,:])**2)
            d_wc[r_mn,r_mn] = 1000
        ind_min = np.where(d_wc == np.amin(d_wc))[0]
        ind_r = ind_min[0]
        ind_s = ind_min[1]
        n_r = n_cl[ind_min[0]]
        n_s = n_cl[ind_min[1]]
        n_new_rs = n_r+n_s
        w_new_rs = (1/(n_r+n_s))*(n_r*w_cl[ind_r,:]+n_s*w_cl[ind_s,:])
        n_cl[ind_r] = n_new_rs
        n_cl = np.delete(n_cl,ind_s)
        w_cl[ind_r,:] = w_new_rs
        w_cl = np.delete(w_cl,ind_s,axis=0)

        cl_id[[ind_r,ind_s],merge_iter+1]=merge_id
        print('Merge iter #{}: merge_id = {} of clusters = ({},{}) at closest distance = {:.3f}'.format(merge_iter,merge_id,ind_r,ind_s,np.amin(d_wc)))
        merge_id += 1

    
    
    
def plot_bmu_and_umatrix(bmus,umatrix,m,n,subtitles=['BMU','U-matrix']):
    fig, ax = plt.subplots(1,2,figsize=(12,6))
    im = ax[0].hist2d(y = [x[0] for x in bmus], x = [x[1] for x in bmus],bins=[n,m],range = [[-0.5,n-0.5],[-0.5,m-0.5]])
    add_colorbar(fig,ax[0],im[3])
    ax[0].invert_yaxis()
    ax[0].set_xticks(np.arange(0,n,5))
    ax[0].set_yticks(np.arange(0,m,5))
    im = ax[1].imshow(np.squeeze(umatrix))
    add_colorbar(fig,ax[1],im)
    ax[1].set_xticks(np.arange(0,2*n,2))
    ax[1].set_xticklabels(np.arange(0,n))
    ax[1].set_yticks(np.arange(0,2*m,2))
    ax[1].set_yticklabels(np.arange(0,m))
    for i in range(2):
        ax[i].set_title(subtitles[i])
        ax[i].xaxis.tick_top()
        ax[i].axis('equal')
        ax[i].set_xlabel('n-columns')
        ax[i].set_ylabel('m-rows')
    fig.tight_layout()

def parametric_study_SOMClustering(N_rows, Learning_rate_start, Learning_rate_end, N_iter_unsupervised):
    num_of_trainings = len(N_rows)*len(Learning_rate_start)*len(Learning_rate_end)*len(N_iter_unsupervised)
    print('\n---------------------------------------------------------------------------')
    print('Parametric study: {} trainings'.format(num_of_trainings))
    print('---------------------------------------------------------------------------')
    param_dict = {}
    umat_dict = {}
    i = 0
    for n_rows in N_rows:
        for learning_rate_start in Learning_rate_start:
            for learning_rate_end in Learning_rate_end:
                for n_iter_unsupervised in N_iter_unsupervised:
                    param_dict[i] = 'n_rows = n_columns = {}, lr_start = {}, lr_end = {}, # iterations = {}'.format(
                        n_rows, learning_rate_start, learning_rate_end,n_iter_unsupervised)
                    print('\nTraining #{}: '.format(i) + param_dict[i] + '\n')
                    som = SOMClustering(n_rows = n_rows, 
                                        n_columns = n_rows, 
                                        learning_rate_start=learning_rate_start,
                                        learning_rate_end=learning_rate_end,
                                        init_mode_unsupervised = 'random_data', 
                                        n_iter_unsupervised = n_iter_unsupervised, 
                                        random_state = 1, 
                                        verbose = True)
                    som.fit(Data_red)
                    umat_dict[i] = som.get_u_matrix()
                    i+=1
                    
    # Make plots parametric study
    for i in range(num_of_trainings):
        plt.figure()
        plt.imshow(np.squeeze(umat_dict[i]))
        plt.title(param_dict[i])

if __name__ == '__main__':
    
    # Get dataset of 16 standarized features, as ndarray 'data'
    File_Name = "JHTDB_standarized_features/JHTDB_time_10-0_n_831x512_y_0-03_standarized_features.h5"
    Data, DF = construct_ndarray_from_standarized_features_h5file(File_Name, sampling = 5)
    Feature_Names = DF.keys()    
    
    # Remove x_s and z_s from clustering dataset, we will not use it for training the algorithm:
    n_features_reduced = 2
    Data_red = Data[:,:-n_features_reduced]
    Feature_Names_red = Feature_Names[:-n_features_reduced]
    
    # Single training and analysis of SOM 
    SOM, DF = single_SOMClustering(data = Data_red, df=DF, m=20, n=20, 
                                   lr_start=0.5, lr_end= 0.01, n_iter=1001,
                                   bmu_umatrix_storing=False, bmu_umatrix_storing_frequency=25000, 
                                   make_final_plots=False)
    # plt.show()
    
    # Parametric study of SOM
    parametric_study_SOMClustering(N_rows = [25],
                                   Learning_rate_start = [0.9],
                                   Learning_rate_end = [0.05],
                                   N_iter_unsupervised = [50000,100000,150000]) 
    plt.show() 
    
    # Quantification error:
    QE = SOM.get_quantization_error()

