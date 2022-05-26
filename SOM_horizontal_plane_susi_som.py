import itertools

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from basic_exploration_turbulence_from_h5_files import \
    get_instant_data_y_ct_from_h5_file
from SOM_wall import plot_predictions
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

    

if __name__ == '__main__':
    
    m = 5
    n = 5
    
    # Get dataset of 16 standarized features, as ndarray 'data'
    File_Name = "JHTDB_standarized_features/JHTDB_time_10-0_n_831x512_y_0-03_standarized_features.h5"
    Data, DF = construct_ndarray_from_standarized_features_h5file(File_Name, sampling = 10)
    Feature_Names = DF.keys()    
    
    # Remove x_s and z_s from clustering dataset, we will not use it for training the algorithm:
    n_features_reduced = 2
    Data_red = Data[:,:-n_features_reduced]
    Feature_Names_red = Feature_Names[:-n_features_reduced]
    
    # Create and fit the SOM clustering algorithm
    som = SOMClustering(n_rows = m, 
                        n_columns = n, 
                        init_mode_unsupervised = 'random_data', 
                        n_iter_unsupervised = 10000, 
                        random_state = 1, 
                        verbose = True)
    som.fit(Data_red)
    
    # Predict dataset (assign a cluster to each datapoint)
    Data_Red_transformed = som.transform(Data_red)
    DF['predictions'] = Data_Red_transformed[:,0]*10+Data_Red_transformed[:,1]
    plot_predictions(DF,m,n,sampling=1,marker_size=2)
    
    # Best Matching clusters
    bmu_list = som.get_bmus(Data_red)
    plt.figure()
    plt.hist2d([x[0] for x in bmu_list], [x[1] for x in bmu_list],bins=[m,n],range = [[-0.5,m-0.5],[-0.5,n-0.5]])

    # u-matrix
    umat = som.get_u_matrix()
    plt.imshow(np.squeeze(umat))
    
    plt.show()   
    
    # Parametric study of SOM
    param_grid = {"n_rows": [5, 10, 20],
                  "learning_rate_start": [0.5, 0.7, 0.9],
                  "learning_rate_end": [0.1, 0.05, 0.005]}    
    dict_param = {} 
    dict_umat = {}
    for i,j,k in itertools.product(range(3),repeat=3):
        n_rows = param_grid['n_rows'][i]
        n_columns = n_rows
        learning_rate_start = param_grid['learning_rate_start'][j]
        learning_rate_end = param_grid['learning_rate_end'][k]
        print('\nSOM Clustering for: n_rows = {}, n_columns = {}, learning_rate_start = {} and learning_rate_end = {}\n'
              .format(n_rows, n_columns, learning_rate_start, learning_rate_end))
        som = SOMClustering(n_rows = n_rows, 
                            n_columns = n_columns, 
                            learning_rate_start=learning_rate_start,
                            learning_rate_end=learning_rate_end,
                            init_mode_unsupervised = 'random_data', 
                            n_iter_unsupervised = 1000, 
                            random_state = 1, 
                            verbose = True)
        som.fit(Data_red)
        dict_param[tuple((i,j,k))] = [n_rows, learning_rate_start, learning_rate_end]
        dict_umat[tuple((i,j,k))] = som.get_u_matrix()
    
    plt.figure()
    plt.imshow(np.squeeze(umat))
    
    for i,j,k in itertools.product(range(3),repeat=3):
        param = dict_param[tuple((i,j,k))]
        umat = dict_umat[tuple((i,j,k))]
        plt.figure()
        plt.imshow(np.squeeze(umat))
        str = 'n_rows = n_columns = {}, learning_rate_start = {}, learning_rate_end = {}'.format(param[0],param[1],param[2])
        plt.title(str)
    
    '''
    som = SOM(m=m,n=n,dim=16-n_features_reduced,max_iter=25000, random_state=1, lr=1, sigma = 1, sigma_evolution = 'exponential_decay')
    som.fit(Data_red,epochs=1,shuffle=True,save_param_history=True)
    DF['predictions'] = som.predict(Data_red)
    Distance_Samples_ClusterCenters = som.transform(Data_red,data_type='float16')
    Cluster_Centers_History = som.cluster_centers_history
    
    # Plot SOM clustering resuts:
    Sampling_Space = 2
    Sampling_Iter = 100
    plot_predictions(DF,m,n,Sampling_Space,marker_size=2)
    plot_distance_samples_to_cluster_centers(DF, Distance_Samples_ClusterCenters, m, n, Sampling_Space) 
    plot_cluster_centers_history(Cluster_Centers_History, m, n, Sampling_Iter, Feature_Names_red)
    
    plt.show()
    '''
