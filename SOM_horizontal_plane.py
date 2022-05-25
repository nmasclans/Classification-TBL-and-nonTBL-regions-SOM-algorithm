import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

from basic_exploration_turbulence_from_h5_files import \
    get_instant_data_y_ct_from_h5_file
from my_som import SOM
from SOM_wall import (add_colorbar, plot_distance_samples_to_cluster_centers,
                      plot_predictions)

'''
SOM for horizontal plane, 15 standarized features:
    (1-3) instantaneous velocity: u_s,  v_s, w_s
    (4-5) velocity fluctuations:  uf_s, vf_s
    (6-14) velocity gradients:    dudx_s, dudy_s, dudz_, dvdx_s, dvdy_s, dvdz_s, dwdx_s, dwdy_s, dwdz_s
    (15) x-coordinates:           x_s 
'''

def construct_ndarray_from_standarized_features_h5file(file_name):
    ''''
    Constructs the dataset of shape (N_samples, N_features) from a h5 file of 15 standarized features evaluated
    in an horizontal plane, for y_value and time chosen when creating the dataset.
    
    Parameters
    ----------
    file_name : str      : file name of the h5 file with the 15 standarized features of the 
                           simulation snapshot horizontal plane
                           i.e. JHTDB_standarized_features/JHTDB_time_10-0_n_831x512_y_0-03_standarized_features.h5  
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
        u_s = f['u_s'][:]
        v_s = f['v_s'][:]
        w_s = f['w_s'][:]
        uf_s = f['uf_s'][:]
        vf_s = f['vf_s'][:]
        dudx_s = f['dudx_s'][:]
        dudy_s = f['dudy_s'][:]
        dudz_s = f['dudz_s'][:]
        dvdx_s = f['dvdx_s'][:]
        dvdy_s = f['dvdy_s'][:]
        dvdz_s = f['dvdz_s'][:]
        dwdx_s = f['dwdx_s'][:]
        dwdy_s = f['dwdy_s'][:]
        dwdz_s = f['dwdz_s'][:]
        x_s = f['x_s'][:]
        z_s = f['z_s'][:]  
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

def plot_cluster_centers_history(cch,m,n,sampling_iter,feature_names):
    total_iter = cch.shape[-1]
    n_features = cch.shape[2]
    cch = cch[:,:,:,::sampling_iter]
    sampled_iter = np.arange(0,total_iter,sampling_iter)
    n_rows = 6
    n_cols = 3
    if (n == 1): # 1D SOM grid
        for cl in range(m*n):
            fig, ax = plt.subplots(n_rows,n_cols,figsize=(13,5))
            f = 0
            for i in range(n_rows):
                for j in range(n_cols):
                    ax[i,j].plot(sampled_iter,cch[cl,0,f,:])
                    ax[i,j].set_title(feature_names[f])
                    f += 1
                    if f == n_features:
                        break
            fig.suptitle('Center of Cluster {} vs SOM iterations'.format(cl))
            fig.tight_layout()

if __name__ == '__main__':
    
    # SOM grid shape:
    m = 2
    n = 1
    
    # Get dataset of 16 standarized features, as ndarray 'data'
    File_Name = "JHTDB_standarized_features/JHTDB_time_10-0_n_831x512_y_0-03_standarized_features.h5"
    Data, DF = construct_ndarray_from_standarized_features_h5file(File_Name)
    Feature_Names = DF.keys()
    
    # Remove x_s and z_s from clustering dataset, we will not use it for training the algorithm:
    n_features_reduced = 2
    Data_red = Data[:,:-n_features_reduced]
    Feature_Names_red = Feature_Names[:-n_features_reduced]
    
    # Create, fit and predict the SOM clustering algorithm
    som = SOM(m=m,n=n,dim=16-n_features_reduced,max_iter=2e6, random_state=1, lr=1, sigma = 1, sigma_evolution = 'exponential_decay')
    som.fit(Data_red,epochs=3,shuffle=True,save_cluster_centers_history=True)
    DF['predictions'] = som.predict(Data_red)
    Distance_Samples_ClusterCenters = som.transform(Data_red)
    Cluster_Centers_History = som.cluster_centers_history
    
    # Plot SOM clustering resuts:
    Sampling_Space = 5
    Sampling_Iter = 100
    plot_predictions(DF,m,n,Sampling_Space)
    plot_distance_samples_to_cluster_centers(DF, Distance_Samples_ClusterCenters, m, n, Sampling_Space) 
    plot_cluster_centers_history(Cluster_Centers_History, m, n, Sampling_Iter, Feature_Names_red)
    
    plt.show()

