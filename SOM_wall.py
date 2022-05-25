import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

from basic_exploration_turbulence_from_h5_files import \
    get_instant_data_y_ct_from_h5_file
from my_som import SOM

'''
SOM for wall points, 3 input variables:
X1 = dudy_s  (velocity gradient, component dudy)
X2 = dwdy_s  (velocity gradient, component dwdy)
X3 = x_s    (x-coordinate)
'''

def build_dataframe(coord, vel_grad):
    x = coord[:,:,0].reshape(-1)
    dudy = vel_grad[:,:,1].reshape(-1)
    dwdy = vel_grad[:,:,7].reshape(-1)
    df = pd.DataFrame()
    df['x_s'] = x
    df['dudy_s'] = dudy
    df['dwdy_s'] = dwdy
    return df

def standarization(DF):
    mean_values = DF.mean()
    std_values = DF.std()
    DF_standarized = (DF-DF.mean())/DF.std()
    return DF_standarized, mean_values, std_values

def add_colorbar(fig,ax,im):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right',size='5%',pad=0.05)
    fig.colorbar(im,cax=cax,orientation='vertical')

def plot_predictions(df, m,n,sampling=5):
    df = df.iloc[np.arange(0,len(df),sampling)]      
    plt.figure(figsize=(15,6))
    plt.scatter(df['x_s'],df['z_s'],c=df['predictions'],s=10,marker=',')
    plt.xlim(df.iloc[0]['x_s'],df.iloc[-1]['x_s'])
    plt.ylim(df.iloc[0]['z_s'],df.iloc[-1]['z_s'])
    plt.xlabel('x/L')
    plt.ylabel('z/L')
    # Predictions separated by clusters
    if (m == 1 or n == 1): # 1D SOM grid
        fig, ax = plt.subplots(m*n,1,figsize=(14,7))
        for i in range(m*n):
            im = ax[i].scatter(df[df['predictions']==i]['x_s'],df[df['predictions']==i]['z_s'],c=df[df['predictions']==i]['predictions'],s=10,marker=',')
            ax[i].set_title('Cluster {}'.format(i))
            ax[i].set_xlim(df.iloc[0]['x_s'],df.iloc[-1]['x_s'])
            ax[i].set_ylim(df.iloc[0]['z_s'],df.iloc[-1]['z_s'])
            ax[i].set_xlabel('x/L')
            ax[i].set_ylabel('y/L')
        plt.suptitle('Clusters predicitons')
        plt.tight_layout()
    
def plot_distance_samples_to_cluster_centers(df, distance_samples_clusterCenters, m, n, sampling_space=5):
    df = df.iloc[np.arange(0,len(df),sampling_space)] 
    distance_samples_clusterCenters = distance_samples_clusterCenters[::sampling_space,:]
    total_samples = len(df)
    sampled_samples = np.arange(0,total_samples,sampling_space)
    print(sampled_samples)
    print(distance_samples_clusterCenters)
    if (m == 1 or n == 1): # 1D SOM grid
        fig, ax = plt.subplots(m*n,1,figsize=(14,7))
        for i in range(m*n):
            im = ax[i].scatter(df['x_s'],df['z_s'],c=distance_samples_clusterCenters[:,i],s=5,marker='p')
            ax[i].set_title('Cluster {}'.format(i))
            ax[i].set_xlim(df.iloc[0]['x_s'],df.iloc[-1]['x_s'])
            ax[i].set_ylim(df.iloc[0]['z_s'],df.iloc[-1]['z_s'])
            ax[i].set_xlabel('x/L')
            ax[i].set_ylabel('y/L')
            add_colorbar(fig,ax[i],im)
        plt.suptitle('Euclidean distance between samples and cluster centers')
        plt.tight_layout()
        fig, ax = plt.subplots(m*n,1,figsize=(14,7))
        for i in range(m*n):
            ax[i].plot(distance_samples_clusterCenters[:,i]) 
            ax[i].set_title('Cluster {}'.format(i))
            ax[i].set_xlabel('Training samples')
            ax[i].set_ylabel('Distance')
            ax[i].grid()
        plt.suptitle('Euclidean distance between samples and cluster centers')
        plt.tight_layout()  
    else: # 2D SOM grid
        fig, ax = plt.subplots(m,n)
        i = 0
        for i_m in range(m):
            for i_n in range(n):
                im = ax[i_m,i_n].scatter(df['x_s'],df['z_s'],c=distance_samples_clusterCenters[:,i],s=2)
                ax[i_m,i_n].set_title('Cluster {}'.format(i))
                ax[i_m,i_n].set_xlim(df.iloc[0]['x_s'],df.iloc[-1]['x_s'])
                ax[i_m,i_n].set_ylim(df.iloc[0]['z_s'],df.iloc[-1]['z_s'])
                ax[i_m,i_n].set_xlabel('x/L')
                ax[i_m,i_n].set_ylabel('y/L')
                add_colorbar(fig,ax[i_m,i_n],im)
                i += 1
        plt.suptitle('Euclidean distance between samples and cluster centers')
        plt.tight_layout()
        fig, ax = plt.subplots(m,n)
        i = 0
        for i_m in range(m):
            for i_n in range(n):
                ax[i_m,i_n].plot(distance_samples_clusterCenters[:,i]) 
                ax[i_m,i_n].set_title('Cluster {}'.format(i))
                ax[i_m,i_n].set_xlabel('Training samples')
                ax[i_m,i_n].set_ylabel('Distance')
                ax[i_m,i_n].grid()
                i += 1  
        plt.suptitle('Euclidean distance between samples and cluster centers')
        plt.tight_layout()
        
def plot_cluster_centers_history(cch,m,n,sampling_iter):
    total_iter = cch.shape[-1]
    cch = cch[:,:,:,::sampling_iter]
    sampled_iter = np.arange(0,total_iter,sampling_iter)
    if (n == 1): # 1D SOM grid
        for cl in range(m*n):
            fig, ax = plt.subplots(3,1,figsize=(10,8))
            ax[0].plot(sampled_iter,cch[cl,0,0,:])
            ax[0].set_title('weight x_s')
            ax[1].plot(sampled_iter,cch[cl,0,1,:])
            ax[1].set_title('weight du/dy_s')
            ax[2].plot(sampled_iter,cch[cl,0,2,:])
            ax[2].set_title('weight dw/dy_s')
            fig.suptitle('Center of Cluster {} vs SOM iterations'.format(cl))
            fig.tight_layout()

if __name__ == '__main__':
    
    # SOM grid shape:
    m = 2
    n = 1
    
    # Get dataset + dataset transformations:
    File_Name = "JHTDB_time_10-0_n_831x512_y_0-03.h5"
    Coord, _, Vel_Grad, Time, Y_Value = get_instant_data_y_ct_from_h5_file(File_Name)
    DF = build_dataframe(Coord,Vel_Grad)
    DF_stand, _, _ = standarization(DF)
    DF_stand_ndarray = DF_stand.to_numpy()
    
    # Create, fit and predict the SOM clustering algorithm
    som = SOM(m=m,n=n,dim=3,max_iter=3e6, random_state=1, lr=1, sigma = 1, sigma_evolution = 'constant')
    som.fit(DF_stand_ndarray,epochs=3,shuffle=True,save_cluster_centers_history=True)
    DF['z_s'] = Coord[:,:,2].reshape(-1)
    DF['predictions']=som.predict(DF_stand_ndarray)
    Distance_Samples_ClusterCenters = som.transform(DF_stand_ndarray)
    
    # Plot SOM clustering resuts:
    Sampling_Space = 5
    Sampling_Iter = 100
    plot_predictions(DF,m,n,Sampling_Space)
    plot_distance_samples_to_cluster_centers(DF, Distance_Samples_ClusterCenters, m, n, Sampling_Space) 
    plot_cluster_centers_history(som.cluster_centers_history, m, n, Sampling_Iter)
    
    plt.show()

