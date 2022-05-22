import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

from basic_exploration_turbulence_from_h5_files import \
    get_instant_data_y_ct_from_h5_file
from my_som import SOM

'''
SOM for wall points, 3 input variables:
X1 = dudy  (velocity gradient, component dudy)
X2 = dwdy  (velocity gradient, component dwdy)
X3 = xs    (x-coordinate)
'''

def build_dataframe(coord, vel_grad):
    xs = coord[:,:,0].reshape(-1)
    dudy = vel_grad[:,:,1].reshape(-1)
    dwdy = vel_grad[:,:,7].reshape(-1)
    df = pd.DataFrame()
    df['xs'] = xs
    df['dudy'] = dudy
    df['dwdy'] = dwdy
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

def plot_cluster_centers_history(cch):
    plt.figure()
    plt.plot(cch[0,0,0,:],label='weight xs')
    plt.plot(cch[0,0,1,:],label='weight dudy')
    plt.plot(cch[0,0,2,:],label='weight dwdy')
    plt.title('Center of Cluster 0 vs SOM iterations')
    plt.legend()
    plt.figure()
    plt.plot(cch[1,0,0,:],label='weight xs')
    plt.plot(cch[1,0,1,:],label='weight dudy')
    plt.plot(cch[1,0,2,:],label='weight dwdy')
    plt.title('Center of Cluster 1 vs SOM iterations')
    plt.legend()

if __name__ == '__main__':
    
    File_Name = "JHTDB_time_10-0_n_831x512_y_0-0.h5"
    Coord, _, Vel_Grad, Time, Y_Value = get_instant_data_y_ct_from_h5_file(File_Name)
    DF = build_dataframe(Coord,Vel_Grad)
    DF_stand, _, _ = standarization(DF)
    DF_stand_ndarray = DF_stand.to_numpy()
    m = 3
    n = 3
    som = SOM(m=m,n=n,dim=3,max_iter=512,random_state=1,sigma=1, lr=1)
    som.fit(DF_stand_ndarray,epochs=1,shuffle=True,save_cluster_centers_history=True)
    
    DF['zs'] = Coord[:,:,2].reshape(-1)
    DF['predictions']=som.predict(DF_stand_ndarray)
    distance_samples_to_each_cluster_center = som.transform(DF_stand_ndarray)
    
    # Reduced dataset for ploting:
    sampling = 10
    DF_reduced = DF.iloc[np.arange(0,len(DF),sampling)]  
    plt.figure()
    plt.scatter(DF_reduced['xs'],DF_reduced['zs'],c=DF_reduced['predictions'])
    
    # Distance to each cluster center
    dist_cluster_center = {}
    i = 0
    for i in range(m*n):
        dist_cluster_center[i] = distance_samples_to_each_cluster_center[::sampling,i]
        
    fig, ax = plt.subplots(m,n)
    i = 0
    for i_m in range(m):
        for i_n in range(n):
            im = ax[i_m,i_n].scatter(DF_reduced['xs'],DF_reduced['zs'],c=dist_cluster_center[i])
            ax[i_m,i_n].set_title('Euclidean Distance to Center of Cluster {}'.format(i))
            add_colorbar(fig,ax[i_m,i_n],im)
            i += 1
    
    fig, ax = plt.subplots(m,n)
    i = 0
    for i_m in range(m):
        for i_n in range(n):
            ax[i_m,i_n].plot(dist_cluster_center[i]) 
            i += 1  
    
    plot_cluster_centers_history(som.cluster_centers_history)
    
    plt.show()

