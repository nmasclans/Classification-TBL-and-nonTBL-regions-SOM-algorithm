import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn_som.som import SOM

from basic_exploration_turbulence_from_h5_files import \
    get_instant_data_y_ct_from_h5_file

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

if __name__ == '__main__':
    
    File_Name = "JHTDB_time_10-0_n_831x512_y_0-0.h5"
    Coord, _, Vel_Grad, Time, Y_Value = get_instant_data_y_ct_from_h5_file(File_Name)
    DF = build_dataframe(Coord,Vel_Grad)
    DF_stand, _, _ = standarization(DF)
    DF_stand_ndarray = DF_stand.to_numpy()
    som = SOM(m=2,n=1,dim=3,max_iter=1000,random_state=1)
    som.fit(DF_stand_ndarray,epochs=50,shuffle=True)
    
    DF['zs'] = Coord[:,:,2].reshape(-1)
    DF['predictions']=som.predict(DF_stand_ndarray)
    
    # Reduced dataset for ploting:
    sampling = 5
    DF_reduced = DF.iloc[np.arange(0,len(DF),sampling)]  
    plt.figure()
    plt.scatter(DF_reduced['xs'],DF_reduced['zs'],c=DF_reduced['predictions'])
    plt.show()

