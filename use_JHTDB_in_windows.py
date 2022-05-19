# pyJHTDB are failed to compile on windows. One alternative way might be to use zeep package.
import h5py
import numpy as np
import zeep

# Query web servies details:
client = zeep.Client('http://turbulence.pha.jhu.edu/service/turbulence.asmx?WSDL')
ArrayOfFloat = client.get_type('ns0:ArrayOfFloat')
ArrayOfArrayOfFloat = client.get_type('ns0:ArrayOfArrayOfFloat')
SpatialInterpolation = client.get_type('ns0:SpatialInterpolation')
TemporalInterpolation = client.get_type('ns0:TemporalInterpolation')
Token = 'edu.upc.nuria.masclans-51109d6e'

def query_data_velocity(time, points):
    # Spatial interpolation = 4, Lag4, 4 point Lagrange
    # Temporal interpolation: None, no time interpoltion
    result=client.service.GetData_Python('GetVelocity', Token,"transition_bl", time, 
                                     SpatialInterpolation("Lag4"), TemporalInterpolation("None"), points) 
    return result

def query_data_velocity_gradient(time, points):
    # Spatial interpolation = None_Fd4 or Fd4Lag4
    # Temporal interpolation: None, no time interpoltion
    result=client.service.GetData_Python('GetVelocityGradient', Token,"transition_bl", time, 
                                     SpatialInterpolation("Fd4Lag4"), TemporalInterpolation("None"), points)
    return result

def get_velocity_data(x_min=30.2185, x_max=800, y_min=0, y_max=26.488, z_min=0, z_max=240, nx=831, ny=280, nz=512, time=0.25):
    # Time and spatial coordinates
    x = np.linspace(x_min, x_max, nx, dtype = 'float32')
    y = np.linspace(y_min, y_max, ny, dtype = 'float32')
    z = np.linspace(z_min, z_max, nz, dtype = 'float32')
    coord = np.empty((nx, ny, nz, 3), dtype = 'float32') # done for having all values at same resolution
    coord[:, :, :, 0] = x[:, np.newaxis, np.newaxis]
    coord[:, :, :, 1] = y[np.newaxis, :, np.newaxis]
    coord[:, :, :, 2] = z[np.newaxis, np.newaxis, :]
    
    # Convert to JHTDB structures (for web service query)
    x_coord=ArrayOfFloat(coord[:,:,:,0].reshape(-1).tolist())
    y_coord=ArrayOfFloat(coord[:,:,:,1].reshape(-1).tolist())
    z_coord=ArrayOfFloat(coord[:,:,:,2].reshape(-1).tolist())
    points_for_query=ArrayOfArrayOfFloat([x_coord,y_coord,z_coord])
    
    # Get velocity
    print('\n Requesting velocity at {0}x{1}x{2} points...'.format(nx,ny,nz))
    result = query_data_velocity(time, points_for_query)
    vel=np.array(result).reshape((nx, ny, nz, 3))
        
    # Get velocity gradient
    print('\n Requesting velocity gradient at {0}x{1}x{2} points...'.format(nx,ny,nz))
    result = query_data_velocity_gradient(time, points_for_query)
    vel_grad=np.array(result).reshape((nx, ny, nz, 9))
    
    # Save data in h5 file:
    file_name = "JHTDB_time_"+str(time).replace('.','-')+"_n_"+str(nx)+"x"+str(ny)+"x"+str(nz)+".h5"
    file_path = "JHTDB_h5_files/" + file_name
    with h5py.File(file_path, "w") as f:
        f.attrs.create('time',time)
        f.attrs.create('nx',nx)
        f.attrs.create('ny',ny)
        f.attrs.create('nz',nz)
        f.create_dataset('coord',data=coord)
        f.create_dataset('vel',data=vel)
        f.create_dataset('vel_grad',data=vel_grad)
    print("\nDatabase file created: " + file_path) 
    return coord, vel, vel_grad

def get_velocity_data_y_ct(x_min=30.2185, x_max=800, y_value=0, z_min=0, z_max=240, nx=831, nz=512, time=0.25):
    
    # Time and spatial coordinates
    x = np.linspace(x_min, x_max, nx, dtype = 'float32')
    z = np.linspace(z_min, z_max, nz, dtype = 'float32')
    coord = np.empty((nx, nz, 3), dtype = 'float32') # done for having all values at same resolution
    coord[:, :, 0] = x[:, np.newaxis]
    coord[:, :, 1] = y_value
    coord[:, :, 2] = z[np.newaxis, :]
    
    # Convert to JHTDB structures (for web service query)
    x_coord=ArrayOfFloat(coord[:,:,0].reshape(-1).tolist())
    y_coord=ArrayOfFloat(coord[:,:,1].reshape(-1).tolist())
    z_coord=ArrayOfFloat(coord[:,:,2].reshape(-1).tolist())
    points_for_query=ArrayOfArrayOfFloat([x_coord,y_coord,z_coord])
    
    # Get velocity
    print('\n Requesting velocity at {0}x{1} points...'.format(nx,nz))
    result = query_data_velocity(time, points_for_query)
    vel=np.array(result).reshape((nx, nz, 3))
        
    # Get velocity gradient
    print('\n Requesting velocity gradient at {0}x{1} points...'.format(nx,nz))
    result = query_data_velocity_gradient(time, points_for_query)
    vel_grad=np.array(result).reshape((nx, nz, 9))
    
    # Save data in h5 file:
    file_name = "JHTDB_time_"+str(time).replace('.','-')+"_n_"+str(nx)+"x"+str(nz)+"_y_"+str(y_value).replace('.','-')+".h5"
    file_path = "JHTDB_h5_files/" + file_name
    with h5py.File(file_path, "w") as f:
        f.attrs.create('time',time)
        f.attrs.create('nx',nx)
        f.attrs.create('nz',nz)
        f.attrs.create('y_value',y_value)
        f.create_dataset('coord',data=coord)
        f.create_dataset('vel',data=vel)
        f.create_dataset('vel_grad',data=vel_grad)
    print("\nDatabase file created: " + file_path) 
    return coord, vel, vel_grad    

if __name__ == '__main__':
    Time=10.0
    # Coord, Vel, Vel_Gradient = get_velocity_data(x_min=30.2185, x_max=800, y_min=0, y_max=26.488, z_min=0, z_max=240, nx=5, ny=5, nz=5, time=Time)
    Coord, Vel, Vel_Gradient = get_velocity_data_y_ct(x_min=30.2185, x_max=1000, y_value=0.5, z_min=0, z_max=240, nx=831, nz=512, time=Time)
    
    # plt.show()
    