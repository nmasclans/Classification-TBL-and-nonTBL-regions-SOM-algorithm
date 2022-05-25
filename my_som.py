"""
Script to implement simple self organizing map using PyTorch, with methods
similar to clustering method in sklearn. 
Inspired on sklearn-som.SOM() class

@author: Núria Masclans
Created: 20-05-2022
"""

import numpy as np


class SOM():
    """
    The 2-D, rectangular grid self-organizing map class using Numpy.
    """
    def __init__(self, m=3, n=3, dim=3, lr=1, sigma=1, max_iter=1000000,random_state=None, sigma_evolution = 'constant'):
        """
        Parameters
        ----------
        m : int, default=3
            The shape along dimension 0 (vertical) of the SOM.
        n : int, default=3
            The shape along dimesnion 1 (horizontal) of the SOM.
        dim : int, default=3
            The dimensionality (number of features) of the input space.
        lr : float, default=1
            The initial step size for updating the SOM weights.
        sigma : float, optional
            Optional parameter for magnitude of change to each weight. Does not
            update over training (as does learning rate). Higher values mean
            more aggressive updates to weights.
        max_iter : int, optional
            Optional parameter to stop training if you reach this many
            interation.
        random_state : int, optional
            Optional integer seed to the random number generator for weight
            initialization. This will be used to create a new instance of Numpy's
            default random number generator (it will not call np.random.seed()).
            Specify an integer for deterministic results.
        """
        # Initialize descriptive features of SOM
        self.m = m
        self.n = n
        self.dim = dim
        self.shape = (m, n)
        self.lr_initial = lr
        self.lr = lr
        self.sigma_initial = sigma
        self.sigma = sigma
        self.max_iter = int(max_iter)
        self.sigma_evolution = sigma_evolution

        # Initialize weights
        self.random_state = random_state
        rng = np.random.default_rng(random_state)
        self.weights = rng.normal(size=(m * n, dim))
        self._locations = self._get_locations(m, n)

        # Set after fitting
        self._inertia = None
        self._n_iter = None
        self._trained = False
        self._cluster_centers_history = None
        self._lambda = None

    def get_weights(self):
        return self.weights
    
    def _get_locations(self, m, n):
        """
        Return the indices of an m by n array.
        """
        return np.argwhere(np.ones(shape=(m, n))).astype(np.int64)

    def _find_bmu(self, x):
        """
        Find the index of the best matching unit for the input vector x.
        """
        # Stack x to have one row per weight
        x_stack = np.stack([x]*(self.m*self.n), axis=0)
        # Calculate distance between x and each weight
        distance = np.linalg.norm(x_stack - self.weights, axis=1)
        # Find index of best matching unit
        return np.argmin(distance)

    def step(self, x):
        """
        Do one step of training on the given input vector.
        """
        # Stack x to have one row per weight
        x_stack = np.stack([x]*(self.m*self.n), axis=0)

        # Get index of best matching unit
        bmu_index = self._find_bmu(x)

        # Find location of best matching unit
        bmu_location = self._locations[bmu_index,:]

        # Find square distance from each weight to the BMU
        stacked_bmu = np.stack([bmu_location]*(self.m*self.n), axis=0)
        bmu_distance = np.sum(np.power(self._locations.astype(np.float64) - stacked_bmu.astype(np.float64), 2), axis=1)

        # Compute update neighborhood
        neighborhood = np.exp((bmu_distance / (2 * (self.sigma ** 2))) * -1)
        local_step = self.lr * neighborhood

        # Stack local step to be proper shape for update
        local_multiplier = np.stack([local_step]*(self.dim), axis=1)

        # Multiply by difference between input and weights
        delta = local_multiplier * (x_stack - self.weights)

        # Update weights
        self.weights += delta

    def _compute_point_intertia(self, x):
        """
        Compute the inertia of a single point. Inertia defined as squared distance
        from point to closest cluster center (BMU)
        """
        # Find BMU
        bmu_index = self._find_bmu(x)
        bmu = self.weights[bmu_index]
        # Compute sum of squared distance (just euclidean distance) from x to bmu
        return np.sum(np.square(x - bmu))

    def fit(self, X, epochs=1, shuffle=True, save_cluster_centers_history=False):
        """
        Take data (a tensor of type float64) as input and fit the SOM to that
        data for the specified number of epochs.

        Parameters
        ----------
        X : ndarray
            Training data. Must have shape (n, self.dim) where n is the number
            of training samples.
        epochs : int, default=1
            The number of times to loop through the training data when fitting.
        shuffle : bool, default True
            Whether or not to randomize the order of train data when fitting.
            Can be seeded with np.random.seed() prior to calling fit.

        Returns
        -------
        None
            Fits the SOM to the given data but does not return anything.
        """
        
        # Count total number of iterations
        global_iter_counter = 0
        n_samples = X.shape[0]
        total_iterations = int(np.min([epochs * n_samples, self.max_iter]))
        print('Training duration: {0} iterations - {1} epochs'.format(total_iterations, total_iterations/n_samples))
        self._n_iter = total_iterations
        self._lambda = total_iterations/epochs # or epochs/self.max_radius ?
        if save_cluster_centers_history:
            cluster_centers_history = np.zeros((self.m*self.n,self.dim,self._n_iter)) 

        for epoch in range(epochs):
            
            print('Training... - Epoch #{}'.format(epoch))
            
            # Break if past max number of iterations
            if global_iter_counter >= self.max_iter:
                break

            if shuffle:
                rng = np.random.default_rng(self.random_state)
                indices = rng.permutation(n_samples)
            else:
                indices = np.arange(n_samples)

            # Train
            for idx in indices:
                
                # Break if past max number of iterations
                if global_iter_counter >= self.max_iter:
                    break
                input = X[idx]
                
                # Do one step of training
                self.step(input)
                
                # Store cluser centers history
                if save_cluster_centers_history:
                    cluster_centers_history[:,:,global_iter_counter] = self.weights
                
                global_iter_counter += 1
                
                # Update learning rate
                self.lr = self.lr_initial*np.exp((global_iter_counter/self._lambda)*(-1))
                
                # Update neighbourhood radius:
                if self.sigma == 'constant':
                    pass
                else: # self.sigma == 'exponential_decay'
                    self.sigma = self.sigma_initial*np.exp((global_iter_counter/self._lambda)*(-1))
                
                
        # Compute inertia
        inertia = np.sum(np.array([float(self._compute_point_intertia(x)) for x in X]))
        self._inertia = inertia

        # Set n_iter attribute
        # self._n_iter = global_iter_counter

        # Set trained flag
        self._trained = True
        
        # Set weights history
        if save_cluster_centers_history:
            self._cluster_centers_history = cluster_centers_history

        print('Training done!\n')
        return

    def predict(self, X):
        """
        Predict cluster for each element in X.

        Parameters
        ----------
        X : ndarray
            An ndarray of shape (n, self.dim) where n is the number of samples.
            The data to predict clusters for.

        Returns
        -------
        labels : ndarray
            An ndarray of shape (n,). The predicted cluster index for each item
            in X.
        """
        # Check to make sure SOM has been fit
        if not self._trained:
            raise NotImplementedError('SOM object has no predict() method until after calling fit().')

        # Make sure X has proper shape
        assert len(X.shape) == 2, f'X should have two dimensions, not {len(X.shape)}'
        assert X.shape[1] == self.dim, f'This SOM has dimension {self.dim}. Received input with dimension {X.shape[1]}'

        labels = np.array([self._find_bmu(x) for x in X])
        print('Prediction done!\n')
        return labels

    def transform(self, X):
        """
        Transform the data X into cluster distance space.

        Parameters
        ----------
        X : ndarray
            Data of shape (n, self.dim) where n is the number of samples. The
            data to transform.

        Returns
        -------
        transformed : ndarray
            Transformed data of shape (n, self.n*self.m). The Euclidean distance
            from each item in X to each cluster center.
        """
        # Stack data and cluster centers
        X_stack = np.stack([X]*(self.m*self.n), axis=1)
        cluster_stack = np.stack([self.weights]*X.shape[0], axis=0)

        # Compute difference
        diff = X_stack - cluster_stack

        # Take and return norm
        print('Transformation done!\n')
        return np.linalg.norm(diff, axis=2)

    def fit_predict(self, X, **kwargs):
        """
        Convenience method for calling fit(X) followed by predict(X).

        Parameters
        ----------
        X : ndarray
            Data of shape (n, self.dim). The data to fit and then predict.
        **kwargs
            Optional keyword arguments for the .fit() method.

        Returns
        -------
        labels : ndarray
            ndarray of shape (n,). The index of the predicted cluster for each
            item in X (after fitting the SOM to the data in X).
        """
        # Fit to data
        self.fit(X, **kwargs)

        # Return predictions
        return self.predict(X)

    def fit_transform(self, X, **kwargs):
        """
        Convenience method for calling fit(X) followed by transform(X). Unlike
        in sklearn, this is not implemented more efficiently (the efficiency is
        the same as calling fit(X) directly followed by transform(X)).

        Parameters
        ----------
        X : ndarray
            Data of shape (n, self.dim) where n is the number of samples.
        **kwargs
            Optional keyword arguments for the .fit() method.

        Returns
        -------
        transformed : ndarray
            ndarray of shape (n, self.m*self.n). The Euclidean distance
            from each item in X to each cluster center.
        """
        # Fit to data
        self.fit(X, **kwargs)

        # Return points in cluster distance space
        return self.transform(X)

    @property
    def cluster_centers(self):
        return self.weights.reshape(self.m, self.n, self.dim)

    @property
    def inertia(self):
        if self._inertia is None:
            raise AttributeError('SOM does not have inertia until after calling fit()')
        return self._inertia

    @property
    def n_iter(self):
        if self._n_iter is None:
            raise AttributeError('SOM does not have n_iter_ attribute until after calling fit()')
        return self._n_iter
    
    @property
    def cluster_centers_history(self):
        if self._cluster_centers_history is None:
            raise AttributeError('SOM does not have cluster centers history until after calling fit()')
        return self._cluster_centers_history.reshape((self.m, self.n, self.dim,self._n_iter))
