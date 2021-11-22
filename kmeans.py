#!usr/bin/python3
# -*- coding : utf8 -*-

import random;
import numpy as np;

from functions import *;


class KMeans:
    ''' 
        K-means clustering for grouping data into K groups of similar objects

        Parameters
        ----------
        n_iter : int, default=100
            Number of iterations that will do k-means

        init : 'kmeans++' or random' or 'iter', default='kmeans++'
            Centroids initialization method

        n_clusters : int, default=3
            Number of clusters or centroids where the samples
            will be grouped. It is famous K of K-means.

        n_init : int, default=100
            Number of time the k-means algorithm will be run with
            different centroid seeds. The final results will be
            the best output of n_init consecutive runs in terms of inertia.

        seed : int, default=None
            For the reproducibility

        init_centroids : ndarray of shape (n_init, n_clusters_, n_features);
            Contains all the initial centroids used to fit k-means

        Attributes
        ----------
        centroids_ : ndarray of shape (n_clusters, n_features)
            Coordinates of cluster centers.

        labels_ : ndarray of shape (n_samples,)
            Labels of each samples
       
        inertia_ : float
            Sum of squared distances of samples to their closest cluster center
    '''


    def __init__(self, n_clusters=3, init='kmeans++', n_init=10, n_iter=100, seed=None): 

        # Check n_clusters 
        if isinstance(n_clusters, int) is False or n_clusters <= 0:
            raise ValueError("'n_clusters' must be an integer and"
                              " strictly greater than 0. You "
                              "gave %s." % str(n_clusters));

        # Check init
        names_init = ['random', 'iter', 'kmeans++']; 
        if init not in names_init:
            raise ValueError("'init' can only take one of three"
                             " values : 'random', 'iter' or 'kmeans++'"
                             ". You gave %s." % str(init));

        # Check n_init
        if isinstance(n_init, int) is False or n_init <= 0:
            raise ValueError("'n_init' must be an integer and" 
                             "strictly greater than 0."
                             "You gave %s." % str(n_init));

        # Check n_iter
        if isinstance(n_iter, int) is False or n_iter <= 0:
            raise ValueError("'n_iter' must be an integer and"
                             "be strictly greater than 0."
                             "You gave %s." % str(n_iter)); 
        
        # Check seed
        if seed is not None and (isinstance(seed, int) is not True or seed <= 0):
            raise ValueError("'seed' must be an integer and strictly "
                             "greater than 0. You gave %s." % str(seed));

        # Initialization
        self.n_clusters = n_clusters;
        self.init = init;
        self.n_init = n_init;
        self.n_iter = n_iter;
        self.seed = seed;


    def fit(self, X):

        n_samples, n_features = X.shape;
        self.inertia_ = 0.0;
        self.labels_ = np.zeros((n_samples, ), dtype=int)

        # Store all centroids, labels, initialization centroids and inertia
        all_centroids = np.zeros((self.n_init, self.n_clusters, n_features));
        all_labels = np.zeros((self.n_init, n_samples), dtype=int);
        self.init_centroids = np.zeros((self.n_init, self.n_clusters, n_features));
        all_inertia  = np.zeros((self.n_init, ));
        
        # Let's run k-means a finite number of times in order to select the one with
        # the smallest inertia
        for idx in range(0, self.n_init):
            
            # Initializing cluster centroids
            if self.init == 'random':
                self.centroids_ = rand_initialisation(X, self.n_clusters, self.seed, idx+100);
                self.init_centroids[idx] = self.centroids_.copy();

            elif self.init == 'iter':
                self.centroids_ = iter_initialisation(X, self.n_clusters);
                self.init_centroids[idx] = self.centroids_.copy();

            elif self.init == 'kmeans++':
                self.centroids_ = kmeans_plus_plus(X, self.n_clusters, self.seed, idx+100);
                self.init_centroids[idx] = self.centroids_.copy();
            else:
                raise ValueError("Unknown initialization method");


            for iteration in range(0, self.n_iter):

                inertia = 0.0;

                # Array which will store the sum of the samples when they are assigned
                # to the cluster (centroid) closest to them. Then we'll just do an 
                # arithmetic mean to find the new centroids
                centroids = np.zeros((self.n_clusters, n_features));
                
                # Array which will contain the number of samples assigned to each cluster
                samp_per_cluster = np.zeros((self.n_clusters, ), dtype=int);

                for sample in range(0, n_samples):
                    
                    # We will calculate in all self.n_cluster_ distances for the sample
                    dist_samp_clusters = np.zeros((self.n_clusters, ));
                
                    for cluster in range(0, self.n_clusters):
                        norm = np.linalg.norm(X[sample] - self.centroids_[cluster]);
                        dist_samp_clusters[cluster] = np.square(norm);

                    # Find the closest cluster to sample
                    closest_cluster = dist_samp_clusters.argmin();

                    # Join to the sample its cluster
                    self.labels_[sample] = closest_cluster;

                    # Add this sample to the associated cluster
                    centroids[closest_cluster] = centroids[closest_cluster] + X[sample];

                    # Let's increment the number of samples contained in the cluster
                    samp_per_cluster[closest_cluster] = samp_per_cluster[closest_cluster] + 1;

                    # Add this distance to the inertia
                    inertia = inertia + dist_samp_clusters[closest_cluster];   

                # Let's calculate the new centroids
                for cluster in range(0, self.n_clusters):

                    # 1 when the cluster does not contain any example apart from its centroid
                    samp_per_cluster[cluster] = max([samp_per_cluster[cluster], 1]);

                    # New centroids
                    self.centroids_[cluster] = centroids[cluster] / samp_per_cluster[cluster];

                self.inertia_ = inertia;

                all_centroids[idx] = self.centroids_;
                all_labels[idx] = self.labels_;
                all_inertia[idx]  = self.inertia_;

        # Let's select the best k initial centroids, k centroids, labels and inertia 
        # according the ones that have the smallest inertia
        self.inertia_ = all_inertia[all_inertia.argmin()];
        self.centroids_ = all_centroids[all_inertia.argmin()]; 
        self.labels_ = all_labels[all_inertia.argmin()];

        return self;


    def predict(self, X):

        n_samples, n_features = X.shape;
        predictions = np.zeros((n_samples, ), dtype=int)

        for sample in range(0, n_samples):
                    
            # Distance from a sample to all centroids
            dist_samp_clusters = np.zeros((self.n_clusters_, ));
            
            for cluster in range(0, self.n_clusters):
                norm = np.linalg.norm(X[sample] - self.centroids_[cluster]);
                dist_samp_clusters[cluster] = np.square(norm);

            # Find the closest cluster  to sample
            closest_cluster = dist_samp_clusters.argmin();

            # Join sample and its cluster
            predictions[sample] = closest_cluster;
        
        return predictions


