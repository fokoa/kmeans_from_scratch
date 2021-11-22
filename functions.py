#!usr/bin/python3
# -*- coding : utf8 -*-

import numpy as np;


def first_second(X, center):
    """ Compute the first and second centroid """

    n_samples, n_features = X.shape;

    # As it will have n_samples values of distance
    dist_array = np.zeros((n_samples, ));

    for sample in range(0, n_samples-1):
        dist_array[sample] = np.linalg.norm( X[sample] - center ); # Store all distance

    # The subscript in 'dist_array' which have the bigger value matches vector or sample in 
    # 'X' that is the fastest from the 'center'
    return X[dist_array.argmax()];


def third_other(X, centroid_1, centroid_2):
    """ Compute the third and centroid greater than 3"""

    n_samples, n_features = X.shape;

    # As it will have n_samples values of distance
    dist_array = np.zeros((n_samples, ));

    for sample in range(0, n_samples):
        dist_1 = np.linalg.norm( X[sample] - centroid_1 );
        dist_2 = np.linalg.norm( X[sample] - centroid_2 );
        dist_array[sample] = (dist_1 + dist_2);

    # The subscript in 'dist_array' which have the bigger value matches vector or sample in 
    # 'X' that is the vector with maximum sum of squared distances between that vector and 
    # 'centroid_1' and that vector and the 'centroid_2'
    return X[dist_array.argmax()];
   

def iter_initialisation(X, n_clusters):
    """ Initialize according an heuristic """

    n_features = X.shape[1];
    zero_centroid = np.zeros((n_features, )); # nul centroid
    centroids = np.zeros((n_clusters, n_features)); # array of centroids

    for cluster in range(0, n_clusters):
        if cluster == 0:
            centroids[cluster] = first_second(X, zero_centroid); # the first centroid according heuristic

        elif cluster == 1:
            centroids[cluster] = first_second(X, centroids[cluster-1]) # the second centroid according heuristic

        elif cluster >= 2:
            centroids[cluster] = third_other(X, centroids[cluster-2], centroids[cluster-1]);
    
    return centroids;


def rand_initialisation(X, n_clusters, seed, cste):
    """ Initialize vector centers from X randomly """

    index = [];
    repeat = n_clusters;
    
    # Take one index
    if seed is None:
        idx = np.random.RandomState().randint(X.shape[0]);
    else:
        idx = np.random.RandomState(seed+cste).randint(X.shape[0]);

    while repeat != 0:    

        # Let's check that we haven't taken this index yet
        if idx not in index:
            index.append(idx);
            repeat = repeat - 1;

        if seed is not None:
            idx = np.random.RandomState(seed+cste+repeat).randint(X.shape[0]);

    return X[index];


def kmeans_plus_plus(X, n_clusters, seed, cste):
    """ Initialisaton of centroids according heuristic kmeans++ """

    n_samples, n_features = X.shape;
    centroids = np.zeros( (1, n_features));

    # First centroid is randomly selected from the data points X
    if seed is None:
        centroids = X[np.random.RandomState().randint(X.shape[0])];
    else:
        centroids = X[np.random.RandomState(seed+cste).randint(X.shape[0])];

    # Let's select remaining "n_clusters - 1" centroids
    for cluster_idx in range(1, n_clusters):

        # Array that will store distances of data points from nearest centroid
        distances = np.zeros((n_samples, ));

        for sample_idx in range(n_samples):
            minimum = np.inf;
              
            # Let's compute distance of 'point' from each of the previously
            # selected centroid and store the minimum distance
            for j in range(0, centroids.shape[0]):
                dist = np.square( np.linalg.norm(X[sample_idx] - centroids[j]) );
                minimum = min(minimum, dist);
            
            distances[sample_idx] = minimum;
        
        centroids = np.vstack((centroids, X[np.argmax(distances)]));
        # distances = np.zeros((n_samples, ));

    return centroids


