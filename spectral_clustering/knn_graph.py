import numpy as np
from scipy.spatial.distance import cdist

def knn_graph(X, k, threshold):
    '''
    KNN_GRAPH Construct W using KNN graph

        Input:
            X - data point features, n-by-p maxtirx.
            k - number of nn.
            threshold - distance threshold.

        Output:
            W - adjacency matrix, n-by-n matrix.
    '''

    # YOUR CODE HERE
    # begin answer
    N, P = X.shape
    W = np.zeros((N, N))
    # distance matrix
    dist = cdist(X, X)
    dist = np.where(dist > threshold, np.Infinity, dist)
    # sort per row
    idx = np.argsort(dist, axis=1)

    # binary edge weights here
    for i in range(N):
        W[i, idx[i, 1:k+1]] = 1 

    return W
    # end answer
