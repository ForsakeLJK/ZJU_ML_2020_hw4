import numpy as np
from kmeans import kmeans

def spectral(W, k):
    '''
    SPECTRUAL spectral clustering

        Input:
            W: Adjacency matrix, N-by-N matrix
            k: number of clusters

        Output:
            idx: data point cluster labels, n-by-1 vector.
    '''
    # YOUR CODE HERE
    # begin answer
    #TODO
    D = np.diag(np.sum(W, axis=1))
    L = D - W

    eigvalue, eigvector = np.linalg.eig(L)
    k_idx = np.argsort(eigvalue)[0:k]
    eigvector = eigvector[:, k_idx]

    idx = kmeans(eigvector, k)

    return idx


    # end answer
