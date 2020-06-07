import numpy as np

def PCA(data):
    '''
    PCA	Principal Component Analysis

    Input:
      data      - Data numpy array. Each row vector of fea is a data point.
    Output:
      eigvector - Each column is an embedding function, for a new
                  data point (row vector) x,  y = x*eigvector
                  will be the embedding result of x.
      eigvalue  - The sorted eigvalue of PCA eigen-problem.
    '''

    # YOUR CODE HERE
    # Hint: you may need to normalize the data before applying PCA
    # begin answer
    N, P = data.shape
    # P-by-N
    # data = data.T
    # let E(row) = 0
    mean = np.mean(data, axis=0)
    data = data - mean
    data = data.T
    S = np.cov(data)
    eigvalue, eigvector = np.linalg.eig(S)
    # descending order
    idx = np.argsort(-eigvalue)
    eigvalue = eigvalue[idx]
    eigvector = eigvector[:, idx]

    return eigvalue, eigvector
    # end answer
