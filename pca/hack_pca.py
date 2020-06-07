import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from pca import PCA
from scipy.ndimage import rotate

def hack_pca(filename):
    '''
    Input: filename -- input image file name/path
    Output: img -- image without rotation
    '''
    img_r = (plt.imread(filename)).astype(np.float64)

    # YOUR CODE HERE
    # begin answer
    img_gray = color.rgb2gray(img_r)

    coord = np.array(np.where(img_gray < 10)).T
    eigvalue, eigvector = PCA(coord)

    angle = np.arctan(eigvector[0, 0]/eigvector[0, 1]) * 180 / np.pi
    
    img = rotate(img_r, -angle).astype(np.int)

    return img

    # end answer
