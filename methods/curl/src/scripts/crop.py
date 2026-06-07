from skimage import view_as_windows
import numpy as np

def random_crop(imgs, out):
    """
    Vectorized random crop
    args:
        imgs: shape (B,C,H,W)
        out: output size (e.g. 84)
    """
    
    # n: batch size.
    n = imgs.shape[0]
    img_size = imgs.shape[-1] # e.g. 100
    crop_max = img_size - out
    
    imgs = np.transpose(imgs, (0, 2, 3, 1))
    
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    
    # creates all sliding window
    # combinations of size (out)
    
    windows = view_as_windows(
        imgs, (1, out, out, 1))[..., 0,:,:, 0]
    
    # selects a random window 
    # for each batch element
    cropped = windows[np.arange(n), w1, h1]
    return cropped