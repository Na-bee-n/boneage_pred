''' his program Enhance the images Using 'CLAHE'
    C-Contrast
    L-Limited
    A-Adaptive
    H-Histogram
    E-Equalization
'''
import cv2

def enhance_img(path1, path2, df):
    '''
    Reads image and enhances contrast using OpenCV
    
    Parameters
    ----------
    path1: directory where images are stored
    path2: directory where enhanced images will be stored
    df: dataframe

    Returns
    ----------
    Contrast-enhanced image
    '''
    # Create list of filenames of training images
    filenames = list(df['filename'])

    # Iterate through each filename
    for filename in filenames:
        # Open & enhance images
        img = cv2.imread(path1 + filename, 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(64, 64))
        cl1 = clahe.apply(img)
        
        # Save the contrast-enhanced image
        cv2.imwrite(path2 + filename, cl1)
