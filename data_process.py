import numpy as np
import scipy
import scipy.misc
import cv2

def crop(image, face_box, res):
    
    ul = np.array([face_box[0], face_box[1]]) # upper left corner x1, y1
    br = np.array([face_box[2], face_box[3]]) # buttom right corner x2, y2
    
    x1 = max(0, ul[0] - 15)
    x2 = min(image.shape[1], br[0] + 15)
    y1 = max(0, ul[1] - 15)
    y2 = min(image.shape[0], br[1] + 15)

    croppedImg = image[y1: y2, x1: x2, :]

    croppedImg = cv2.resize(croppedImg, (int(res[0]), int(res[1])), interpolation=cv2.INTER_CUBIC)
    return croppedImg

def normalize(image_data, mean_color):
    '''
    Normalize image intensities from [0, 255] between [0, 1]
    '''
    image_data = image_data / 255.0

    for i in range(image_data.shape[-1]):
        image_data[:, :, i] -= mean_color[i]

    return image_data

def transform_kp(image, face_box, kp, res):
    '''
    Transform (translate and scale) the landmarks to the cropped image 
    '''
    newkp = np.copy(kp)
    height, width, _ = image.shape 

    ul = np.array([face_box[0], face_box[1]]) # upper left corner x1, y1
    br = np.array([face_box[2], face_box[3]]) # buttom right corner x2, y2

    old_x = br[0] - ul[0] + 30
    old_y = br[1] - ul[1] + 30
    # trnsfrom keypoints
    for i in range(kp.shape[0]):
        newkp[i, 0] -= ul[0] - 15
        newkp[i, 0] *= res[1]/old_x
        newkp[i, 0] = np.round(newkp[i, 0])

        newkp[i, 1] -= ul[1] - 15
        newkp[i, 1] *= res[0]/old_y
        newkp[i, 1] = np.round(newkp[i, 1])

    return newkp

def gaussian_kernel(x0, y0, sigma, res):
    '''
    Generate a 2D gaussian 
    '''
    x = np.arange(0, res[1], 1, float)
    y = np.arange(0, res[0], 1, float)[:, np.newaxis] 
    return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

def generate_hm(res, landmarks, sigma=3):
    '''
    Generate a stack of heatmaps for each image
    '''
    num_landmarks = len(landmarks)
    hm = np.zeros((res[1], res[0], num_landmarks), dtype=np.float32)
    for i in range(num_landmarks):
        if not np.array_equal(landmarks[i], [-1, -1]):
            hm[:, :, i] = gaussian_kernel(landmarks[i][0], landmarks[i][1], sigma, res)
        else:
            hm[:, :, i] = np.zeros((res[0], res[1]))
    return hm
