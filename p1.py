import numpy as np
import scipy.spatial as sp
import cv2

global img
# img = cv2.imread('dummy.png',cv2.IMREAD_UNCHANGED)
img = cv2.imread('venn.png',cv2.IMREAD_COLOR)
imgType = type(img[0,0]).__name__

def dist(img, point):
    lines, columns, channels = img.shape
    A = img.reshape((lines*columns, channels))
    B = np.zeros(shape=(1,3))
    B[0]=point
    distances = sp.distance.cdist(A,B, metric='euclidean')
    return distances
     

def mouse_callback(event, column, line, flags, params):
    if event == 1: #left button in my mac
        if (imgType == 'uint8'): #grayscale
            print (img [line, column])
        elif (imgType == 'ndarray'): #color         
            bgr = img [ line, column ]           
            print("(line, column):({},{}); BGR ({})".format(line, column,bgr))
            mask = create_mask(img, bgr)
            inv_mask = np.invert (mask)
            lines, columns, channels = img.shape
            result_image = np.zeros((lines, columns, channels), np.uint8)
            result_image[:,:]=(0,0,255)
            
            

            # result_image = cv2.bitwise_and(result_image, (mask * img))
            # cv2.imshow('result1', mask*img)
            # cv2.imshow('result2', inv_mask*result_image)
            red = cv2.add((mask*img), (inv_mask*result_image))
            cv2.imshow('red', red)
def create_mask (img, point):
    distances = dist(img, point)
    lines, columns, channels = img.shape
    mask = distances.reshape((lines, columns, 1))
    mask = (mask>=13)
    return mask
def apply_mask (img, mask): 

    # I want to put mask on top-left corner, So I create a ROI
    rows,cols,channels = img.shape
    roi = img[0:rows, 0:cols ]
    
    # # Now create a mask of logo and create its inverse mask also
    # img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    # print (img2gray)
    ret, bgmask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    cv2.imgshow('bg', bgmask)
    # mask_inv = cv2.bitwise_not(mask)
    # # Now black-out the area of logo in ROI
    # img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
    # # Take only region of logo from logo image.
    # img2_fg = cv2.bitwise_and(img2,img2,mask = mask)
    # # Put logo in ROI and modify the main image
    # dst = cv2.add(img1_bg,img2_fg)
    # img1[0:rows, 0:cols ] = dst
    # cv2.imgwrite('result.png', img1)

while(1):
    cv2.imshow('window',img)
    cv2.setMouseCallback('window',mouse_callback)
    k = cv2.waitKey(33)
    if k==27:    # Esc key to stop
        break

