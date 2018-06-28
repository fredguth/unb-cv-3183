import numpy as np
import cv2
import glob
import os
import imutils
import math
import scipy.spatial as sp
from sklearn.cluster import MiniBatchKMeans


def xrange(x):
    return iter(range(x))

def apply_screen(frame, screen, bgr):
    bgr = np.asarray(bgr)
    
    mask = create_mask(frame, bgr)
    inv_mask = np.invert (mask)
    lines, columns, channels = frame.shape
    result_image = np.zeros((lines, columns, channels), np.uint8)
    result_image[:,:]=screen
    black = cv2.add((mask*frame), (inv_mask*result_image))
    return black

def create_mask (img, point):
    return (dist(img, point)>=35)

def dist(img, point):
    
    lines, columns, channels = img.shape
    A = img.reshape((lines*columns, channels))
    B = point.reshape((1,3))
    distances = sp.distance.cdist(A,B, metric='euclidean')
    distances = distances.reshape((lines, columns, 1))
    return distances
    


def simplify(image):
    num_down = 2       # number of downsampling steps
    num_bilateral = 7  # number of bilateral filtering steps


    # downsample image using Gaussian pyramid
    for _ in xrange(num_down):
        image = cv2.pyrDown(image)

    # repeatedly apply small bilateral filter instead of
    # applying one large filter
    for _ in xrange(num_bilateral):
        image = cv2.bilateralFilter(image, d=9,
                                        sigmaColor=9,
                                        sigmaSpace=7)

    # upsample image to original size
    for _ in xrange(num_down):
        image = cv2.pyrUp(image)

    return image

def process(image, contour, hull=False, crop=False):

    if hull:
        contour = cv2.convexHull(contour)

    foreground = cv2.drawContours(image.copy(), [contour], 0, (0, 0, 255), -1)
    redmask = np.where((foreground[:, :, 0] == 0) &
                        (foreground[:, :, 1] == 0) &
                        (foreground[:, :, 2] == 255))
    
    fg = image.copy()
    fg[:, :, :] = 0
    fg[redmask] = 1

    bg = image.copy()
    bg[:, :, :] = 1
    bg[redmask] = 0
    result_image = image.copy()
    result_image[:, :] = (255, 255, 255)
    image = cv2.add((fg*image), (bg*result_image))

    if crop:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        scale = 1.15  # cropping margin, 1 == no margin
        W = rect[1][0]
        H = rect[1][1]

        Xs = [i[0] for i in box]
        Ys = [i[1] for i in box]
        x1 = min(Xs)
        x2 = max(Xs)
        y1 = min(Ys)
        y2 = max(Ys)

        rotated = False
        angle = rect[2]

        if angle < -45:
            angle += 90
            rotated = True

        center = (int((x1+x2)/2), int((y1+y2)/2))
        size = (int(scale*(x2-x1)), int(scale*(y2-y1)))
        # # mostly for debugging purposes
        # cv2.circle(img_box, center, 10, (0, 255, 0), -1)

        M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)

        # !!!Important: which image to crop
        cropped = cv2.getRectSubPix(image, size, center)
        cropped = cv2.warpAffine(cropped, M, size)

        croppedW = W if not rotated else H
        croppedH = H if not rotated else W

        image = cv2.getRectSubPix(
            cropped, (int(croppedW*scale), int(croppedH*scale)), (size[0]/2, size[1]/2))
    return image

def apply_mask(matrix, mask, fill_value):
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    return masked.filled()



def apply_threshold(matrix, low_value, high_value):
    low_mask = matrix < low_value
    matrix = apply_mask(matrix, low_mask, low_value)

    high_mask = matrix > high_value
    matrix = apply_mask(matrix, high_mask, high_value)

    return matrix

def quantitize(img, n):
    
    (h, w) = img.shape[:2]

    img = np.reshape(img, (h * w, 3))
    clt = MiniBatchKMeans(n_clusters=n)
    labels = clt.fit_predict(img)
    quant = clt.cluster_centers_.astype("uint8")[labels]

    quant = np.reshape(quant, (h, w, 3))
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)


    return quant

def simplest_cb(img, percent):
    assert img.shape[2] == 3
    assert percent > 0 and percent < 100

    half_percent = percent / 200.0

    channels = cv2.split(img)

    out_channels = []
    for channel in channels:
        assert len(channel.shape) == 2
        # find the low and high precentile values (based on the input percentile)
        height, width = channel.shape
        vec_size = width * height
        flat = channel.reshape(vec_size)

        assert len(flat.shape) == 1

        flat = np.sort(flat)

        n_cols = flat.shape[0]

        low_val = flat[math.floor(n_cols * half_percent)]
        high_val = flat[math.ceil(n_cols * (1.0 - half_percent))]

        # print("Lowval: ", low_val)
        # print("Highval: ", high_val)

        # saturate below the low percentile and above the high percentile
        thresholded = apply_threshold(channel, low_val, high_val)
        # scale the channel
        normalized = cv2.normalize(
            thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
        out_channels.append(normalized)

    return cv2.merge(out_channels)

filenames = glob.glob('./dataset/*/*')
videos = glob.glob('./videos/*.mp4')

#create directory for Ground Truth
for video in videos:
  video = video.replace('./videos/', '')
  video = video.replace('.mp4', '')
  directory = './GT/'+video
  if not os.path.exists(directory):
      print (directory)
      os.makedirs(directory)


for filename in filenames:

  image = cv2.imread(filename, cv2.IMREAD_COLOR)
#   image = simplest_cb(image, 1)
#   
#   green = apply_screen(image, (200,255,200))
  simpler = quantitize(image, 6)
  green = apply_screen(simpler, (255,255,255), [0, 106, 70])
  green = apply_screen(green, (255,255,255), [0, 65, 0])
  
#   white = apply_screen(green, (255,255,255), [50,255,50])
  gray = cv2.cvtColor(green, cv2.COLOR_BGR2YCrCb)

  gray = gray[:,:,0]

#   gray = cv2.cvtColor(green, cv2.COLOR_BGR2GRAY)
  ret, thresh = cv2.threshold(gray.copy(), 1, 255,  cv2.THRESH_BINARY | cv2.THRESH_OTSU)

  mask = 255 - thresh
  result_image = image.copy()[:,:,:]=255
#   image = cv2.add((mask*image), (thresh*result_image))
  image[thresh]= 255
#   mask = thresh

  _, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

  for contour in contours:
    area = cv2.contourArea(contour)    
    # only contours large enough to contain object
    if area > 300000:
      image = process(image, contour, hull=False, crop=True)
  


  filename = filename.replace('dataset', 'GT')
  
  print (filename)
  
  cv2.imwrite(filename, image)
  
