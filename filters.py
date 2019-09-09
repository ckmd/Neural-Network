# gabor filter basic explanation https://cvtuts.wordpress.com/2014/04/27/gabor-filters-a-practical-overview/
import numpy as np
import cv2

def build_filters():
    filters = []
    ksize = 7
    lambd = [10.0,9.0,8.0,7.0]
    for la in lambd:
        for theta in np.arange(0, np.pi, np.pi / 4):
            # getGaborKernel(Size ksize, double sigma, double theta, double lambd, double gamma, double psi=CV_PI*0.5, int ktype=CV_64F )
            kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, la, 0.5, 0, ktype=cv2.CV_32F)
            kern /= 1.5*kern.sum()
            filters.append(kern)
    return filters

filter = np.array(build_filters())

# Custom filter / kernel
l1_filter = np.zeros((6,3,3))
# edge filter
l1_filter[0, :, :] = np.array([[[-1, 0, 1], 
                                [-1, 0, 1], 
                                [-1, 0, 1]]])
l1_filter[1, :, :] = np.array([[[1,   1,  1], 
                                [0,   0,  0], 
                                [-1, -1, -1]]])
# left sobel
l1_filter[2, :, :] = np.array([[[1,   0,  -1], 
                                [2,   0,  -2], 
                                [1,   0,  -1]]])
# right sobel
l1_filter[3, :, :] = np.array([[[-1,  0,   1], 
                                [-2,  0,   2], 
                                [-1,  0,   1]]])
# top sobel
l1_filter[4, :, :] = np.array([[[1,   2,   1], 
                                [0,   0,   0], 
                                [-1, -2, - 1]]])
# bottom sobel
l1_filter[5, :, :] = np.array([[[-1, -2,  -1], 
                                [0,   0,   0], 
                                [1,   2,   1]]])

if __name__ == "__main__":
    for f in filter:
        print(f)
        cv2.imshow('pic',f)
        cv2.waitKey(1000)
        # exit()