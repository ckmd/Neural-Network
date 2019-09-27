# gabor filter basic explanation https://cvtuts.wordpress.com/2014/04/27/gabor-filters-a-practical-overview/
import numpy as np
import cv2

def build_filters(ksize, lamb, gamma):
    filters = []
    for theta in np.arange(0, np.pi, np.pi / 4):
        # getGaborKernel(Size ksize, double sigma, double theta, double lambd, double gamma, double psi=CV_PI*0.5, int ktype=CV_64F )
        kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, lamb, gamma, 0, ktype=cv2.CV_32F)
        kern /= 1.5*kern.sum()
        filters.append(kern)
    return filters
# 31 = 8,0.5
# 15 = 8,0.5
# 7 = 7,2
# 3 = 5,2
filter1 = np.array(build_filters(31,8,0.5))
filter2 = np.array(build_filters(15,8,0.5))
filter3 = np.array(build_filters(7,7,2))
filter4 = np.array(build_filters(3,5,2))
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
    print(filter1[0])
    for ind,f in enumerate(filter1):
        print(np.amax(f))
        cv2.imshow('pic',f)
        cv2.waitKey(1000)
        cv2.imwrite('img_name'+str(ind)+'.jpg', f * 255)
    f = np.ones((31,31));
    cv2.imsave('pic',f)
    cv2.waitKey(1000)
    exit()