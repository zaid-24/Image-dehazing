import cv2
import math
from cv2 import imshow
from cv2 import PSNR
import numpy as np
import csv
import timeit
# in darkchannel function we have find out of pixelx of colour out of r,g,b having mininmum value .then we have form a dark image with respect to that colour channel

def calculate_psnr(img1, img2):
    img1 = img1.astype('float64')/255
    img2 = img2.astype('float64')/255
    img1=cv2.resize(img1,(400,600))
    img2=cv2.resize(img2,(400,600))
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    img1=cv2.resize(img1,(400,600))
    img2=cv2.resize(img2,(400,600))
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()    
def calculate_ssim(img1, img2):
    img1=cv2.resize(img1,(400,600))
    img2=cv2.resize(img2,(400,600))
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')
        
def DarkChannel(im):
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = np.ones((5, 5), np.uint8)
    dark = cv2.erode(dc, kernel)
    return dark

def AtmLight(im, dark):
    [h, w] = im.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000), 1))
    darkvec = dark.reshape(imsz)
    imvec = im.reshape(imsz, 3)
    indices = darkvec.argsort()
    indices = indices[(imsz-numpx)::]
    atmsum = np.zeros([1, 3])
    for ind in range(1, numpx):
       atmsum = atmsum + imvec[indices[ind]]
    A = atmsum / numpx
    return A

def Estimate_transmissino(image, A):
    omega = 0.95
    image3 = np.empty(image.shape, image.dtype)
    for ind in range(0, 3):
        image3[:, :, ind] = image[:, :, ind]/A[0, ind]
    transmission = 1 - omega*DarkChannel(image3)
    return transmission

#guidance image: I filtering input image: p  regularization parameter: eps
def Guidedfilter(im, p, r, eps): # time complexity O(1)  r=radius
    mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(im*p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I*mean_p
    mean_II = cv2.boxFilter(im*im, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I*mean_I
    a = cov_Ip/(var_I + eps)
    b = mean_p - a*mean_I
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
    filter = mean_a*im + mean_b
    return filter


def Refine_Transmission(image, et):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = np.float64(gray_image)/255
    r = 60
    eps = 0.0001
    t = Guidedfilter(gray_image, et, r, eps)
    return t


def Recover(im, t, A, tx=0.1):
    res = np.empty(im.shape, im.dtype)
    t = cv2.max(t, tx)
    for ind in range(0, 3):
        res[:, :, ind] = (im[:, :, ind]-A[0, ind])/t + A[0, ind]
    return res


if __name__ == '__main__':
    starttime = timeit.default_timer()
    starting_path="./Hazy Images/"
    ending_path="_outdoor_hazy.jpg"
    starting_path_for_saving="./dehazed images/"
    ending_path_for_saving="__dehazed.png"
    starting_path_for_realimg="./Ground Truth/"
    ending_path_for_realimg="_outdoor_GT.jpg"
    f = open('values.csv', 'w')
    writer = csv.writer(f)
    for i in range(1,17):
        k=i
        if k<10:
            k=str(0)+str(i)
        name=starting_path+str(k)+ending_path
        inputimage = cv2.imread(name)
        I = inputimage.astype('float64')/255
        dark = DarkChannel(I)
        A = AtmLight(I, dark)
        te = Estimate_transmissino(I, A)
        t = Refine_Transmission(inputimage, te)
        J = Recover(I, t, A, 0.1)
        path_to_save=starting_path_for_saving+str(k)+ending_path_for_saving
        cv2.imwrite(path_to_save,J*255)  
        path_for_ground_truth=starting_path_for_realimg+str(k)+ending_path_for_realimg      
        final=cv2.imread(path_for_ground_truth)
        row=[int(k),calculate_psnr(J,final),calculate_ssim(J,final)]
        writer.writerow(row)
    print("The time taken is:", timeit.default_timer() - starttime,"s")
#below is code for video part which is not working that's why i have commented it
#     cap = cv2.VideoCapture('./Hazy Video/hazy.mp4')
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
#     video = cv2.VideoWriter('Dehaze.mp4', fourcc, 30,(400,600))
# # Check if camera opened successfully
#     if (cap.isOpened() == False):
#         print("Error opening video  file")
# # Read until video is completed
#     for i in range(360):
#   # Capture frame-by-frames
#         ret, frame = cap.read()
#         if ret == True:
#     # Display the resulting frame
#             # cv2.imshow('Frame', frame)
#             # inputimage = cv2.imread(fn)
#             fram=cv2.resize(frame,(400,600))
#             I = fram.astype('float64')/255
#             dark = DarkChannel(I)
#             A = AtmLight(I, dark) 
#             te = Estimate_transmissino(I, A)
#             t = Refine_Transmission(fram, te)
#             J = Recover(I, t, A, 0.1)
#             img_arr=[]
#             img_arr.append(J)
            
#             video.write(img_arr[0])
#             if cv2.waitKey(10) & 0xFF == ord('q'):
#                 break
#         else:
#             break
# 	# When everything done, release the video capture object
#     cap.release()
#     cv2.destroyAllWindows()
#     video.release()
#     cv2.waitKey() 
