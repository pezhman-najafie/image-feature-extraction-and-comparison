import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage import feature
from colorama import Fore, Back, Style

#calculate mean variance
def calculate_mean_variance(image):
    rows, cols = image.shape
    division_rows = np.array_split(image, 3, axis=0)
    division_cols = [np.array_split(row, 3, axis=1) for row in division_rows]

    feature_vector1 = []
    feature_vector2 = []

    for i in range(3):
        for j in range(3):
            region = division_cols[i][j]
            mean_value = np.mean(region)
            variance_value = np.var(region)
            feature_vector1.extend([mean_value])
            feature_vector2.extend([variance_value])

    return feature_vector1, feature_vector2

#calculate lbp
def calculate_lbp(image, radius=1, samples=8):

    quantized_lbp = local_binary_pattern(image, samples, radius, method='uniform')
    return quantized_lbp.astype(np.uint8)

#calculate ltp
def calculate_ltp(image):
    return local_binary_pattern(image, P=8, R=1, method='uniform')

#calculate_lpq
def calculate_lpq(image, radius=1, samples=8):
    lbp = local_binary_pattern(image, P=samples, R=radius, method='uniform')
    lbp_min = lbp.min()
    lbp_max = lbp.max()
    quantized_lbp = np.floor((lbp - lbp_min) / (lbp_max - lbp_min) * 255)
    return quantized_lbp.astype(np.uint8)

#calculate_hog
def calculate_hog(image):
    return feature.hog(image, pixels_per_cell=(8, 8), block_norm='L2-Hys')

# Load the first image
input_image1 = cv2.imread('image4.jpg')
resized_image1 = cv2.resize(input_image1, (120, 120))
gray_image1 = cv2.cvtColor(resized_image1, cv2.COLOR_BGR2GRAY)

# Calculate features for the first image
mean_var1,variance_var1= calculate_mean_variance(gray_image1)

lbp1 = calculate_lbp(gray_image1)
ltp1 = calculate_ltp(gray_image1)
lpq1 = calculate_lpq(gray_image1)
hog1 = calculate_hog(gray_image1)

# Load the second image
input_image2 = cv2.imread('image4.jpg')
resized_image2 = cv2.resize(input_image2, (120, 120))
gray_image2 = cv2.cvtColor(resized_image2, cv2.COLOR_BGR2GRAY)

# Calculate features for the second image
mean_var2,variance_var2= calculate_mean_variance(gray_image2)
lbp2 = calculate_lbp(gray_image2)
ltp2 = calculate_ltp(gray_image2)
lpq2 = calculate_lpq(gray_image2)
hog2 = calculate_hog(gray_image2)

# Calculate Euclidean distances
mean_distance = np.linalg.norm(np.array(mean_var1) - np.array(mean_var2))
variance_distance = np.linalg.norm(np.array(variance_var1) - np.array(variance_var2))
lbp_distance = np.linalg.norm(lbp1 - lbp2)
ltp_distance = np.linalg.norm(ltp1 - ltp2)
lpq_distance = np.linalg.norm(lpq1 - lpq2)
hog_distance = np.linalg.norm(hog1 - hog2)

# Print the distances
print(Fore.GREEN)
print("Euclidean Distance between Mean Vectors:", mean_distance)
print(Fore.WHITE)
print("Euclidean Distance between Variance Vectors:", variance_distance)
print(Fore.CYAN)
print("Euclidean Distance between LBP Vectors:", lbp_distance)
print(Fore.YELLOW)
print("Euclidean Distance between LTP Vectors:", ltp_distance)
print(Fore.BLUE)
print("Euclidean Distance between LPQ Vectors:", lpq_distance)
print(Fore.RED)
print("Euclidean Distance between HOG Vectors:", hog_distance)