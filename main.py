import cv2
import numpy as np

# Link do material utilizado e do código:
# https://www.geeksforgeeks.org/image-registration-using-opencv-python/

# Abre os arquivos das imagens.
img1_color = cv2.imread("align.jpg")  # Image to be aligned.
img2_color = cv2.imread("ref.jpg")  # Reference image.

# Converte as imagens para grayscale.
img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)
height, width = img2.shape

# Create ORB detector with 5000 features.
orb_detector = cv2.ORB_create(5000)

# Encontra keypoints e descriptors.
# O primeiro argumento é a imagem, o segundo a máscara (nesse caso, não é necessária).
kp1, d1 = orb_detector.detectAndCompute(img1, None)
kp2, d2 = orb_detector.detectAndCompute(img2, None)

# Match features between the two images. Combina os atributos das duas imagens.
# We create a Brute Force matcher with
# Hamming distance as measurement mode.
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match the two sets of descriptors. Combina os descriptors das duas imagens.
# Descriptors are histograms of the image gradients to characterize the appearance of a keypoint
# Alteração para lista.
matches = list(matcher.match(d1, d2))

# Sort matches on the basis of their Hamming distance.
matches.sort(key=lambda x: x.distance)

# Take the top 90 % matches forward.
matches = matches[:int(len(matches) * 0.9)]
no_of_matches = len(matches)

# Define empty matrices of shape no_of_matches * 2.
p1 = np.zeros((no_of_matches, 2))
p2 = np.zeros((no_of_matches, 2))

for i in range(len(matches)):
    p1[i, :] = kp1[matches[i].queryIdx].pt
    p2[i, :] = kp2[matches[i].trainIdx].pt

# Find the homography matrix.
# Realiza as operações da Matriz Homográfica.
homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

# Use this matrix to transform the colored image wrt the reference image.
# Utiliza a matrix homográfica obtida para transformar a imagem.
transformed_img = cv2.warpPerspective(img1_color,
                                      homography, (width, height))

# Save the output.
cv2.imwrite('output.jpg', transformed_img)
