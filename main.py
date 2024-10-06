import cv2
import numpy as np
import matplotlib.pyplot as plt


def calculate_matches(image1, image2):
    # Load the images
    img1 = cv2.imread(image1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2, cv2.IMREAD_GRAYSCALE)

    # Initialize the SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors for both images
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # Initialize the FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Perform matching
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Apply Lowe's ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    return good_matches, keypoints1, keypoints2


def dlt(matrix1, matrix2, n=4):
    A = np.zeros((2*n,9)) # filling matrix A with zeros
    zero = np.zeros((3,))
    for i in range(n): # creating matrix A as in presentation 7 slide 44
        dr1 = matrix1[i,:]
        dr2 = matrix2[i,:]
        A[2 * i:2 * i + 2, :] = np.vstack([np.concatenate((zero, -dr1, dr1 * dr2[1])),
                                           np.concatenate((dr1, zero, -dr1 * dr2[0]))])
    _, _, V = np.linalg.svd(A) # Calculating SVD decomposition, A = U D (V.T)
    H = V[8].reshape(3, 3) # h is the last column of V. We reshape it into a 3 by 3 matrix to get the homography
    return H / H[2,2]  # normalizing the homography so the last element is 1. We can also just return H but the numbers can be very small


# Normalizing the estimation matrix so that z-value is 1
def normalize(mat):
    for i in range (mat.shape[0]):
        mat[i,:] /= mat[i,2]
    return mat


def RANSAC(coordinates1, coordinates2, threshold, max_iterations=1000):
    sn = 4
    # Adding a third column to represent the z. Assuming normalized values where z=1 for all points.
    coordinates1 = np.hstack([coordinates1, np.ones((coordinates1.shape[0], 1))])
    coordinates2 = np.hstack([coordinates2, np.ones((coordinates2.shape[0], 1))])

    # Generating a set of random values for coordinates picking
    rand_int = np.random.randint(coordinates1.shape[0], size=sn)
    x_points = coordinates1[rand_int,:] # random points from one image
    x_tag_points = coordinates2[rand_int,:] # random points from other image

    # generating a homography using dlt function
    h_best = dlt(x_points, x_tag_points, sn)

    # estimating coordinates 2 using the homography and coordinates 1
    mat = (h_best @ (coordinates1.T)).T

    # Normalizing the estimation matrix so that z-value is 1
    mat = normalize(mat)

    # calculating distance between estimation and actual coordinates 2
    dist2 = mat - coordinates2

    # counting the number of distances less than the threshold
    counter2 = np.count_nonzero(np.sum(np.power(dist2, 2), 1) < threshold)

    # looping
    for i in range(max_iterations):

        # Generating a set of random values for coordinates picking
        rand_int = np.random.randint(coordinates1.shape[0], size=sn)
        x_points = coordinates1[rand_int, :] # random points from one image
        x_tag_points = coordinates2[rand_int, :] # random points from other image

        # generating a homography using dlt function
        h1 = dlt(x_points, x_tag_points, sn)

        # estimating coordinates 2 using the homography and coordinates 1
        mat = (h1 @ (coordinates1.T)).T

        # Normalizing the estimation matrix so that z-value is 1
        mat = normalize(mat)

        # calculating distance between estimation and actual coordinates 2
        dist1 = mat - coordinates2
        pwr =np.power(dist1, 2)

        # counting the number of distances less than the threshold
        counter1=np.count_nonzero(np.sum(pwr, 1) < threshold)

        # Checking if the newly generated homography is better than the currently best homography
        if counter1>counter2:
            # If the new homography causes more distances to be less than the threshold, then it is a better homography
            h_best = h1 # Updating the best homography to be the new homography
            counter2 =counter1 # Updating the counter

    return h_best, counter2  # return best homography, and inliers


def stitch_images(image1, image2, homography):
    # Get images height and width , turn them to 1X2 vectors
    h1, w1 = cv2.imread(image1).shape[:2]
    h2, w2 = cv2.imread(image2).shape[:2]
    corners1 = np.array([[0, 0], [0, h1], [w1, h1], [w1, 0]], dtype=np.float32).reshape(-1, 1, 2)
    corners2 = np.array([[0, 0], [0, h2], [w2, h2], [w2, 0]], dtype=np.float32).reshape(-1, 1, 2)
    # Translate the homography to take under account the field of view of the second image
    corners2_transformed = cv2.perspectiveTransform(corners2, np.linalg.inv(homography).astype(np.float32))
    all_corners = np.concatenate((corners1, corners2_transformed), axis=0)
    x, y, w, h = cv2.boundingRect(all_corners)

    # Adjust the homography matrix to map from img2 to img1
    H_adjusted = np.linalg.inv(homography)

    # Warp the images
    img1_warped = cv2.warpPerspective(cv2.imread(image1), np.eye(3), (w, h))
    img2_warped = cv2.warpPerspective(cv2.imread(image2), H_adjusted, (w, h))

    # Combine the warped images into a single output image
    output = cv2.addWeighted(img1_warped, 0.5, img2_warped, 0.5, 0)
    plt.imshow(img1_warped)
    plt.figure()
    plt.imshow(img2_warped)

    # Create a mask for the overlapping region
    mask1 = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask1, [np.int32(corners1)], (255))
    mask2 = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask2, [np.int32(corners2_transformed)], (255))
    overlap_mask = cv2.bitwise_and(mask1, mask2)
    not_overlap_img2_mask = cv2.bitwise_and(cv2.bitwise_not(overlap_mask), mask2)

    # Blend only the overlapping region

    blended = cv2.addWeighted(img1_warped, 0.5, img2_warped, 0.5, 0)

    # Copy img1_warped and img2_warped to blended using the overlap_mask
    blended = cv2.bitwise_and(blended, blended, mask=overlap_mask)
    blended += cv2.bitwise_and(img1_warped, img1_warped, mask=cv2.bitwise_not(overlap_mask))
    blended += cv2.bitwise_and(img2_warped, img2_warped, mask=not_overlap_img2_mask)

    plt.figure()
    plt.imshow(blended, cmap='gray')
    cv2.imwrite('panoramic_image.jpg', blended)

# Main

# Make sure you use the right path
image1= './Image1.jpg'
image2 = './Image2.jpg'

# Calculate good matches between the images and obtain keypoints
matches, keypoints1, keypoints2 = calculate_matches(image1, image2)
# Extract coordinates of the keypoints
coordinates1 = np.float32([keypoints1[match.queryIdx].pt for match in matches])
coordinates2 = np.float32([keypoints2[match.trainIdx].pt for match in matches])

# RANSAC to find the best homography
homography, inliers = RANSAC(coordinates1, coordinates2, threshold=0.5)

print("inliers:", inliers)
print(homography)

# Stitch the images together
stitch_images(image1, image2, homography)