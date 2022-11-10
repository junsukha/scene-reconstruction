import numpy as np
import cv2
import random

def calculate_projection_matrix(image, markers):
    """
    To solve for the projection matrix. You need to set up a system of
    equations using the corresponding 2D and 3D points. See the handout, Q5
    of the written questions, or the lecture slides for how to set up these
    equations.

    Don't forget to set M_34 = 1 in this system to fix the scale.

    :param image: a single image in our camera system
    :param markers: dictionary of markerID to 4x3 array containing 3D points
    :return: M, the camera projection matrix which maps 3D world coordinates
    of provided aruco markers to image coordinates
    """
    ######################
    # Do not change this #
    ######################

    # Markers is a dictionary mapping a marker ID to a 4x3 array
    # containing the 3d points for each of the 4 corners of the
    # marker in our scanning setup
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000)
    parameters = cv2.aruco.DetectorParameters_create()

    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(
        image, dictionary, parameters=parameters)
    markerIds = [m[0] for m in markerIds]
    markerCorners = [m[0] for m in markerCorners]

    points2d = []
    points3d = []

    for markerId, marker in zip(markerIds, markerCorners):
        if markerId in markers:
            for j, corner in enumerate(marker):
                points2d.append(corner)
                points3d.append(markers[markerId][j])

    points2d = np.array(points2d)
    points3d = np.array(points3d)

    ########################
    # TODO: Your code here #
    ########################
    # This M matrix came from a call to rand(3,4). It leads to a high residual.
    
    
    numOfPoints = points2d.shape[0]
    A = []
    
    for i in range(numOfPoints):        
        A.append([points3d[i,0], points3d[i,1], points3d[i,2], 1, 0, 0, 0, 0, -points3d[i,0]*points2d[i,0], -points3d[i,1]*points2d[i,0], -points3d[i,2]*points2d[i,0]]) 
        
        A.append([0,0,0,0, points3d[i,0], points3d[i,1], points3d[i,2], 1, -points3d[i,0]*points2d[i,1], -points3d[i,1]*points2d[i,1], -points3d[i,2]*points2d[i,1]])
    
    A = np.array(A) 
    b = points2d.reshape(-1,1)   

    M = np.linalg.lstsq(A, b)[0]
    M = np.append(M,1).reshape(3,4)

    return M

def normalize_coordinates(points):
    """
    ============================ EXTRA CREDIT ============================
    Normalize the given Points before computing the fundamental matrix. You
    should perform the normalization to make the mean of the points 0
    and the average magnitude 1.0.

    The transformation matrix T is the product of the scale and offset matrices

    Offset Matrix
    Find c_u and c_v and create a matrix of the form in the handout for T_offset

    Scale Matrix
    Subtract the means of the u and v coordinates, then take the reciprocal of
    their standard deviation i.e. 1 / np.std([...]). Then construct the scale
    matrix in the form provided in the handout for T_scale

    :param points: set of [n x 2] 2D points
    :return: a tuple of (normalized_points, T) where T is the [3 x 3] transformation
    matrix
    """
    ########################
    # TODO: Your code here #
    ########################
    # This is a placeholder with the identity matrix for T replace with the
    # real transformation matrix for this set of points
    T = np.eye(3)

    return points, T

def estimate_fundamental_matrix(points1, points2):
    """
    Estimates the fundamental matrix given set of point correspondences in
    points1 and points2.

    points1 is an [n x 2] matrix of 2D coordinate of points on Image A
    points2 is an [n x 2] matrix of 2D coordinate of points on Image B

    Try to implement this function as efficiently as possible. It will be
    called repeatedly for part IV of the project

    If you normalize your coordinates for extra credit, don't forget to adjust
    your fundamental matrix so that it can operate on the original pixel
    coordinates!

    :return F_matrix, the [3 x 3] fundamental matrix
    """
    ########################
    # TODO: Your code here #
    ########################
    # Af = 0
    # make matrix A
    A = []
    for i in range(len(points1)):
        A.append([points1[i,0]*points2[i,0], points1[i,0]*points2[i,1], points1[i,0], points1[i,1]*points2[i,0], points1[i,1]*points2[i,1], points1[i,1], points2[i,0], points2[i,1], 1])
    
    A=np.array(A)
    U, S, Vh = np.linalg.svd(A)
    F = Vh[-1, :]
    F = np.reshape(F, (3,3))
    
    U, S, Vh = np.linalg.svd(F)
    S[-1] = 0
    F = U @ np.diagflat(S) @ Vh
    
    # This is an intentionally incorrect Fundamental matrix placeholder
    # F_matrix = np.array([[0, 0, -.0004], [0, 0, .0032], [0, -0.0044, .1034]])

    return F.T
    # return F_matrix

def ransac_fundamental_matrix(matches1, matches2, num_iters):
    """
    Find the best fundamental matrix using RANSAC on potentially matching
    points. Run RANSAC for num_iters.

    matches1 and matches2 are the [N x 2] coordinates of the possibly
    matching points from two pictures. Each row is a correspondence
     (e.g. row 42 of matches1 is a point that corresponds to row 42 of matches2)

    best_Fmatrix is the [3 x 3] fundamental matrix, inliers1 and inliers2 are
    the [M x 2] corresponding points (some subset of matches1 and matches2) that
    are inliners with respect to best_Fmatrix

    For this section, use RANSAC to find the best fundamental matrix by randomly
    sampling interest points. You would call the function that estimates the 
    fundamental matrix (either the "cheat" function or your own 
    estimate_fundamental_matrix) iteratively within this function.

    If you are trying to produce an uncluttered visualization of epipolar lines,
    you may want to return no more than 30 points for either image.

    :return: best_Fmatrix, inliers1, inliers2
    """
    # DO NOT TOUCH THE FOLLOWING LINES
    random.seed(0)
    np.random.seed(0)
    
    ########################
    # TODO: Your code here #
    ########################

    # Your RANSAC loop should contain a call to 'estimate_fundamental_matrix()'
    # that you wrote for part II.
    prev_score = 0
    # print(f'num_iters: {num_iters}')
    for i in range(num_iters):
        
        # random several same rows from each matches1 and matches2
        rand_indices = np.random.choice(len(matches1), 8, replace=False)
        
        # used F_matrix, _ = cv2.findFundamentalMat(small_points1_array, small_points2_array, cv2.FM_8POINT, 1e10, 0, 1)
        # Fmatrix = estimate_fundamental_matrix(rows from matches1, rows from matches2)
        # i use these variables only to calcuate F_matrix
        rand_rows_matches1 = np.array(matches1[rand_indices]) # shape = 8*2 maybe..
        rand_rows_matches2 = np.array(matches2[rand_indices])
        
        
        # F_matrix, _ = cv2.findFundamentalMat(rand_rows_matches1, rand_rows_matches2, cv2.FM_8POINT, 1e10, 0, 1)
        F_matrix = estimate_fundamental_matrix(rand_rows_matches1, rand_rows_matches2)
        
        # since xFx'=0, distances =sum(row from matches1 @ Fmatrix @ row from matches2 - 0)
        # i think I should use all matches1 for x and the same for x'
        add_1 = np.ones(shape=(matches1.shape[0],1))
        matches1_add_1 = np.concatenate((matches1, add_1), axis=1)
        matches2_add_1 = np.concatenate((matches2, add_1), axis=1)
        
        errors= np.matrix.diagonal(matches1_add_1 @ F_matrix @ matches2_add_1.T) # pick only diagonal elements        
        
        # calculate score = number of correspondecs / total correspondeces
        # count = np.count_nonzero(abs(errors) < 0.005)
        score = np.count_nonzero(abs(errors) < 0.005) / len(errors)
        
        # used initilized score ( which will be set to 0) and if score > previous score 
        if score > prev_score:
            # pick correspondences that distance < my_choice_threshold
            # also count the number of correspondences
            best_Fmatrix = F_matrix
            inliers_a = matches1[abs(errors) < 0.005]
            inliers_b = matches2[abs(errors) < 0.005]
            prev_score = score
        
        
        


    # best_Fmatrix = estimate_fundamental_matrix(matches1[0:9, :], matches2[0:9, :])
    # inliers_a = matches1[0:29, :]
    # inliers_b = matches2[0:29, :]

    return best_Fmatrix, inliers_a, inliers_b

def matches_to_3d(points1, points2, M1, M2):
    """
    Given two sets of points and two projection matrices, you will need to solve
    for the ground-truth 3D points using np.linalg.lstsq(). For a brief reminder
    of how to do this, please refer to Question 5 from the written questions for
    this project.


    :param points1: [N x 2] points from image1
    :param points2: [N x 2] points from image2
    :param M1: [3 x 4] projection matrix of image2
    :param M2: [3 x 4] projection matrix of image2
    :return: [N x 3] NumPy array of solved ground truth 3D points for each pair of 2D
    points from points1 and points2
    """
    ########################
    # TODO: Your code here #

    # Fill in the correct shape
    points3d = np.zeros((len(points1), 3))

    # Solve for ground truth points
    
    # A * 3dPoint = b
    
    
    for i in range(len(points1)):
        # A and b should be reset for every i
        # constrcut A
        A = [] # final shape should be ((4*len(points1), 3))
        b = []
        
        # using a points from image1 and project matrix of image1
        A.append([M1[2,0]*points1[i,0]-M1[0,0], M1[2,1]*points1[i,0]-M1[0,1], M1[2,2]*points1[i,0]-M1[0,2]])
        A.append([M1[2,0]*points1[i,1]-M1[1,0], M1[2,1]*points1[i,1]-M1[1,1], M1[2,2]*points1[i,1]-M1[1,2]])
        b.append([M1[0,3] - points1[i,0]*M1[2,3]])
        b.append([M1[1,3] - points1[i,1]*M1[2,3]])
        
        # using a points from image2 and project matrix of image2
        A.append([M2[2,0]*points2[i,0]-M2[0,0], M2[2,1]*points2[i,0]-M2[0,1], M2[2,2]*points2[i,0]-M2[0,2]])
        A.append([M2[2,0]*points2[i,1]-M2[1,0], M2[2,1]*points2[i,1]-M2[1,1], M2[2,2]*points2[i,1]-M2[1,2]])
        b.append([M2[0,3] - points2[i,0]*M2[2,3]])
        b.append([M2[1,3] - points2[i,1]*M2[2,3]])
        
        A = np.array(A)
        b = np.array(b)
        np.linalg.lstsq(A, b)
        
        point3d = np.linalg.lstsq(A, b)[0] #point3d shape is (3,1)
        
        points3d[i] = point3d.reshape(3,)
    ########################

    return points3d
