import os
import cv2
import scipy
import math
import numpy as np

from math import dist
from skimage import morphology
from skan import csr, Skeleton

from plantcv import plantcv as pcv
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt


def get_data(path):
    im = cv2.imread(str(path))  # , cv2.IMREAD_GRAYSCALE
    return im


def nothing(x):
    pass


def image_processing(img, omega, manual=False):
    winName = 'Colors of the rainbow'
    # cv2.namedWindow(winName)
    if manual:
        omega = omega
        h, s, v = cv2.split((img))
        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(32, 32))
        v = clahe.apply(v)
        s = (s < np.mean(s) - omega * 255) * s
        winName = 'Colors of the rainbow'
        cv2.namedWindow(winName)
        cv2.createTrackbar('LowerbH', winName, 30, 255, nothing)
        cv2.createTrackbar('UpperbH', winName, 70, 255, nothing)
        while (1):
            lowerbH = cv2.getTrackbarPos('LowerbH', winName)
            upperbH = cv2.getTrackbarPos('UpperbH', winName)
            target = np.bitwise_and(h > lowerbH, h < upperbH) * h
            cv2.imshow(winName, target)
            if cv2.waitKey(1) == ord('q'):
                break
        target[target > 0] = 255
        cv2.destroyAllWindows()
    # else:
    #     lowerbH = 30
    #     upperbH = 105
    #     target = np.bitwise_and(h > lowerbH, h < upperbH) * h
    img = cv2.medianBlur(img, 5)
    im1 = img[:, :, 0]

    mask1 = im1 < np.mean(im1)
    im2 = mask1 * im1
    mask2 = im2 < np.mean(im2[im2 > 0])
    # mask_t = im2 < np.mean(im2[im2 > 0]) + np.std(im2[im2 > 0])
    im3 = im2 * mask2
    mask3 = im3 < np.mean(im3[im3 > 0]) + np.std(im3[im3 > 0])
    # cv2.imshow('im1', im1)
    # cv2.imshow('im2', im2)
    # cv2.imshow('im3', im3)
    # cv2.imshow('im1', im2*mask_t)
    # cv2.waitKey(0)
    im4 = im3 * mask3
    g1 = (im4 > 0) * img[:, :, 1]
    # g1 = (im4 > 0) * cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # g1 = (g1 < np.mean(g1[g1 > 0]) + np.std(g1[g1 > 0])) * g1

    ret3, binary = cv2.threshold(g1, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #  morphology
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    # binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    binary = binary > 0
    r_binary = morphology.remove_small_objects(binary, min_size=300)
    binary = scipy.ndimage.binary_fill_holes(r_binary)

    return binary


def get_skeleton(img, omega):
    n = len(np.shape(img))
    thres_min_size = 20
    binary = image_processing(img, omega)

    # obtain binary skeleton
    skeleton = morphology.skeletonize(binary)
    skeleton = skeleton.astype(np.uint8) * 255
    # pcv.params.debug = "plot"
    pruned_skeleton, _, _ = pcv.morphology.prune(skel_img=skeleton, size=40)

    skeleton_img = pruned_skeleton > 0
    skeleton_img = morphology.skeletonize(skeleton_img).astype(bool)
    #######################################################
    graph_class = csr.Skeleton(skeleton_img)
    stats = csr.branch_statistics(graph_class.graph)

    for ii in range(np.size(stats, axis=0)):
        if stats[ii, 2] <= thres_min_size and stats[ii, 3] == 1:
            # remove the branch
            for jj in range(np.size(graph_class.path_coordinates(ii), axis=0)):
                skeleton_img[int(graph_class.path_coordinates(ii)[jj, 0]), int(
                    graph_class.path_coordinates(ii)[jj, 1])] = False

    # during the short branch removing process it can happen that some branches are not longer connected as the complete three branches intersection is removed
    # therefor the remaining skeleton is dilatated and then skeletonized again
    #######################################################
    sk_dilation = morphology.binary_dilation(skeleton_img)
    sk_final = morphology.skeletonize(sk_dilation)
    sk = Skeleton(sk_final)
    path_coor = [sk.path_coordinates(ii) for ii in range(sk.n_paths)]
    skeleton_idx = np.array(
        [path_coor[ix][j] for ix in range(len(path_coor)) for j in range(len(path_coor[ix]))])
    return sk_final.astype(np.uint8) * 255, binary, sk, path_coor, skeleton_idx


def get_postition(line):
    ori_x, ori_y = np.array(line[:, 1]), np.array(line[:, 0])

    total_length = len(ori_x)
    body_idx = (total_length / 2)
    if total_length % 2 > 0:
        body_x = [ori_x[0], ori_x[int(body_idx)], ori_x[int(body_idx * 2) - 1]]  # , ori_x[body_idx * 3 - 1]]
        body_y = [ori_y[0], ori_y[int(body_idx)], ori_y[int(body_idx * 2) - 1]]  # , ori_y[body_idx * 3 - 1]]
    else:
        body_x = [ori_x[0], ori_x[int(body_idx) - 1], ori_x[int(body_idx * 2) - 1]]  # , ori_x[body_idx * 3 - 1]]
        body_y = [ori_y[0], ori_y[int(body_idx) - 1], ori_y[int(body_idx * 2) - 1]]

    return np.c_[list(map(int, body_x)), list(map(int, body_y))], int(body_idx)


def get_body_to_tail(h_idx, skeleton_idx, l):
    if len(skeleton_idx) % 2 == 0:
        if h_idx == 0:
            body_to_tail = skeleton_idx[l - 1:]
        else:
            body_to_tail = skeleton_idx[0:l]
    else:
        if h_idx == 0:
            body_to_tail = skeleton_idx[l:]
        else:
            body_to_tail = skeleton_idx[0:l + 1]

    return body_to_tail


def azimuthOrient(curr, img):
    h, w = np.shape(img)
    w_size = 32
    # 當前位置- 先前位置判斷方向  and 頭尾兩點-中心點去判斷對不對 累積個4次
    angle = []
    area = []
    for i in range(0, len(curr), 2):
        # 面積判斷
        right, bottom = curr[i] + w_size
        left, top = curr[i] - w_size
        if right > w:
            rr = right - w
            left -= rr
            right = w
        if bottom > h:
            br = bottom - h
            top -= br
            bottom = h
        if left < 0:
            lr = left + 1
            right += (-lr)
            left = 0
        if top < 0:
            tr = top + 1
            bottom += (-tr)
            top = 0
        crop = img[top:bottom, left:right]
        c_area = crop[crop > 0].size
        area.append(c_area)

    if area[0] > area[1]:
        return 0, curr
    else:
        temp = curr[0, [0, 1]]
        curr[0, [0, 1]] = curr[2, [0, 1]]
        curr[2, [0, 1]] = temp
        return 2, curr


def get_endpoint(skeleton):
    # Kernel to sum the neighbours
    kernel = [[1, 1, 1],
              [1, 0, 1],
              [1, 1, 1]]
    # 2D convolution (cast image to int32 to avoid overflow)
    img_conv = scipy.signal.convolve2d(skeleton.astype(np.int32), kernel, mode='same')
    # Pick points where pixel is 255 and neighbours sum 255
    endpoints = np.stack(np.where((skeleton == 255) & (img_conv == 255)), axis=1)

    return endpoints


def assignment_point(track, detect):
    tracks = track
    detections = detect
    if len(tracks) == 0:
        tracks = detections

    N = len(tracks)
    _position = [[] for i in range(N)]
    cost = [[] for i in range(N)]
    for i in range(N):
        for j in range(N):
            cost[i].append(dist(tracks[i], detections[j]))
    cost = np.array(cost)
    row_indices, col_indices = linear_sum_assignment(cost)  # row tracking ,,, col detection

    for i in range(len(_position)):
        track_id = row_indices[i]
        detect_id = col_indices[i]
        _position[track_id] = detections[detect_id]
    return np.array(_position)
    # return np.transpose(np.array(_position), [1, 0])
