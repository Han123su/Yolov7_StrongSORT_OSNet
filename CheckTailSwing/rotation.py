import cv2
import numpy as np
import math

def expand_to_square(img, expand_ratio, fill_color=(0, 0, 0)):
    height, width = img.shape[:2]
    max_dim = max(height, width)
    new_height = int(max_dim * expand_ratio)
    new_width = int(max_dim * expand_ratio)

    expanded_img = np.full((new_height, new_width, 3), fill_color, dtype=np.uint8)
    y_offset = (new_height - height) // 2
    x_offset = (new_width - width) // 2

    expanded_img[y_offset:y_offset + height, x_offset:x_offset + width] = img

    return expanded_img, y_offset, x_offset

def transform_point(point, offset):
    return (point[0] + offset[0], point[1] + offset[1])

def align_images(img1, img2, point1, point2, expand_ratio=1.5):
    expanded_img1, y_offset1, x_offset1 = expand_to_square(img1, expand_ratio)
    expanded_img2, y_offset2, x_offset2 = expand_to_square(img2, expand_ratio)

    new_point1 = transform_point(point1[0], (x_offset1, y_offset1))
    new_point2 = transform_point(point2[0], (x_offset2, y_offset2))

    vector1 = np.array(new_point1) - np.array(transform_point(point1[1], (x_offset1, y_offset1)))
    vector2 = np.array(new_point2) - np.array(transform_point(point2[1], (x_offset2, y_offset2)))

    angle = math.atan2(np.linalg.det([vector1, vector2]), np.dot(vector1, vector2))
    angle_deg = np.degrees(angle)

    height, width = expanded_img1.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    rotated_img2 = cv2.warpAffine(expanded_img2, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    aligned_img = rotated_img2

    return aligned_img
    
