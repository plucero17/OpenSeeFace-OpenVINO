import math
import numpy as np
import cv2


def clamp_to_im(pt, w, h):
    x = pt[0]
    y = pt[1]
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x >= w:
        x = w - 1
    if y >= h:
        y = h - 1
    return (int(x), int(y + 1))


def rotate(origin, point, a):
    a = -a
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(a) * (px - ox) - math.sin(a) * (py - oy)
    qy = oy + math.sin(a) * (px - ox) + math.cos(a) * (py - oy)
    return qx, qy


def angle(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    a = np.arctan2(*(p2 - p1)[::-1])
    return a % (2 * np.pi)


def compensate(p1, p2):
    a = angle(p1, p2)
    return rotate(p1, p2, a), a


def rotate_image(image, a, center):
    (h, w) = image.shape[:2]
    a = np.rad2deg(a)
    M = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), a, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def logit(p, factor=16.0):
    if p >= 1.0:
        p = 0.9999999
    if p <= 0.0:
        p = 0.0000001
    p = p / (1 - p)
    return float(np.log(p)) / float(factor)


def logit_arr(p, factor=16.0):
    p = np.clip(p, 0.0000001, 0.9999999)
    return np.log(p / (1 - p)) / float(factor)


def matrix_to_quaternion(m):
    t = 0.0
    q = [0.0, 0.0, 0, 0.0]
    if m[2, 2] < 0:
        if m[0, 0] > m[1, 1]:
            t = 1 + m[0, 0] - m[1, 1] - m[2, 2]
            q = [t, m[0, 1] + m[1, 0], m[2, 0] + m[0, 2], m[1, 2] - m[2, 1]]
        else:
            t = 1 - m[0, 0] + m[1, 1] - m[2, 2]
            q = [m[0, 1] + m[1, 0], t, m[1, 2] + m[2, 1], m[2, 0] - m[0, 2]]
    else:
        if m[0, 0] < -m[1, 1]:
            t = 1 - m[0, 0] - m[1, 1] + m[2, 2]
            q = [m[2, 0] + m[0, 2], m[1, 2] + m[2, 1], t, m[0, 1] - m[1, 0]]
        else:
            t = 1 + m[0, 0] + m[1, 1] + m[2, 2]
            q = [m[1, 2] - m[2, 1], m[2, 0] - m[0, 2], m[0, 1] - m[1, 0], t]
    q = np.array(q, np.float32) * 0.5 / np.sqrt(t)
    return q
