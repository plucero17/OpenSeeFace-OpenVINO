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


def intersects(r1, r2, amount=0.3):
    area1 = r1[2] * r1[3]
    area2 = r2[2] * r2[3]
    inter = 0.0
    total = area1 + area2

    r1_x1, r1_y1, w, h = r1
    r1_x2 = r1_x1 + w
    r1_y2 = r1_y1 + h
    r2_x1, r2_y1, w, h = r2
    r2_x2 = r2_x1 + w
    r2_y2 = r2_y1 + h

    left = max(r1_x1, r2_x1)
    right = min(r1_x2, r2_x2)
    top = max(r1_y1, r2_y1)
    bottom = min(r1_y2, r2_y2)
    if left < right and top < bottom:
        inter = (right - left) * (bottom - top)
        total -= inter

    if inter / total >= amount:
        return True

    return False


def group_rects(rects):
    rect_groups = {}
    for rect in rects:
        rect_groups[str(rect)] = [-1, -1, []]
    group_id = 0
    for i, rect in enumerate(rects):
        name = str(rect)
        group = group_id
        group_id += 1
        if rect_groups[name][0] < 0:
            rect_groups[name] = [group, -1, []]
        else:
            group = rect_groups[name][0]
        for j, other_rect in enumerate(rects):
            if i == j:
                continue
            if intersects(rect, other_rect):
                rect_groups[str(other_rect)] = [group, -1, []]
    return rect_groups


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
