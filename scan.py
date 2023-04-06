import os
import numpy as np
import cv2

# 设置所需的最小匹配数
MIN_MATCH_COUNT = 15
# 初始化 SIFT 检测器
SIFT = cv2.xfeatures2d.SIFT_create()
# 初始化 FLANN 匹配器
FLANN_INDEX_KDTREE = 1
INDEX_PARAMS = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
SEARCH_PARAMS = dict(checks=50)
FLANN = cv2.FlannBasedMatcher(INDEX_PARAMS, SEARCH_PARAMS)

# 匹配函数


def flann(img1, kp1, des1, img2, kp2, des2, matchPoint) -> bool:
    # 匹配两张图片的描述符
    matches = FLANN.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        # 有效距离
        if m.distance < 0.6 * n.distance:
            good.append(m)
    # 检查是否找到足够好的匹配
    return len(good) > matchPoint


# 样例字典
simples = {}
# 目标图集
targets = {}
# 匹配结果
results = {}

# 样例图目录
SIMPLE_DIR = 'cc'
for file in os.listdir(SIMPLE_DIR):
    print('Loading %s.' % file)
    img = cv2.imread(os.path.join(SIMPLE_DIR, file), 0)
    kp, des = SIFT.detectAndCompute(img, None)
    simples[file] = {'img': img, 'kp': kp, 'des': des}
print('Loaded %d sample images.' % len(simples))

# 目标图片目录
TARGET_DIR = 'target'
for file in os.listdir(TARGET_DIR):
    print('Loading %s.' % file)
    img = cv2.imread(os.path.join(TARGET_DIR, file), 0)
    kp, des = SIFT.detectAndCompute(img, None)
    targets[file] = {'img': img, 'kp': kp, 'des': des}
print('Loaded %d target images.' % len(targets))

# 匹配
for k1, v1 in targets.items():
    temp = []
    for k2, v2 in simples.items():
        if flann(v2['img'], v2['kp'], v2['des'], v1['img'], v1['kp'], v1['des'], MIN_MATCH_COUNT):
            temp.append(k2)
            print('%s matched %s.' % (k1, k2))
    results[k1] = temp
