import numpy as np
import math
import cv2
import random
import torch
from PIL import Image, ImageFilter


def get_meanface(meanface_file):
    """get meanface array from corresponding meanface txt file"""
    with open(meanface_file) as f:
        meanface = f.readlines()[0]    
    
    meanface = meanface.strip().split()
    meanface = [float(x) for x in meanface]
    meanface = np.array(meanface).reshape(-1, 2)
    
    return meanface

def _gaussian(
        size=3, sigma=0.25, amplitude=1, normalize=False, width=None,
        height=None, sigma_horz=None, sigma_vert=None, mean_horz=0.5,
        mean_vert=0.5):
    # handle some defaults
    if width is None:
        width = size
    if height is None:
        height = size
    if sigma_horz is None:
        sigma_horz = sigma
    if sigma_vert is None:
        sigma_vert = sigma
    center_x = mean_horz * width + 0.5
    center_y = mean_vert * height + 0.5
    gauss = np.empty((height, width), dtype=np.float32)
    # generate kernel
    for i in range(height):
        for j in range(width):
            gauss[i][j] = amplitude * math.exp(-(math.pow((j + 1 - center_x) / (
                sigma_horz * width), 2) / 2.0 + math.pow((i + 1 - center_y) / (sigma_vert * height), 2) / 2.0))
    if normalize:
        gauss = gauss / np.sum(gauss)
    return gauss

def draw_gaussian(image, point, sigma):
    '''generate heatmap'''

    # Check if the gaussian is inside
    ul = [np.floor(np.floor(point[0]) - 3 * sigma),
          np.floor(np.floor(point[1]) - 3 * sigma)]
    br = [np.floor(np.floor(point[0]) + 3 * sigma),
          np.floor(np.floor(point[1]) + 3 * sigma)]
    if (ul[0] > image.shape[1] or ul[1] >
            image.shape[0] or br[0] < 1 or br[1] < 1):
        return image
    size = 6 * sigma + 1
    g = _gaussian(size)
    g_x = [int(max(1, -ul[0])), int(min(br[0], image.shape[1])) -
           int(max(1, ul[0])) + int(max(1, -ul[0]))]
    g_y = [int(max(1, -ul[1])), int(min(br[1], image.shape[0])) -
           int(max(1, ul[1])) + int(max(1, -ul[1]))]
    img_x = [int(max(1, ul[0])), int(min(br[0], image.shape[1]))]
    img_y = [int(max(1, ul[1])), int(min(br[1], image.shape[0]))]
    assert (g_x[0] > 0 and g_y[1] > 0)
    correct = False
    while not correct:
        try:
            image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]
            ] = image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]] + g[g_y[0] - 1:g_y[1], g_x[0] - 1:g_x[1]]
            correct = True
        except:
            print('img_x: {}, img_y: {}, g_x:{}, g_y:{}, point:{}, g_shape:{}, ul:{}, br:{}'.format(img_x, img_y, g_x, g_y, point, g.shape, ul, br))
            ul = [np.floor(np.floor(point[0]) - 3 * sigma),
                np.floor(np.floor(point[1]) - 3 * sigma)]
            br = [np.floor(np.floor(point[0]) + 3 * sigma),
                np.floor(np.floor(point[1]) + 3 * sigma)]
            g_x = [int(max(1, -ul[0])), int(min(br[0], image.shape[1])) -
                int(max(1, ul[0])) + int(max(1, -ul[0]))]
            g_y = [int(max(1, -ul[1])), int(min(br[1], image.shape[0])) -
                int(max(1, ul[1])) + int(max(1, -ul[1]))]
            img_x = [int(max(1, ul[0])), int(min(br[0], image.shape[1]))]
            img_y = [int(max(1, ul[1])), int(min(br[1], image.shape[0]))]
            pass
    image[image > 1] = 1
    return image


def _gaussian_gpu(
        size=3, sigma=0.25, amplitude=1, normalize=False, width=None,
        height=None, sigma_horz=None, sigma_vert=None, mean_horz=0.5,
        mean_vert=0.5, device="cpu"):
    # handle some defaults
    if width is None:
        width = size
    if height is None:
        height = size
    if sigma_horz is None:
        sigma_horz = sigma
    if sigma_vert is None:
        sigma_vert = sigma
    center_x = mean_horz * width + 0.5
    center_y = mean_vert * height + 0.5
    gauss = torch.empty((height, width), dtype=torch.float32).to(device)
    # generate kernel
    for i in range(height):
        for j in range(width):
            gauss[i][j] = amplitude * math.exp(-(math.pow((j + 1 - center_x) / (
                sigma_horz * width), 2) / 2.0 + math.pow((i + 1 - center_y) / (sigma_vert * height), 2) / 2.0))
    if normalize:
        gauss = gauss / torch.sum(gauss)
    return gauss

def gen_heatmap(image, point, sigma):
    '''generate heatmap on gpu'''

    # Check if the gaussian is inside
    ul = [torch.floor(torch.floor(point[0]) - 3 * sigma),
          torch.floor(torch.floor(point[1]) - 3 * sigma)]
    br = [torch.floor(torch.floor(point[0]) + 3 * sigma),
          torch.floor(torch.floor(point[1]) + 3 * sigma)]
    if (ul[0] > image.shape[1] or ul[1] >
            image.shape[0] or br[0] < 1 or br[1] < 1):
        return image
    size = 6 * sigma + 1
    g = _gaussian_gpu(size, device=image.device)
    g_x = [int(max(1, -ul[0])), int(min(br[0], image.shape[1])) -
           int(max(1, ul[0])) + int(max(1, -ul[0]))]
    g_y = [int(max(1, -ul[1])), int(min(br[1], image.shape[0])) -
           int(max(1, ul[1])) + int(max(1, -ul[1]))]
    img_x = [int(max(1, ul[0])), int(min(br[0], image.shape[1]))]
    img_y = [int(max(1, ul[1])), int(min(br[1], image.shape[0]))]
    assert (g_x[0] > 0 and g_y[1] > 0)
    correct = False
    while not correct:
        try:
            image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]
            ] = image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]] + g[g_y[0] - 1:g_y[1], g_x[0] - 1:g_x[1]]
            correct = True
        except:
            print('img_x: {}, img_y: {}, g_x:{}, g_y:{}, point:{}, g_shape:{}, ul:{}, br:{}'.format(img_x, img_y, g_x, g_y, point, g.shape, ul, br))
            ul = [torch.floor(torch.floor(point[0]) - 3 * sigma),
                torch.floor(torch.floor(point[1]) - 3 * sigma)]
            br = [torch.floor(torch.floor(point[0]) + 3 * sigma),
                torch.floor(torch.floor(point[1]) + 3 * sigma)]
            g_x = [int(max(1, -ul[0])), int(min(br[0], image.shape[1])) -
                int(max(1, ul[0])) + int(max(1, -ul[0]))]
            g_y = [int(max(1, -ul[1])), int(min(br[1], image.shape[0])) -
                int(max(1, ul[1])) + int(max(1, -ul[1]))]
            img_x = [int(max(1, ul[0])), int(min(br[0], image.shape[1]))]
            img_y = [int(max(1, ul[1])), int(min(br[1], image.shape[0]))]
            pass
    image[image > 1] = 1
    return image

def crop_norm(image, landmark, margin, image_size):      
    """ crop image and normlize landmark"""
    image_height, image_width, _ = image.shape    
    anno_x , anno_y = landmark[:, 0], landmark[:, 1]
    xmin, xmax, ymin, ymax = np.min(anno_x), np.max(anno_x), np.min(anno_y), np.max(anno_y)
    box_w, box_h = (xmax-xmin), (ymax-ymin)

    margin = margin
    # print(margin, xmin, xmax, ymin, ymax)
    xmin = max(int(xmin - margin), 0)
    xmax = min(int(xmax + margin), image_width - 1)
    ymin = max(int(ymin - margin), 0)
    ymax = min(int(ymax + margin), image_height - 1)
    box_w, box_h = (xmax-xmin), (ymax-ymin)

    anno_x = (anno_x - xmin) / box_w
    anno_y = (anno_y - ymin) / box_h
    norm_landmarks = np.concatenate([anno_x.reshape(-1,1), anno_y.reshape(-1,1)], axis=1)  
    image_crop = image[int(ymin):int(ymax), int(xmin):int(xmax), :]
    image_crop = cv2.resize(image_crop, (image_size, image_size))

    return image_crop, norm_landmarks

def random_translate(image, landmark, pose):
    if random.random() > 0.5:
        image_height, image_width = image.size
        a = 1
        b = 0
        #c = 30 #left/right (i.e. 5/-5)
        c = int((random.random()-0.5) * 60)
        d = 0
        e = 1
        #f = 30 #up/down (i.e. 5/-5)
        f = int((random.random()-0.5) * 60)
        image = image.transform(image.size, Image.AFFINE, (a, b, c, d, e, f))
        landmark_translate = landmark.copy()
        landmark_translate[:, 0] -= 1.*c/image_width
        landmark_translate[:, 1] -= 1.*f/image_height
        landmark_translate = landmark_translate.flatten()
        landmark_translate[landmark_translate < 0] = 0
        landmark_translate[landmark_translate > 1] = 1
        landmark_translate = landmark_translate.reshape(-1,2)
        return image, landmark_translate, pose
    else:
        return image, landmark, pose

def random_blur(image, landmark, pose):
    if random.random() > 0.7:
        image = image.filter(ImageFilter.GaussianBlur(random.random()*5))
    return image, landmark, pose

def random_occlusion(image, landmark, pose):
    if random.random() > 0.5:
        image_np = np.array(image).astype(np.uint8)
        image_np = image_np[:,:,::-1]
        image_height, image_width, _ = image_np.shape
        occ_height = int(image_height*0.4*random.random())
        occ_width = int(image_width*0.4*random.random())
        occ_xmin = int((image_width - occ_width - 10) * random.random())
        occ_ymin = int((image_height - occ_height - 10) * random.random())
        image_np[occ_ymin:occ_ymin+occ_height, occ_xmin:occ_xmin+occ_width, 0] = int(random.random() * 255)
        image_np[occ_ymin:occ_ymin+occ_height, occ_xmin:occ_xmin+occ_width, 1] = int(random.random() * 255)
        image_np[occ_ymin:occ_ymin+occ_height, occ_xmin:occ_xmin+occ_width, 2] = int(random.random() * 255)
        image_pil = Image.fromarray(image_np[:,:,::-1].astype('uint8'), 'RGB')
        return image_pil, landmark, pose
    else:
        return image, landmark, pose

def random_flip(image, landmark, pose, points_flip):
    if random.random() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        landmark = landmark[points_flip, :]
        landmark[:,0] = 1-landmark[:,0]
        landmark = landmark
        pose[0] = -pose[0]
        pose[2] = -pose[2]
        return image, landmark, pose
    else:
        return image, landmark, pose

def random_rotate(image, landmark, pose, angle_max=30):
    if random.random() > 0.5:
        center_x = 0.5
        center_y = 0.5
        landmark_center = landmark - np.array([center_x, center_y])
        theta_max = np.radians(angle_max)
        theta = random.uniform(-theta_max, theta_max)
        angle = np.degrees(theta)
        image = image.rotate(angle)

        c, s = np.cos(theta), np.sin(theta)
        rot = np.array(((c,-s), (s, c)))
        landmark_center_rot = np.matmul(landmark_center, rot)
        landmark_rot = landmark_center_rot + np.array([center_x, center_y])
        return image, landmark_rot, pose
    else:
        return image, landmark, pose