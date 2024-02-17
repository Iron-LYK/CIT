# ------------------------------------------------------------------------------------
# This file is used for loading the data and annotations of dataset.
# Copyright © 2022 Li Yao-kun <liyk58@mail2.sysu.edu.cn>
# To find more details, please refer to: https://github.com/Iron-LYK/CIT
# ------------------------------------------------------------------------------------
 
import os 
import cv2
import hdf5storage
import numpy as np
import scipy.io as scio
import PIL.Image as Image
from PIL import ImageFile
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from .dataset_utils import *

ImageFile.LOAD_TRUNCATED_IMAGES = True

def visual_landmarks(image, landmarks):
	for i in range(len(landmarks)):
		x = landmarks[i][0]
		y = landmarks[i][1]
		image = cv2.circle(image, (int(x),int(y)), radius=1, color=(0, 0, 255), thickness=1)
	cv2.imshow("visual_landmarks", image)
	cv2.waitKey(0)

class Dataset_COFW(Dataset):
    def __init__(self, cfg, dataset):
        self.data_name = "COFW"
        self.root = cfg.DATASET.ROOT+ "/" + self.data_name + "/"
        self.imgsize = cfg.DATASET.IMAGESIZE
        self.margin = cfg.DATASET.MARGIN
        self.mode = dataset
        self.aug = cfg.DATASET.AUGMENT
        self.randomgray = transforms.RandomGrayscale(0.2)
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])])
        points_flip = [2, 1, 4, 3, 7, 8, 5, 6, 10, 9, 12, 11, 15, 16, 13, 14, 18, 17, 20, 19, 21, 22, 24, 23, 25, 26, 27, 28, 29]
        self.points_flip = (np.array(points_flip)-1).tolist()        
        self.load_data(self.mode)
    
    def load_data(self, mode):
        self.mat_file = hdf5storage.loadmat(self.root+"COFW_color.mat")
        self.pose = np.load((self.root + "COFW_pose.npy"))
        if mode == "train":
            self.images = self.mat_file['IsTr']
            self.annos = self.mat_file['phisTr']
        else:
            self.images = self.mat_file['IsT']
            self.annos = self.mat_file['phisT']    

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):        

        anno = self.annos[index, :] 
        visibility = self.annos[index, 58:]
        pose = self.pose[index, :] 

        img = self.images[index,0]        
        if len(img.shape) == 2:     # grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)        
        else:                       
            img = img[:,:,::-1]     # swap rgb channel to bgr
        landmarks = np.array([anno[:29], anno[29:58]]).T
        img, norm_landmarks = crop_norm(img, landmarks, self.margin, self.imgsize)              
         
        # data augmentation                               
        img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        if self.mode == "train":
            for type in self.aug:                
                type_name = eval("random_" + type)
                if type == "flip":
                    img, norm_landmarks, pose = type_name(img, norm_landmarks, pose, self.points_flip)
                else:
                    img, norm_landmarks, pose = type_name(img, norm_landmarks, pose)                
            img = self.randomgray(img)
        orig_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        landmarks = norm_landmarks*img.size[0]
        img = self.transform(img)        
        
        # generate heatmap
        heatmap = np.zeros((landmarks.shape[0], 64, 64))
        for j in range(landmarks.shape[0]):
            if landmarks[j][0] > 0:
                heatmap[j] = draw_gaussian(heatmap[j], landmarks[j]/4.0+1, 1)                             

        return img, orig_img, heatmap, landmarks, visibility, pose       

class Dataset_MERL_RAV_FLOP(Dataset):   
    def __init__(self, cfg, dataset):
        
        self.root = cfg.DATASET.ROOT+ "/MERL_RAV_FLOP"
        file_path = self.root + "/{}_files.txt".format(dataset)
        with open(file_path) as f:
            self.imgnames = f.read().splitlines() 
        
        self.imgsize = cfg.DATASET.IMAGESIZE
        self.margin = cfg.DATASET.MARGIN
        self.aug = cfg.DATASET.AUGMENT
        self.mode = dataset
        self.resize = transforms.Resize((self.imgsize, self.imgsize))
        self.randomgray = transforms.RandomGrayscale(0.2)
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])])
        points_flip = [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 28, 29, 30, 31, 36, 35, 34, 33, 32, 46, 45, 44, 43, 48, 47, 40, 39, 38, 37, 42, 41, 55, 54, 53, 52, 51, 50, 49, 60, 59, 58, 57, 56, 65, 64, 63, 62, 61, 68, 67, 66]
        self.points_flip = (np.array(points_flip)-1).tolist()                            

    def __len__(self):
        return len(self.imgnames)
    
    def __getitem__(self, index):        
        img_path = self.imgnames[index]
        
        img = Image.open(img_path).convert('RGB')
        anno = np.load(img_path.replace("jpg","npy"))       
        visibility = anno[-73:-5]
        pose = anno[-5:-2]   
        bbox_w, bbox_h = anno[-2], anno[-1]

        # Crop the face loosely
        pt2d = np.array([anno[:68],anno[68:136]]).T   
        x_min = min(pt2d[:, 0])
        y_min = min(pt2d[:, 1])
        x_max = max(pt2d[:, 0])
        y_max = max(pt2d[:, 1])
        anno_x , anno_y = pt2d[:, 0], pt2d[:, 1]     

        ad = self.margin
        h = y_max-y_min
        w = x_max-x_min        
        x_min = max(int(x_min - ad * w), 0)
        x_max = min(int(x_max + ad * w), img.width - 1)
        y_min = max(int(y_min - (ad+0.25) * h), 0)
        y_max = min(int(y_max + ad * h), img.height - 1)

        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
        anno_x = (anno_x - x_min) / (x_max-x_min)
        anno_y = (anno_y - y_min) / (y_max-y_min)
        norm_landmarks = np.concatenate([anno_x.reshape(-1,1), anno_y.reshape(-1,1)], axis=1)                             

        img = self.resize(img)
        norm_factor = np.sqrt((bbox_h*self.imgsize/(y_max - y_min))*(bbox_w*self.imgsize/(x_max - x_min)))      
        orig_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)                
        landmarks = norm_landmarks*img.size[0]         
        img = self.transform(img)                
        
        # generate heatmap
        heatmap = np.zeros((landmarks.shape[0], 64, 64))
        for j in range(landmarks.shape[0]):
            if landmarks[j][0] > 0:
                heatmap[j] = draw_gaussian(heatmap[j], landmarks[j]/4.0+1, 1)               

        return img, orig_img, heatmap, landmarks, visibility, pose, norm_factor


class Dataset_BIWI(Dataset):    
    def __init__(self, cfg, dataset):
        self.mode = dataset.split("/")[-1]
        self.root = cfg.DATASET.ROOT+ "/BIWI/" + dataset
        self.aug = cfg.DATASET.AUGMENT
        self.imgsize = cfg.DATASET.IMAGESIZE
        self.margin = cfg.DATASET.MARGIN
        self.randomgray = transforms.RandomGrayscale(0.2)
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])])
        points_flip = [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 28, 29, 30, 31, 36, 35, 34, 33, 32, 46, 45, 44, 43, 48, 47, 40, 39, 38, 37, 42, 41, 55, 54, 53, 52, 51, 50, 49, 60, 59, 58, 57, 56, 65, 64, 63, 62, 61, 68, 67, 66]
        self.points_flip = (np.array(points_flip)-1).tolist()        
        self.load_data()
    
    def load_data(self):        
        self.imgs = np.load((self.root + "/"+ "img.npy"),allow_pickle=True) 
        self.landmark = np.load((self.root + "/"+ "landmark.npy"),allow_pickle=True)
        self.pose = np.load((self.root + "/"+ "pose.npy"),allow_pickle=True)                           


    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):        

        img = self.imgs[index]
        landmark = self.landmark[index][:,:2]
        pose = self.pose[index, :]
        visibility = np.ones((68))
                       
        image_height, image_width, _ = img.shape
        norm_landmarks = np.ones((68,2))
        norm_landmarks[:, 0] = landmark[:, 0]/image_width
        norm_landmarks[:, 1] = landmark[:, 1]/image_height                    
                
        # data augmentation                               
        img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))          
        if self.mode == "train":
            for type in self.aug:                
                type_name = eval("random_" + type)
                if type == "flip":
                    img, norm_landmarks, pose = type_name(img, norm_landmarks, pose, self.points_flip)
                else:
                    img, norm_landmarks, pose = type_name(img, norm_landmarks, pose)                
            img = self.randomgray(img)    
                
        orig_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        landmarks = norm_landmarks*img.size[0]
        img = self.transform(img)        
        
        # generate heatmap
        heatmap = np.zeros((landmarks.shape[0], 64, 64))
        for j in range(landmarks.shape[0]):
            if landmarks[j][0] > 0:
                heatmap[j] = draw_gaussian(heatmap[j], landmarks[j]/4.0+1, 1)                        

        return img, orig_img, heatmap, landmarks, visibility, pose 

class Dataset_AFLW2000(Dataset):    
    def __init__(self, cfg, dataset):
        file_path = cfg.DATASET.ROOT+ "/AFLW2000/filename.txt"
        with open(file_path) as f:
            self.imgnames = f.read().splitlines()        
        self.data_path = dataset
        self.imgsize = cfg.DATASET.IMAGESIZE
        self.margin = cfg.DATASET.MARGIN
        self.resize = transforms.Resize((256, 256))
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])])
        points_flip = [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 28, 29, 30, 31, 36, 35, 34, 33, 32, 46, 45, 44, 43, 48, 47, 40, 39, 38, 37, 42, 41, 55, 54, 53, 52, 51, 50, 49, 60, 59, 58, 57, 56, 65, 64, 63, 62, 61, 68, 67, 66]
        self.points_flip = (np.array(points_flip)-1).tolist()                                   

    def __len__(self):
        return len(self.imgnames)
    
    def __getitem__(self, index):        

        img = Image.open(os.path.join(self.data_path, self.imgnames[index]))
        mat = scio.loadmat(os.path.join(self.data_path, self.imgnames[index].replace("jpg","mat")))

        pose_para = mat['Pose_Para'][0]
        pitch = pose_para[0] * 180 / np.pi
        yaw = pose_para[1] * 180 / np.pi
        roll = pose_para[2] * 180 / np.pi     
        pose = np.array([yaw, pitch, roll])           
        visibility = np.ones((68))     

        # Crop the face loosely
        pt = mat["pt3d_68"][:2, :]  
        pt_x , pt_y = pt[0, :], pt[1, :]
        x_min = min(pt_x)
        y_min = min(pt_y)
        x_max = max(pt_x)
        y_max = max(pt_y)      

        ad = self.margin
        h = y_max-y_min
        w = x_max-x_min        
        x_min = max(int(x_min - ad * w), 0)
        x_max = min(int(x_max + ad * w), img.width - 1)
        y_min = max(int(y_min - (ad+0.25) * h), 0)
        y_max = min(int(y_max + ad * h), img.height - 1)

        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
        pt_x = (pt_x - x_min) / (x_max-x_min)
        pt_y = (pt_y - y_min) / (y_max-y_min)
        norm_landmarks = np.concatenate([pt_x.reshape(-1,1), pt_y.reshape(-1,1)], axis=1)                             

        img = self.resize(img)
        orig_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        landmarks = norm_landmarks*img.size[0]
        img = self.transform(img)                
        
        # generate heatmap
        heatmap = np.zeros((landmarks.shape[0], 64, 64))
        for j in range(landmarks.shape[0]):
            if landmarks[j][0] > 0:
                heatmap[j] = draw_gaussian(heatmap[j], landmarks[j]/4.0+1, 1)                        

        return img, orig_img, heatmap, landmarks, visibility, pose

class Dataset_300W_LP(Dataset):    
    def __init__(self, cfg, dataset):
        file_path = cfg.DATASET.ROOT+ "/300W_LP/filename.txt"
        with open(file_path) as f:
            self.imgnames = f.read().splitlines()        
        self.data_path = dataset
        self.imgsize = cfg.DATASET.IMAGESIZE
        self.margin = cfg.DATASET.MARGIN
        self.aug = cfg.DATASET.AUGMENT
        self.resize = transforms.Resize((256, 256))
        self.transform = transforms.Compose([transforms.RandomGrayscale(0.2),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])])
        points_flip = [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 28, 29, 30, 31, 36, 35, 34, 33, 32, 46, 45, 44, 43, 48, 47, 40, 39, 38, 37, 42, 41, 55, 54, 53, 52, 51, 50, 49, 60, 59, 58, 57, 56, 65, 64, 63, 62, 61, 68, 67, 66]
        self.points_flip = (np.array(points_flip)-1).tolist()                                      

    def __len__(self):
        return len(self.imgnames)
    
    def __getitem__(self, index):        

        img = Image.open(os.path.join(self.data_path, 
            self.imgnames[index].split("_")[0], self.imgnames[index]))
        pose_mat = scio.loadmat(os.path.join(self.data_path, 
            self.imgnames[index].split("_")[0], self.imgnames[index].replace("jpg","mat")))
        landmark_mat = scio.loadmat(os.path.join(self.data_path, 
            "landmarks/"+self.imgnames[index].split("_")[0], self.imgnames[index].replace(".jpg","_pts.mat")))

        pose_para = pose_mat['Pose_Para'][0]
        pitch = pose_para[0] * 180 / np.pi
        yaw = pose_para[1] * 180 / np.pi
        roll = pose_para[2] * 180 / np.pi     
        pose = np.array([yaw, pitch, roll])           
        visibility = np.ones((68))        

        # Crop the face loosely
        pt2d = landmark_mat["pts_3d"]     
        x_min = min(pt2d[:, 0])
        y_min = min(pt2d[:, 1])
        x_max = max(pt2d[:, 0])
        y_max = max(pt2d[:, 1])
        anno_x , anno_y = pt2d[:, 0], pt2d[:, 1]     
       
        ad = self.margin
        h = y_max-y_min
        w = x_max-x_min        
        x_min = max(int(x_min - ad * w), 0)
        x_max = min(int(x_max + ad * w), img.width - 1)
        y_min = max(int(y_min - (ad+0.25) * h), 0)
        y_max = min(int(y_max + ad * h), img.height - 1)

        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
        anno_x = (anno_x - x_min) / (x_max-x_min)
        anno_y = (anno_y - y_min) / (y_max-y_min)
        norm_landmarks = np.concatenate([anno_x.reshape(-1,1), anno_y.reshape(-1,1)], axis=1)                             

        # data augmentation                               
        for type in self.aug:                
            type_name = eval("random_" + type)
            if type == "flip":
                img, norm_landmarks, pose = type_name(img, norm_landmarks, pose, self.points_flip)
            else:
                img, norm_landmarks, pose = type_name(img, norm_landmarks, pose)                

        img = self.resize(img)
        orig_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        landmarks = norm_landmarks*img.size[0]
        img = self.transform(img)                
        
        # generate heatmap
        heatmap = np.zeros((landmarks.shape[0], 64, 64))
        for j in range(landmarks.shape[0]):
            if landmarks[j][0] > 0:
                heatmap[j] = draw_gaussian(heatmap[j], landmarks[j]/4.0+1, 1)                        

        return img, orig_img, heatmap, landmarks, visibility, pose                 
