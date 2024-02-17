# ---------------------------------------------------------------------------------------
# This file is a utils file of CIT, and some codes are borrowed from AWing loss and HRNet
# Copyright © 2022 Li Yao-kun <liyk58@mail2.sysu.edu.cn>
# To find more details, please refer to: https://github.com/Iron-LYK/CIT
# ---------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import os
import logging
import time
from collections import namedtuple
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps

import torch
import torch.optim as optim
import torch.nn as nn

class interpolation(nn.Module):
    def __init__(self, interpolation_points, half_length):
        super(interpolation, self).__init__()
        self.inter_points = interpolation_points    # total num of interpolation points
        self.half_length = torch.tensor([[[half_length, half_length]]], dtype=torch.float32)    
        self.half_length.requires_grad = False        
    
    def forward(self, feature_map, anchor):
        Bs = anchor.size(0)
        N = anchor.size(1)
        feature_dim = feature_map.size()
        half_length = (self.half_length.to(anchor.device) / (feature_dim[2])).repeat(Bs, 1, 1)
        bounding_min = torch.clamp(anchor - half_length, 0.0, 1.0)
        bounding_max = torch.clamp(anchor + half_length, 0.0, 1.0)
        bounding_box = torch.cat((bounding_min, bounding_max), dim=2)

        bounding_xs = torch.nn.functional.interpolate(bounding_box[:,:,0::2], size=self.inter_points,
                                                      mode='linear', align_corners=True)       
        bounding_ys = torch.nn.functional.interpolate(bounding_box[:,:,1::2], size=self.inter_points,
                                                      mode='linear', align_corners=True)       
        
        
        bounding_xs, bounding_ys = bounding_xs.unsqueeze(3).repeat_interleave(self.inter_points, dim=3), \
                                   bounding_ys.unsqueeze(2).repeat_interleave(self.inter_points, dim=2)
        meshgrid = torch.stack([bounding_xs, bounding_ys], dim=-1).view(Bs, N * self.inter_points * self.inter_points, 2)
        meshgrid.detach().requires_grad = False         
        potential_anchor = meshgrid.detach() * (feature_dim[2] - 1)
        potential_anchor = torch.clamp(potential_anchor, 0, feature_dim[2] - 1)
        anchor_pixel = self._get_interploate(potential_anchor, feature_map, feature_dim)

        return anchor_pixel

    def _flatten_tensor(self, input):
        return input.contiguous().view(input.nelement())

    def _get_index_point(self, input, anchor, feature_dim):
        # print(anchor.shape)

        index = anchor[:, :, 1] * feature_dim[2] + anchor[:, :, 0]
        # print(index.shape)
        output_list = []
        for i in range(feature_dim[0]):
            output_list.append(torch.index_select(input[i].contiguous().flatten(1), 1, index[i]))
        output = torch.stack(output_list)

        return output.permute(0, 2, 1).contiguous()

    def _get_interploate(self, potential_anchor, feature_maps, feature_dim):
        anchors_lt = potential_anchor.floor().long()    
        anchors_rb = potential_anchor.ceil().long()     

        anchors_lb = torch.stack([anchors_lt[:, :, 0], anchors_rb[:, :, 1]], 2)     
        anchors_rt = torch.stack([anchors_rb[:, :, 0], anchors_lt[:, :, 1]], 2)    

        vals_lt = self._get_index_point(feature_maps, anchors_lt.detach(), feature_dim)
        vals_rb = self._get_index_point(feature_maps, anchors_rb.detach(), feature_dim)
        vals_lb = self._get_index_point(feature_maps, anchors_lb.detach(), feature_dim)
        vals_rt = self._get_index_point(feature_maps, anchors_rt.detach(), feature_dim)

        coords_offset_lt = potential_anchor - anchors_lt.type(potential_anchor.data.type())

        vals_t = vals_lt + (vals_rt - vals_lt) * coords_offset_lt[:, :, 0:1]
        vals_b = vals_lb + (vals_rb - vals_lb) * coords_offset_lt[:, :, 0:1]
        mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_lt[:, :, 1:2]

        return mapped_vals

def cv_crop(image, landmarks, center, scale, resolution=256, center_shift=0):
    new_image = cv2.copyMakeBorder(image, center_shift,
                                   center_shift,
                                   center_shift,
                                   center_shift,
                                   cv2.BORDER_CONSTANT, value=[0,0,0])
    new_landmarks = landmarks.copy()
    if center_shift != 0:
        center[0] += center_shift
        center[1] += center_shift
        new_landmarks = new_landmarks + center_shift
    length = 200 * scale
    top = int(center[1] - length // 2)
    bottom = int(center[1] + length // 2)
    left = int(center[0] - length // 2)
    right = int(center[0] + length // 2)
    y_pad = abs(min(top, new_image.shape[0] - bottom, 0))
    x_pad = abs(min(left, new_image.shape[1] - right, 0))
    top, bottom, left, right = top + y_pad, bottom + y_pad, left + x_pad, right + x_pad
    new_image = cv2.copyMakeBorder(new_image, y_pad,
                                   y_pad,
                                   x_pad,
                                   x_pad,
                                   cv2.BORDER_CONSTANT, value=[0,0,0])
    new_image = new_image[top:bottom, left:right]
    new_image = cv2.resize(new_image, dsize=(int(resolution), int(resolution)),
                           interpolation=cv2.INTER_LINEAR)
    new_landmarks[:, 0] = (new_landmarks[:, 0] + x_pad - left) * resolution / length
    new_landmarks[:, 1] = (new_landmarks[:, 1] + y_pad - top) * resolution / length
    return new_image, new_landmarks

def show_landmarks(img, landmarks,v):
    """draw landmarks with the visibility"""

    x,y = landmarks[:,0],landmarks[:,1]
    for i in range(len(x)):  
        temp_x, temp_y, temp_v = int(x[i]), int(y[i]), int(v[i])
        if temp_v == 1:
            cv2.circle(img, (int(temp_x),int(temp_y)), 1, (0, 0, 255), 2)
        else:
            cv2.circle(img, (int(temp_x),int(temp_y)), 1, (80, 200, 120), 2)

    now = time.time()
    cv2.imshow("img_{}".format(now),img)
    cv2.waitKey(0)

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, heatmap, landmarks, boundary, weight_map= sample['image'], sample['heatmap'], sample['landmarks'], sample['boundary'], sample['weight_map']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
            # image_small = np.expand_dims(image_small, axis=2)
        image = image.transpose((2, 0, 1))
        boundary = np.expand_dims(boundary, axis=2)
        boundary = boundary.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).float().div(255.0),
                'heatmap': torch.from_numpy(heatmap).float(),
                'landmarks': torch.from_numpy(landmarks).float(),
                'boundary': torch.from_numpy(boundary).float().div(255.0),
                'weight_map': torch.from_numpy(weight_map).float()}

def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )

    # Get the RGB buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring (fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (w, h, 3)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll (buf, 3, axis=2)
    return buf

def get_coord_from_hm(hm):
    "with CPU"    
    B, C, H, W = hm.shape
    preds = np.zeros([B, C, 2])
    for i in range(B):
        for j in range(C):
            y_x = np.unravel_index(np.argmax(hm[i, j, :]),hm[i, j, :].shape)   
            preds[i,j,0] = y_x[1]
            preds[i,j,1] = y_x[0]
    return preds

def get_NME(pred_landmarks, gt_landmarks, fail_count, save_nmes, normfactor=None, gt_op=None, dataset=None):
    '''
       Calculate total NME for a batch of data
       @Param:
           pred_heatmaps: a batch of heatmaps tensor with size of ([batch, num_keypoints, heatmap_size[0], heatmap_size[1]])
           gt_landmarks : a batch of ground truth landmark coordinates with size of ([batch, num_keypoints, (pred_heatmaps.ndim-2)])
           fail_count   : number of keypoints where prediction fails
           save_nmes    : nme values for all samples
       @Returns:
           fail_count   : number of keypoints where prediction fails
           save_nmes    : nme values for all samples
    '''

    for i in range(pred_landmarks.shape[0]):
        pred_landmark = pred_landmarks[i] * 4.0
        gt_landmark = gt_landmarks[i]

        if "AFLW2000" in dataset:      
            minx, maxx = np.min(gt_landmark[:, 0]), np.max(gt_landmark[:, 0])
            miny, maxy = np.min(gt_landmark[:, 1]), np.max(gt_landmark[:, 1])
            norm_factor = np.sqrt((maxx - minx) * (maxy - miny))
            single_nme = np.mean(np.linalg.norm(pred_landmark - gt_landmark, axis=1)) / norm_factor 
            save_nmes.append(single_nme)                                       
            if single_nme > 0.1:
                fail_count += 1 
            continue             
        elif "MERL_RAV_FLOP" in dataset:
            # landmark_bbox
            minx, maxx = np.min(gt_landmark[:, 0]), np.max(gt_landmark[:, 0])
            miny, maxy = np.min(gt_landmark[:, 1]), np.max(gt_landmark[:, 1])
            norm_factor = np.sqrt((maxx - minx) * (maxy - miny))
            
            
            op = gt_op[i]
            v = abs(op)
            if np.sum(v) == 0:
                continue
            single_nme = np.sum((np.linalg.norm(pred_landmark - gt_landmark, axis=1))*v) / (norm_factor*np.sum(v))          
            save_nmes.append(single_nme)                                 
            if single_nme > 0.1:
                fail_count += 1 
            continue                                    
        elif "COFW" in dataset:
            norm_factor = np.linalg.norm(gt_landmark[16]- gt_landmark[17])   
        elif "300W_LP" or "BIWI" in dataset:  
            norm_factor = np.linalg.norm(gt_landmark[36]- gt_landmark[45])                 
        single_nme = np.mean(np.linalg.norm(pred_landmark - gt_landmark, axis=1)) / norm_factor
        save_nmes.append(single_nme)
        if single_nme > 0.1:
            fail_count += 1

    return fail_count, save_nmes 

def get_fr_and_auc(nmes, thres=0.1, step=0.0001):
    """
        Calculate total failure rate and accuracy
        @Param:
            nmes: a list to store each single nme
        @Return:
            fr: falure rate
            auc: accuracy
    """
    num_data = len(nmes)
    xs = np.arange(0, thres + step, step)
    ys = np.array([np.count_nonzero(nmes <= x) for x in xs]) / float(num_data)
    fr = 1.0 - ys[-1]
    auc = simps(ys, x=xs) / thres
    return fr, auc

def plot_PRcurve(precision, recall):
    plt.figure()
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('PR curve')
    plt.plot(recall, precision)
    plt.show()

def calc_recall(precision, recall):
    '''calculate recall when precision is 80%'''
    idx = np.where(precision>0.8)
    if len(idx)==0:
        target_recall = 0
    else:
        target_recall = recall[idx]
        target_recall = np.max(target_recall) *100
    return target_recall

def create_logger(cfg, cfg_name):
    root_output_dir = Path(cfg.OUTPUT_DIR)  
    
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.TRAIN_DATASET + '_' + cfg.DATASET.TEST_DATASET
    dataset = dataset.replace(':', '_')
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_dir = '{}_{}'.format(cfg_name, time_str)
    final_output_dir = root_output_dir / dataset / model / cfg_name / log_dir
    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)   

    log_file = '{}.log'.format(log_dir)    
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    
    # Create and configure a root logger    
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()    
    
    # Print logs with level greater than or equal to INFO
    logger.setLevel(logging.INFO)
    
    # add a Handler to the root logger
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
        (cfg_name + '_' + time_str)

    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir), final_log_file

def get_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            filter(lambda x: x.requires_grad is not False ,model.parameters()),
            lr=cfg.TRAIN.LR
        )    
    return optimizer

def get_shared_layer_params(model):    
    for module_name, module in model.shared_feature.named_modules():
        if 'bn' in module_name:
            module.eval()                
        for name, param in module.named_parameters():
            # print(param)
            yield param     

def get_task_layer_params(model, type=None):
    if type is not None:
        for module_name, module in eval("model.transformer_{}.named_modules()".format(type)):
            if 'bn' in module_name:
                module.eval()                
            for name, param in module.named_parameters():
                # print(param)
                yield param    
    else:
        names = [item[0] for item in model._modules.items()]
        for name in names:
            if name != "shared_feature":            
                for module_name, module in eval("model.{}.named_modules()".format(name)):
                    if 'bn' in module_name:
                        module.eval()                
                    for name, param in module.named_parameters():
                        # print(param)
                        yield param

def freeze_bn(m):
    if isinstance(m, nn.BatchNorm2d):
        m.eval()

class EarlyStopping:
    """ This EarlyStopping class is inspired from https://github.com/Bjarten/early-stopping-pytorch 
        Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, task_name, output_dir, patience=15, verbose=False, trace_func = print):
        """
        Args:
            task_name (str): Model name, used to decide which metric to focus on
            output_dir (str): Path for the checkpoint to be saved to.
            patience (int): How long to wait after last time validation metrics improved.
                            Default: 15
            verbose (bool): If True, prints a message for each validation metrics improvement. 
                            Default: False                                    
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.task = task_name
        self.counter = 0
        self.best_metric = None
        self.best_model = False
        self.early_stop = False
        self.path = os.path.join(output_dir, 'best_model.pth')
        self.trace_func = trace_func
        self.improved_metric = 0
    
    def __call__(self, val_metric, states):

        if self.best_metric is None:
            self.best_metric = val_metric
        
        if "hp" in self.task and "op" not in self.task:
            if val_metric[2] < self.best_metric[2]:
                self.best_model = True
                self.improved_metric = 2
            else:
                self.best_model = False 
        if "op" in self.task and "lm" not in self.task:
            if val_metric[1] > self.best_metric[1]:
                self.best_model = True
                self.improved_metric = 1
            else:
                self.best_model = False         
        else:        
            if val_metric[0] < self.best_metric[0]:
                self.best_model = True 
                self.improved_metric = 0      
            # elif val_metric[1] > self.best_metric[1]:
            #     self.best_model = True
            #     self.improved_metric = 1
            else:
                self.best_model = False            
        
        if not self.best_model:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:            
            self.save_checkpoint(val_metric, states)
            self.best_metric = val_metric
            self.counter = 0

    def save_checkpoint(self, val_metric, states):
        '''Saves model when validation metric improved.'''
        idx = self.improved_metric
        if self.verbose:            
            self.trace_func(f'Validation metric improved ({self.best_metric[idx]:.6f} --> {val_metric[idx]:.6f}).  Saving model ...')
        torch.save(states['state_dict'], self.path)

def get_model_summary(model, *input_tensors, item_length=26, verbose=False):
    """
    :param model:
    :param input_tensors:
    :param item_length:
    :return:
    """

    summary = []

    ModuleDetails = namedtuple(
        "Layer", ["name", "input_size", "output_size", "num_parameters", "multiply_adds"])
    hooks = []
    layer_instances = {}

    def add_hooks(module):

        def hook(module, input, output):
            class_name = str(module.__class__.__name__)

            instance_index = 1
            if class_name not in layer_instances:
                layer_instances[class_name] = instance_index
            else:
                instance_index = layer_instances[class_name] + 1
                layer_instances[class_name] = instance_index

            layer_name = class_name + "_" + str(instance_index)

            params = 0

            if class_name.find("Conv") != -1 or class_name.find("BatchNorm") != -1 or \
               class_name.find("Linear") != -1:
                for param_ in module.parameters():
                    params += param_.view(-1).size(0)

            flops = "Not Available"
            if class_name.find("Conv") != -1 and hasattr(module, "weight"):
                flops = (
                    torch.prod(
                        torch.LongTensor(list(module.weight.data.size()))) *
                    torch.prod(
                        torch.LongTensor(list(output.size())[2:]))).item()
            elif isinstance(module, nn.Linear):
                flops = (torch.prod(torch.LongTensor(list(output.size()))) \
                         * input[0].size(1)).item()

            if isinstance(input[0], list):
                input = input[0]
            if isinstance(output, tuple):
                output = output[0]
                if isinstance(output, tuple):
                    output = output[0]

            summary.append(
                ModuleDetails(
                    name=layer_name,
                    input_size=list(input[0].size()),
                    output_size=list(output.size()),
                    num_parameters=params,
                    multiply_adds=flops)
            )

        if not isinstance(module, nn.ModuleList) \
           and not isinstance(module, nn.Sequential) \
           and module != model:
            hooks.append(module.register_forward_hook(hook))

    model.eval()
    model.apply(add_hooks)

    space_len = item_length

    model(*input_tensors)
    for hook in hooks:
        hook.remove()

    details = ''
    if verbose:
        details = "Model Summary" + \
            os.linesep + \
            "Name{}Input Size{}Output Size{}Parameters{}Multiply Adds (Flops){}".format(
                ' ' * (space_len - len("Name")),
                ' ' * (space_len - len("Input Size")),
                ' ' * (space_len - len("Output Size")),
                ' ' * (space_len - len("Parameters")),
                ' ' * (space_len - len("Multiply Adds (Flops)"))) \
                + os.linesep + '-' * space_len * 5 + os.linesep

    params_sum = 0
    flops_sum = 0
    for layer in summary:
        params_sum += layer.num_parameters
        if layer.multiply_adds != "Not Available":
            flops_sum += layer.multiply_adds
        if verbose:
            details += "{}{}{}{}{}{}{}{}{}{}".format(
                layer.name,
                ' ' * (space_len - len(layer.name)),
                layer.input_size,
                ' ' * (space_len - len(str(layer.input_size))),
                layer.output_size,
                ' ' * (space_len - len(str(layer.output_size))),
                layer.num_parameters,
                ' ' * (space_len - len(str(layer.num_parameters))),
                layer.multiply_adds,
                ' ' * (space_len - len(str(layer.multiply_adds)))) \
                + os.linesep + '-' * space_len * 5 + os.linesep

    details += os.linesep \
        + "Total Parameters: {:,}".format(params_sum) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Total Multiply Adds (For Convolution and Linear Layers only): {:,} GFLOPs".format(flops_sum/(1024**3)) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Number of Layers" + os.linesep
    for layer in layer_instances:
        details += "{} : {} layers   ".format(layer, layer_instances[layer])

    return details


# batch*n
def normalize_vector( v, use_gpu=True):
    batch=v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))# batch
    if use_gpu:
        v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
    else:
        v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8])))  
    v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
    v = v/v_mag
    return v
    
# u, v batch*n
def cross_product( u, v):
    batch = u.shape[0]
    #print (u.shape)
    #print (v.shape)
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
        
    out = torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1)#batch*3
        
    return out
        
    
#poses batch*6
#poses
def compute_rotation_matrix_from_ortho6d(poses, use_gpu=True):
    x_raw = poses[:,0:3]#batch*3
    y_raw = poses[:,3:6]#batch*3
        
    x = normalize_vector(x_raw, use_gpu) #batch*3
    z = cross_product(x,y_raw) #batch*3
    z = normalize_vector(z, use_gpu)#batch*3
    y = cross_product(z,x)#batch*3
        
    x = x.view(-1,3,1)
    y = y.view(-1,3,1)
    z = z.view(-1,3,1)
    matrix = torch.cat((x,y,z), 2) #batch*3*3
    return matrix


#input batch*4*4 or batch*3*3
#output torch batch*3 x, y, z in radiant
#the rotation is in the sequence of x,y,z
def compute_euler_angles_from_rotation_matrices(rotation_matrices, use_gpu=True):
    batch=rotation_matrices.shape[0]
    R=rotation_matrices
    sy = torch.sqrt(R[:,0,0]*R[:,0,0]+R[:,1,0]*R[:,1,0])
    singular= sy<1e-6
    singular=singular.float()
        
    x=torch.atan2(R[:,2,1], R[:,2,2])
    y=torch.atan2(-R[:,2,0], sy)
    z=torch.atan2(R[:,1,0],R[:,0,0])
    
    xs=torch.atan2(-R[:,1,2], R[:,1,1])
    ys=torch.atan2(-R[:,2,0], sy)
    zs=R[:,1,0]*0
        
    if use_gpu:
        out_euler=torch.autograd.Variable(torch.zeros(batch,3).cuda())
    else:
        out_euler=torch.autograd.Variable(torch.zeros(batch,3))  
    out_euler[:,0]=x*(1-singular)+xs*singular
    out_euler[:,1]=y*(1-singular)+ys*singular
    out_euler[:,2]=z*(1-singular)+zs*singular
        
    return out_euler