# ------------------------------------------------------------------------------
# Modified by Yanjie Li (leeyegy@gmail.com)
# ------------------------------------------------------------------------------

import torch
import torch.nn.functional as F
from torch import nn
from .GCN_base import GCN_layer
from timm.models.layers.weight_init import trunc_normal_
from utils import interpolation, compute_rotation_matrix_from_ortho6d, compute_euler_angles_from_rotation_matrices
from einops.layers.torch import Rearrange
import copy
from typing import Optional


class Transformer_LANDMARK(nn.Module):
    def __init__(self, *, cfg, num_keypoints, dim, depth, heads,apply_init=False,heatmap_size=[64,64]):
        super().__init__()

        self.cfg = cfg
        self.token_dim = dim                                      
        self.dim_feedforward = cfg.MODEL.DIM_FEEDFORWARD                   
        self.num_keypoints = num_keypoints                   

        self.feature_dim = cfg.MODEL.EXTRA.STAGE4.NUM_CHANNELS[0]
        self.feature_dim = 270
        self.conv = nn.Conv2d(
                in_channels=self.feature_dim,
                out_channels=cfg.MODEL.DIM,
                kernel_size=cfg.MODEL.EXTRA.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if cfg.MODEL.EXTRA.FINAL_CONV_KERNEL == 3 else 0)        

        if "op" in cfg.MODEL.NAME and "hp" not in cfg.MODEL.NAME:
            self.pos_embedding = nn.Parameter(torch.randn(heatmap_size[0]*heatmap_size[1]+1, 1, self.token_dim))
        elif "hp" in cfg.MODEL.NAME and "op" not in cfg.MODEL.NAME:
            self.pos_embedding = nn.Parameter(torch.randn(heatmap_size[0]*heatmap_size[1]+1, 1, self.token_dim))
        elif "hp" in cfg.MODEL.NAME and "op" in cfg.MODEL.NAME:
            self.pos_embedding = nn.Parameter(torch.randn(heatmap_size[0]*heatmap_size[1]+2, 1, self.token_dim))            
        else:
            self.pos_embedding = nn.Parameter(torch.randn(heatmap_size[0]*heatmap_size[1], 1, self.token_dim))

        self.pose_to_embedding = nn.Linear(3, self.token_dim) 
        self.op_to_embedding = nn.Linear(self.num_keypoints,self.token_dim)

        # transformer
        encoder_layer = TransformerEncoderLayer(
            d_model=self.token_dim, nhead=heads, dim_feedforward=self.dim_feedforward,
            activation='relu')        
        self.transformer = TransformerEncoder(encoder_layer, depth)        

        self.final_layer = nn.Conv2d(
            in_channels=self.token_dim,
            out_channels=cfg['MODEL']['NUM_LANDMARKS'],
            kernel_size=cfg['MODEL']['EXTRA']['FINAL_CONV_KERNEL'],
            stride=1,
            padding=1 if cfg['MODEL']['EXTRA']['FINAL_CONV_KERNEL'] == 3 else 0
        )

        if apply_init:
            self.apply(self._init_weights)            

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, feature, mask_map, occlusion_probability, pose):   
        
        x = self.conv(feature[:,:self.feature_dim,:,:])

        B, C, H, W = x.shape
        x = x.flatten(2).permute(2, 0, 1)        

        if "op" in self.cfg.MODEL.NAME and "hp" in self.cfg.MODEL.NAME:
            op_token = self.op_to_embedding(occlusion_probability.unsqueeze(1)).permute(1,0,2)
            pose_token = self.pose_to_embedding(pose.unsqueeze(1)).permute(1,0,2)
            x = torch.cat((x,op_token,pose_token), dim=0)
        elif "op" in self.cfg.MODEL.NAME:            
            op_token = self.op_to_embedding(occlusion_probability.unsqueeze(1)).permute(1,0,2)                     
            x = torch.cat((x,op_token), dim=0)
        elif "hp" in self.cfg.MODEL.NAME:
            pose_token = self.pose_to_embedding(pose.unsqueeze(1)).permute(1,0,2)
            x = torch.cat((x,pose_token), dim=0)        
        
        x, _ = self.transformer(x, pos=self.pos_embedding)  

        x = x[:H*W]
        x = x.permute(1, 2, 0).contiguous().view(B, C, H, W)
        x = self.final_layer(x) 
        
        if mask_map is not None:
            x = x*mask_map                       
        return x            

class Transformer_OCCLUSION(nn.Module):
    def __init__(self, *, cfg, num_keypoints, dim, depth, heads, apply_init=False):
        super().__init__()

        self.cfg = cfg
        self.token_dim = dim                                       
        self.interpolation_points = cfg.MODEL.NUM_INTER_POINTS
        self.local_patch_size = cfg.MODEL.LOCAL_PATCH_SIZE
        self.dim_feedforward = cfg.MODEL.DIM_FEEDFORWARD                                             
        self.num_keypoints = num_keypoints                   

        self.pos_embedding = nn.Parameter(torch.randn(num_keypoints+1, 1, self.token_dim))

        self.pose_to_embedding = nn.Linear(3, self.token_dim) 

        # interpolation
        self.get_patch_feature = interpolation(self.interpolation_points, int(self.local_patch_size/2))

        # transformer
        encoder_layer = TransformerEncoderLayer(
            d_model=self.token_dim, nhead=heads, dim_feedforward=self.dim_feedforward,
            activation='relu')        
        self.transformer = TransformerEncoder(encoder_layer, depth)        

        feature_dim = 270
        self.conv1 =nn.Sequential(
            nn.Conv2d(
                in_channels=feature_dim,
                out_channels=feature_dim,
                kernel_size=1,
                stride=1,
                padding=1 if cfg.MODEL.EXTRA.FINAL_CONV_KERNEL == 3 else 0),
            nn.BatchNorm2d(feature_dim, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=feature_dim,
                out_channels=cfg.MODEL.DIM,
                kernel_size=cfg.MODEL.EXTRA.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if cfg.MODEL.EXTRA.FINAL_CONV_KERNEL == 3 else 0)
        )
        self.feature_extractor = nn.Conv2d(self.token_dim, self.token_dim, self.interpolation_points, bias=False)
        self.predict_head = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, 1),
            Rearrange('b c e -> b (c e)'),
            nn.Sigmoid()
        )
        if apply_init:
            self.apply(self._init_weights)              

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, feature, norm_lm, op, pose):

        feature = self.conv1(feature) 

        B = feature.shape[0] 
        N = norm_lm.shape[1]  

        local_patch = self.get_patch_feature(feature, norm_lm).view(B, N, 
                                self.interpolation_points,self.interpolation_points, self.token_dim)    # (bs, dim, h, w)
        local_patch = local_patch.view(B*N, self.interpolation_points, 
                                self.interpolation_points, self.token_dim).permute(0, 3, 2, 1)
        valp = self.feature_extractor(local_patch).view(B, N, self.token_dim)                           # (bs, n, dim) 
        x = valp.permute(1, 0, 2) 
        
        pose_embedding = self.pose_to_embedding(pose.unsqueeze(1)).permute(1, 0, 2)                     # (1, bs, dim)     
        
        x = torch.cat((x,pose_embedding), dim=0)        
        
        x, _ = self.transformer(x, pos=self.pos_embedding)
        x = x[:self.num_keypoints].permute(1, 0, 2)                                                     # (bs, n, dim)        
        
        x = torch.mul(x, (1-op.unsqueeze(2)))
        op = self.predict_head(x)

        valp = valp*op.reshape(B,N,1)

        return op, valp

class Transformer_POSE(nn.Module):
    def __init__(self, *, cfg, num_keypoints, dim, depth, heads, apply_init=False, heatmap_size=[64,64]):
        super().__init__()
        self.cfg = cfg
        self.token_dim = dim                             
        self.num_keypoints = num_keypoints 
        self.dim_feedforward = cfg.MODEL.DIM_FEEDFORWARD      
        self.interpolation_points = cfg.MODEL.NUM_INTER_POINTS
        self.local_patch_size = cfg.MODEL.LOCAL_PATCH_SIZE

        # interpolation
        self.get_patch_feature = interpolation(self.interpolation_points, int(self.local_patch_size/2))
        self.feature_extractor = nn.Conv2d(self.token_dim, self.token_dim, self.interpolation_points, bias=False)

        # make embedding
        self.op_to_embedding = nn.Linear(self.num_keypoints, self.token_dim)
        self.map = nn.Linear(3, self.token_dim)
        self.lm_to_embedding = nn.Linear(2, self.token_dim)
        self.topo_pos_embedding = nn.Parameter(torch.randn(self.num_keypoints, 1, self.token_dim))
        self.local_pos_embedding = nn.Parameter(torch.randn(self.num_keypoints, 1, self.token_dim))
        self.global_pos_embedding = nn.Parameter(torch.randn(self.num_keypoints, 1, self.token_dim))
        
        # batch normalization
        self.bn_1 = nn.BatchNorm1d(self.num_keypoints)
        self.bn_2 = nn.BatchNorm1d(self.num_keypoints)
        self.bn_3 = nn.BatchNorm1d(self.num_keypoints)
        self.bn_4 = nn.BatchNorm1d(self.num_keypoints)
        self.bn_5 = nn.BatchNorm1d(self.num_keypoints)
        self.bn_6 = nn.BatchNorm1d(self.num_keypoints)        

        encoder_layer_topological = TransformerEncoderLayer(
            d_model=self.token_dim, nhead=heads, dim_feedforward=self.dim_feedforward,
            activation='relu')        
        self.transformer_topological = TransformerEncoder(encoder_layer_topological, depth)  

        encoder_layer_local = TransformerEncoderLayer(
            d_model=self.token_dim, nhead=heads, dim_feedforward=self.dim_feedforward,
            activation='relu')
        self.transformer_local = TransformerEncoder(encoder_layer_local, depth)

        encoder_layer_global = TransformerEncoderLayer(
            d_model=self.token_dim, nhead=heads, dim_feedforward=self.dim_feedforward,
            activation='relu')
        self.transformer_global = TransformerEncoder(encoder_layer_global, depth) 
 
        feature_dim = 270
        self.conv1 =nn.Sequential(
            nn.Conv2d(
                in_channels=feature_dim,
                out_channels=feature_dim*2,
                kernel_size=3,
                stride=2,
                padding=1),
            nn.BatchNorm2d(feature_dim*2, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=feature_dim*2,
                out_channels= feature_dim*4,
                kernel_size=3,
                stride=2,
                padding=1),
            nn.BatchNorm2d(feature_dim*4, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=feature_dim*4,
                out_channels= feature_dim*8,
                kernel_size=3,
                stride=2,
                padding=1),
            nn.BatchNorm2d(feature_dim*8, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=1),
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(feature_dim*8, self.token_dim)
        )
        self.avgpool = nn.AvgPool1d(heatmap_size[0]*heatmap_size[1]+1)

        self.conv2 = nn.Sequential( nn.Conv1d(2*self.token_dim, self.token_dim, 1),
                                    nn.BatchNorm1d(self.token_dim, momentum=BN_MOMENTUM),
                                    nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential( nn.Conv1d(2*self.token_dim, self.token_dim, 1),
                                    nn.BatchNorm1d(self.token_dim, momentum=BN_MOMENTUM),
                                    nn.ReLU(inplace=True))               
        self.conv4 = nn.Conv1d(self.token_dim, 64, 1) 
        self.conv5 = nn.Conv1d(2*self.token_dim, 64, 1)                                         
        self.pool = nn.AvgPool1d(self.num_keypoints)                                  
        self.local_conv =nn.Sequential(
            nn.Conv2d(
                in_channels=feature_dim,
                out_channels=feature_dim,
                kernel_size=1,
                stride=1,
                padding=1 if cfg.MODEL.EXTRA.FINAL_CONV_KERNEL == 3 else 0),
            nn.BatchNorm2d(feature_dim, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=feature_dim,
                out_channels=cfg.MODEL.DIM,
                kernel_size=cfg.MODEL.EXTRA.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if cfg.MODEL.EXTRA.FINAL_CONV_KERNEL == 3 else 0)   
        )

        self.mlp = nn.Sequential(  
            nn.LayerNorm(64),
            nn.Linear(64, 3)
        )
                   
        if apply_init:
            self.apply(self._init_weights)            
    
    def _init_weights(self, m):
        # print("Initialization...")
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, feature, norm_lm, valp, pose, iteration=None):            
     
        B = feature.shape[0] 
        N = norm_lm.shape[1]          
        shared_feature = self.conv1(feature)                                                                        
        shared_feature = shared_feature.unsqueeze(1).repeat(1,N,1)                                                      

        if valp is None :
            feature_embedding = self.local_conv(feature)
            local_patch = self.get_patch_feature(feature_embedding, norm_lm).view(B, N, 
                                    self.interpolation_points,self.interpolation_points, self.token_dim)            
            local_patch = local_patch.view(B*N, self.interpolation_points, 
                                    self.interpolation_points, self.token_dim).permute(0, 3, 2, 1)      
            valp = self.feature_extractor(local_patch).view(B, N, self.token_dim)                                          

        lm_embedding = self.lm_to_embedding(norm_lm).permute(1, 0, 2)                                               
        topological_feature, _ = self.transformer_topological(lm_embedding, pos=self.topo_pos_embedding)
        
        local_feature = torch.cat((self.bn_1(topological_feature.permute(1, 0, 2)), self.bn_2(valp)), dim=2)        
        local_feature = self.conv2(local_feature.permute(0,2,1)).permute(0,2,1)
        local_feature, _ = self.transformer_local(local_feature.permute(1, 0, 2), pos=self.local_pos_embedding)       
   
        global_feature = torch.cat((self.bn_3(shared_feature), self.bn_4(local_feature.permute(1, 0, 2))), dim=2)   
        global_feature = self.conv3(global_feature.permute(0,2,1)).permute(0,2,1)
        global_feature, _ = self.transformer_global(global_feature.permute(1, 0, 2), pos=self.global_pos_embedding)        
        
        if iteration > 0:
            pose_prior = self.map(pose.unsqueeze(1)).repeat(1,N,1)     
            global_feature = torch.cat((self.bn_5(global_feature.permute(1,0,2)), self.bn_6(pose_prior)), dim=2)       
            x = self.conv5(global_feature.permute(0,2,1))
        else:
            x = self.conv4(global_feature.permute(1,2,0))                                              

        x = self.pool(x).squeeze(2)                                                                     

        x = self.mlp(x)                                                                             

        return x   

BN_MOMENTUM = 0.1

class TransformerEncoderLayer(nn.Module):
    """ Code borrowed from https://github.com/facebookresearch/detr/blob/master/models/transformer.py"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[torch.Tensor] = None,
                     src_key_padding_mask: Optional[torch.Tensor] = None,
                     pos: Optional[torch.Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[torch.Tensor] = None,
                    src_key_padding_mask: Optional[torch.Tensor] = None,
                    pos: Optional[torch.Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        if self.return_atten_map:
            src2, att_map = self.self_attn(q, k, value=src,
                                           attn_mask=src_mask,
                                           key_padding_mask=src_key_padding_mask)
        else:
            src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        if self.return_atten_map:
            return src, att_map
        else:
            return src

    def forward(self, src,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                pos: Optional[torch.Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class TransformerEncoder(nn.Module):
    "Modified from https://github.com/facebookresearch/detr/blob/master/models/transformer.py"
    def __init__(self, encoder_layer, num_layers, norm=None, pe_only_at_begin=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src,
                mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                pos: Optional[torch.Tensor] = None):
        output = src

        num = 0
        middle_output = []
        for layer in self.layers:
            output = layer(output, src_mask=mask,  pos=pos,
                            src_key_padding_mask=src_key_padding_mask)
            num+=1
            if num==1:
                middle_output = output
        if self.norm is not None:
            output = self.norm(output)

        return output, middle_output

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
