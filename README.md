# Cascaded Iterative Transformer for Jointly Predicting Facial Landmark, Occlusion Probability and Head Pose
## Official PyTorch implementation of the IJCV'23 paper
[Paper](https://link.springer.com/article/10.1007/s11263-023-01935-2) | [Code](https://github.com/Iron-LYK/CIT)
<div align=center><img src="https://github.com/Iron-LYK/CIT/blob/main/assets/overview.jpg" width="800" height="450" /></div>

## Abstract
Landmark detection under large pose with occlusion has been one of the challenging problems in the field of facial analysis. Recently, many works have predicted pose or occlusion together in the multi-task learning (MTL) paradigm, trying to tap into their dependencies and thus alleviate this issue. However, such implicit dependencies are weakly interpretable and inconsistent with the way humans exploit inter-task coupling relations, i.e., accommodating the induced explicit effects. This is one of the essentials that hinders their performance. To this end, in this paper, we propose a Cascaded Iterative Transformer (CIT) to jointly predict facial landmark, occlusion probability, and pose. The proposed CIT, besides implicitly mining task dependencies in a shared encoder, innovatively employs a cost-effective and portability-friendly strategy to pass the decoders’
predictions as prior knowledge to human-like exploit the coupling-induced effects. Moreover, to the best of our knowledge, no dataset contains all these task annotations simultaneously, so we introduce a new dataset termed MERL-RAV-FLOP based on the MERL-RAV dataset. We conduct extensive experiments on several challenging datasets (300W-LP, AFLW2000-3D, BIWI, COFW, and MERL-RAV-FLOP) and achieve remarkable results.

## Citation
If you find this work useful, please consider citing:
```bibtex
@article{li2023cascaded,
  title={Cascaded Iterative Transformer for Jointly Predicting Facial Landmark, Occlusion Probability and Head Pose},
  author={Li, Yaokun and Tan, Guang and Gou, Chao},
  journal={International Journal of Computer Vision},
  pages={1--16},
  year={2023},
  publisher={Springer}
```
</pre>
In addition, please also consider citing the great [MERL-RAV](https://github.com/abhi1kumar/MERL-RAV_dataset), [AFLW](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/) dataset, and [face_alignment](https://github.com/1adrianb/face-alignment), who have been very supportive of our work. 

<!-- 
 -->
 
# Dataset
- [Download](##Download)
- [Instruction](##Instruction)
- [Notation](##Notation)
- [Contact](##Contact)


## Download

We have already prepared our dataset. If you need it, please send an email to <liyk58@mail2.sysu.edu.cn> to obtain the download link.

## introduction

We propose the MERL-RAV-FLOP dataset based on the MERL-RAV dataset. From the perspective of efficiency, we adopt a simple and efficient semi-automated annotation process, i.e., automatic annotation followed by manual annotation, which is as follows:

First, we follow the [MERL-RAV instruction](https://github.com/abhi1kumar/MERL-RAV_dataset) to download the [AFLW dataset](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/) and prepare the original MERL-RAV Dataset. The directory structure of the prepared MERL-RAV dataset is shown below:
<pre>
|--MERL_RAV_Dataset 
        |--merl_rav_organized
        |        |-- frontal
        |        |      |--testset
        |        |      |      |--image00019.jpg
        |        |      |      |--image00019.pts
        |        |      |      |--...
        |        |      |--trainset
        |        |
        |        |-- left 
        |        |      |--testset
        |        |      |--trainset
        |        |
        |        |-- lefthalf
        |        |      |--testset
        |        |      |--trainset
        |        |
        |        |-- right
        |        |      |--testset
        |        |      |--trainset
        |        |
        |        |-- righthalf
        |               |--testset
        |               |--trainset
        |
        |--merl_rav_labels
        |--aflw
        |--common_functions.py
        |--organize_merl_rav_using_aflw_and_our_labels.py
</pre>

Next, we use [face_alignment](https://github.com/1adrianb/face-alignment) and the AFLW dataset to acquire pose and landmarks automatically. The concept behind this is to get the coarse-grained annotations automatically, and then get the fine-grained annotations manually, which saves our time in such a semi-automated way. 

Specifically, we match the file_id of the MERL-RAV dataset with the faces in the AFLW dataset to select the correct faces with pose, and then use face_alignment to predict the same 68 keypoints as MERL-RAV for the selected samples, which is done to coarse-grain fill in the missing landmarks in MERL-RAV dataset. This process is detailed in `make_files.py` (see the downloaded file) and is achieved by executing the following command. Note that before executing this command, you need to put the three pre-processed files `Faces`, `FacePose`, and `FaceRect` about the AFLW dataset into the aflw folder (again, these files are in the downloaded file).

<pre>
python make_files.py
</pre>

After above, the `merl_rav_organized` folder will be in the following form. The `train_files.txt` and `test_files.txt` are the list of training and testing files for the MERL-RAV-FLOP dataset, and the extra `xxx.npy` files for each sample in each folder are the corresponding annotation files for the samples (`xxx.pts` files are no longer used).

<pre>
|--merl_rav_organized
         |-- frontal
         |      |--testset
         |      |      |--image00019.jpg
         |      |      |--image00019.npy
         |      |      |--image00019.pts
         |      |      |--...
         |      |--trainset
         |
         |-- left 
         |-- lefthalf
         |-- right
         |-- righthalf
         |-- train_files.txt
         |-- test_files.txt
</pre>

Finally, after performing the above automated pre-processing, we cleaned the data manually, by checking and correcting the landmark and visibility annotations that are incorrect in the above process. After our manual modifications, the final `train_files.txt`, `test_files.txt`, and `.npy` files are shown in the download link above. The directories are shown below.

<pre>
|--MERL_RAV_FLOP
|        |-- frontal
|        |      |--testset
|        |      |      |--image00019.jpg
|        |      |      |--image00019.npy
|        |      |      |--...
|        |      |--trainset
|        |
|        |-- left 
|        |-- lefthalf
|        |-- right
|        |-- righthalf
|        |-- train_files.txt
|        |-- test_files.txt
|--make_files.py        
|--FacePose 
|--FaceRect 
|--Faces
</pre>

Each `.npy` file is an array with size of (209,). Taking `image00019.npy` as an illustration, the meaning represented by each dimension of this array is shown below:
<pre>
import numpy as np
annotation = np.load("image00019.npy")
x_coordinates = annotation[:68]
y_coordinates = annotation[68:136]
bbox_w, bbox_h = annotation[-2], annotation[-1]
pose = annotation[-5:-2]
visibility = annotation[-73:-5]
</pre>

## Contact
Feel free to contact <liyk58@mail2.sysu.edu.cn> if you have any doubts or questions.
