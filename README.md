## PatchAugNet
* For details, please refer to [Project Main Page](https://whu-usi3dv.github.io/PatchAugNet/)

### Benchmark Datasets
* Oxford dataset (for baseline config training/testing: PointNetVLAD, PPT-Net, Minkloc3DV2)
* NUS (in-house) Datasets (for testing: PointNetVLAD, PPT-Net, Minkloc3DV2)
  * university sector (U.S.)
  * residential area (R.A.)
  * business district (B.D.)
* Self-Collected Datasets (we will publish it in the future, welcome to follow our work！)
  * wuhan hankou (for training/testing: PointNetVLAD, PPT-Net, Minkloc3DV2, PatchAugNet)
  * whu campus (for testing: PointNetVLAD, PPT-Net, Minkloc3DV2, PatchAugNet)

### Project Code
#### Pre-requisites
Docker image: 
```
docker pull zouxh22135/pc_loc:v1
```

#### Dataset set-up
* Download the zip file of the Oxford RobotCar and 3-Inhouse benchmark datasets found [here](https://drive.google.com/open?id=1H9Ep76l8KkUpwILY-13owsEMbVCYTmyx) and extract the folder.
* Generate pickle files: We store the positive and negative point clouds to each anchor on pickle files that are used in our training and evaluation codes. The files only need to be generated once. The generation of these files may take a few minutes.
* Note: please check dataset info in 'datasets/dataset_info.py'
* Datasets defined in 'datasets/dataset_info.py', you can switch datasets by '--dataset' argument:
  * oxford
  * university, residential, business
  * hankou, campus
```
# For Oxford RobotCar / 3-Inhouse Datasets
python datasets/place_recognition_dataset.py

# For Self-Collected Dataset (we will publish it in the future, welcome to follow our work！)
python datasets/scene_dataset.py
```

#### Place Recognition: Training and Evaluation
* Build the third parties
```
cd libs/pointops && python setup.py install && cd ../../
cd libs/chamfer_dist && python setup.py install && cd ../../
cd libs/emd_module && python setup.py install && cd ../../
cd libs/KNN_CUDA && python setup.py install && cd ../../
```

* Train / Eavaluate PointNetVLAD / PPT-Net / Minkloc3DV2 / PatchAugNet
```
# Train PointNetVLAD / PPT-Net / Minkloc3D V2 / PatchAugNet on Oxford
python place_recognition/train_place_recognition.py --config configs/[pointnet_vlad / pptnet_origin / patch_aug_net].yaml --train_dataset oxford --test_dataset oxford

# Evaluate PointNetVLAD / PPT-Net / Minkloc3D V2 / PatchAugNet on Oxford, and save top k
python place_recognition/evaluate.py --model_type [model type] --weight [weight pth file] --dataset oxford --exp_dir [exp_dir]

Note: model types include [pointnet_vlad / pptnet / pptnet_l2_norm / minkloc3d_v2 / patch_aug_net]
      datasets include [oxford / university / residential / business / hankou / campus]
```

* Train Minkloc3D V2, see [Minkloc3DV2](https://github.com/jac99/MinkLoc3Dv2)

* Model pretrained on Self-Collected Dataset: https://drive.google.com/drive/folders/1w5Yekh-Yq2SjQmrAsVRWAWtB7xHletmK?usp=drive_link

#### TODO
* Optimize the code.

#### Citation
If you find the code or trained models useful, please consider citing:
```
@inproceedings{zou2023patchaugnet,
  title={PatchAugNet: Patch feature augmentation-based heterogeneous point cloud place recognition in large-scale street scenes},
  author={Xianghong Zou and Jianping Li and Yuan Wang and Fuxun Liang and Weitong Wu and Haiping Wang and Bisheng Yang and Zhen Dong},
  journal={ISPRS J},
  year={2023}
}
```

#### Acknowledgement
Our code refers to [PointNetVLAD](https://github.com/mikacuy/pointnetvlad), [PPT-Net](https://github.com/fpthink/PPT-Net), [Minkloc3DV2](https://github.com/jac99/MinkLoc3Dv2).
