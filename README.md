# My Pseudo-LiDARv2 based on Argo dataset

## Convert Argo dataset to Kitti format
Using to convert argo dataset to Kitti, only using the stero image data from all ring cameras.
```bash
MyPseudoLidar$ python myargostereotokittiadapter.py --all_argodata

Total number of logs: 20
#images: 83515
#lidar: 3794
working on a log c6911883-1843-3727-8eaa-41dc8cda8993
....

Total number of logs: 20
#images: 99470
#lidar: 4518
working on a log 53037376-5303-5303-5303-553038557184
....

Total number of logs: 20
#images: 85761
#lidar: 3896
working on a log e9bb51af-1112-34c2-be3e-7ebe826649b4
....
Total number of logs: 5
#images: 20110
#lidar: 914

Total number of logs: 24
#images: 110466
#lidar: 5015

```
The generated Argo Kitti format dataset is located in
```bash
MyPseudoLidar$ ls /Developer/Dataset/Argoverse/argoverse-conv-rect-all/training/
argo_kitti_link.txt  calib  image_2  image_3  label_2  train.txt  val.txt  velodyne
```
argo_kitti_link.txt file is used to track the relationship between Argo and Kitti-format data, other folders are similar to Kitti

## Perform training
```bash
MyPseudoLidar$ python src/main.py -c ./src/configs/sdn_argo_fulltrain.config --dynamic_bs --resume /Developer/3DObject/MyPseudoLidarresults/sdn_argo/checkpoint_525.pth.tar --data_tag Argo_trainfull --argo
```
Trained model is located in
```bash
$ ls /Developer/3DObject/MyPseudoLidarresults/sdn_argo
checkpoint_795.pth.tar  model_best.pth.tar
```
checkpoint_795 is the last model, available in "P100: /Developer/3DObject/MyPseudoLidarresults/sdn_argo/checkpoint_795.pth.tar"

## Perform evaluation
```bash
MyPseudoLidar$ python src/main.py -c ./src/configs/sdn_argo_fulltrain.config --dynamic_bs --evaluate --data_tag Argo_trainfull --argo --resume /Developer/3DObject/MyPseudoLidarresults/sdn_argo/checkpoint_795.pth.tar
...
[2021-01-10 13:09:04 main.py:206] INFO     EVALUATE:1816        L 1.936 RLI 5.590 RLO 0.164 ABS 0.070 SQ 1.076 DEL 0.941 DELQ 0.968 DELC 0.981 Time:0.282
[2021-01-10 13:09:04 main.py:206] INFO     EVALUATE:1817        L 1.936 RLI 5.590 RLO 0.164 ABS 0.070 SQ 1.075 DEL 0.941 DELQ 0.968 DELC 0.981 Time:0.282
[2021-01-10 13:09:04 main.py:206] INFO     EVALUATE:1818        L 1.936 RLI 5.590 RLO 0.164 ABS 0.070 SQ 1.075 DEL 0.941 DELQ 0.968 DELC 0.981 Time:0.282
[2021-01-10 13:09:05 main.py:206] INFO     EVALUATE:1819        L 1.936 RLI 5.590 RLO 0.164 ABS 0.070 SQ 1.075 DEL 0.941 DELQ 0.968 DELC 0.981 Time:0.282
```
When "--evaluate" is added, the main.py will perform evaluation instead of training. One pdf figure file will be saved in the MyPseudoLidar root folder that documents all the major metrics
If using VSCode to debug the code, the args setup is
```bash
"args": ["-c", "./src/configs/sdn_argo_fulltrain.config", "--dynamic_bs", "--evaluate", "--resume", "/Developer/3DObject/MyPseudoLidarresults/sdn_argo/checkpoint_795.pth.tar", "--data_tag", "Argo_trainfull", "--argo"],
```

## Generate estimated depth map
```bash
MyPseudoLidar$ python src/main.py -c ./src/configs/sdn_argo_fulltrain.config --dynamic_bs --generate_depth_map --resume /Developer/3DObject/MyPseudoLidarresults/sdn_argo/checkpoint_795.pth.tar --data_tag Argo_trainfull --argo
```
Using the trained model to predict the depth map based on the TestImgLoader (val_data), the generated depth map is: "/Developer/3DObject/MyPseudoLidarresults/sdn_argo/depth_maps/Argo_trainfull/". All depth files are .npy from 000000.npy to 001902.npy

