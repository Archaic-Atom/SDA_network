
# SDA_network
This is the implementation of the paper **Synthetic-to-real domain adaptation joint spatial feature transform for stereo matching**, Xing Li, Yangyu Fan, Zhibo Rao, Guoyun Lv, and Shiya Liu.
The code was written by Xing Li and Zhibo Rao.  

We propose a new method that translates the style of synthetic domain dataset to the real domain but maintains content and spatial information. 

**Synthetic-to-real domain translated results.** Blue circles emphasize the difference between the synthetic and translated images, including overall tone (blue sky), local color (the leaves), and sunlight reflection (the illuminate direction).  

<img src="images/1.jpg" width="900"/><br />
**The architecture of our proposed SDA network.** The core of our approach is to 1) compel the generated images to preserve content and spatial information with inputs, 2) prevent generated stereo left-right pairs mismatch. For this purpose, we leverage cues for edge features through a spatial feature transform layer to enforce spatial consistency between stereo images. Furthermore, we adopt the warp loss to encourage the warpped left image approach to the original left image.

## Result Videos
  
<img src='images/sceneflow_sf_tsf.gif' width="900"/><br />
**Left: Original SceneFlow. Right: SDA-Net generated translated SceneFlow.** Resolution: 960x574 <br />     
<img src='images/sceneflow_disp_error.gif' width="900"/><br />
**Disparity estimation results on the SceneFlow testing set. Model only trained on the translated SceneFlow training set.** Resolution: 960x574 <br />  
<img src='images/kitti_image_disp.gif' width="900"/><br />
**Disparity estimation results on the KITTI training set. Model only trained on syntehtic datasets.** Resolution: 1242x375 <br />  

## Software Environment
1. OS Environment  
    os == windows 10  
    cudaToolKit == 10.0  
    cudnn == 7.6.5  
2. Python Environment  
    python == 3.7.9  
    tensorflow == 1.15.0  
    numpy == 1.19.2  
    opencv-python == 4.5.1.48  
    pillow == 8.1.0  
## Data Preparation
Download [Scene Flow Datasets](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html), [Synthia](https://synthia-dataset.net/downloads/), [KITTI 2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo), [KITTI 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)

## 1. Generate train list and test list.
Get the Training list or Testing list （You need rewrite the code by your path, and my related code can be found in Source/Tools）
```
$ ./GenPath.sh
```
Please check the path. The source code in Source/Tools.

## 2. Train SDA-Net
Run the TrainSDANet.sh (This is training process. note that please check the img path should be found in related path, e.g. ./Dataset/trainlist_Kitti_Sceneflow.txt)
```
$ sh TrainSDANet.sh
```
Please carefully check the path in related file.

## 3. Test SDA_Net:
Run the TestSDANet.sh to output transalted stereo images.
```
$ sh TestSDANet.sh
```

## File Structure
```
.                          
├── Source # source code                 
│   ├── Basic       
│   ├── Evaluation       
│   └── ...                
├── Dataset # Get it by ./Source/Tools/GenPath.sh, you need build folder                   
│   ├── trainlist.txt   
│   ├── labellist.txt   
│   └── ...                
├── Result # The data of Project. Auto Bulid                   
│   ├── output.log   
│   ├── train_acc.csv   
│   └── ...       
├── ResultImg # The image of Result. Auto Bulid                   
│   ├── 000001_10.png   
│   ├── 000002_10.png   
│   └── ...       
├── PAModel # The saved model. Auto Bulid                   
│   ├── checkpoint   
│   └── ...   
├── log # The graph of model. Auto Bulid                   
│   ├── events.out.tfevents.1605153366.DESKTOP-GHD7UKT       
│   └── ...       
├── TrainSDANet.sh
├── TestSDANet.sh  
├── LICENSE
├── requirements.txt
└── README.md               
```
If you find SDA-Network useful for your work please cite:

@ARTICLE{lisynthetic,
  author={Li, Xing and Fan, Yangyu and Rao, Zhibo and Lv, Guoyun and Liu, Shiya},
  journal={IEEE Signal Processing Letters}, 
  title={Synthetic-to-real domain adaptation joint spatial feature transform for stereo matching}, 
  year={2021},
  pages={1-5},
  doi={10.1109/LSP.2021.3125264}}