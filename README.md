# SDA_network
This is the implementation of the paper **Synthetic-to-real domain adaptation joint spatial feature transform for stereo matching**, Xing Li, Yangyu Fan, Zhibo Rao, Guoyun Lv, and Shiya Liu.
The code was written by Xing Li and Zhibo Rao.

### Software Environment
1. OS Environment  
    os == windows 10  
    cudaToolKit == 10.0  
    cudnn == 7.6.5  
2. Python Environment  
    python == 3.7.9  
    tensorflow == 1.15.0  
    numpy == 1.19.2  
    opencv-python == 4.5.1.48  
    PIL == 5.1.0  

%### Model
%We have upload our model in baidu disk:
%https://pan.baidu.com/s/11FNUv8M5L4aO_Are9UjRUA
%password: qrho

### Hardware Environment
- GPU: 1080TI * 4 or other memory at least 11G.(Batch size: 2)  
if you not have four gpus, you could change the para of model. The Minimum hardware requirement:  
- GPU: memory at least 5G. (Batch size: 1)

### Train the model by running:
1. Get the Training list or Testing list
```
$ ./GenPath.sh
```
Please check the path. The source code in Source/Tools.

2. Run the pre-training.sh
```
$ ./Pre-Train.sh
```

3. Run the trainstart.sh
```
$ ./TrainKitti2012.sh # for kitti2012
$ ./TrainKitti2015.sh # for kitti2015
```

4. Run the teststart.sh
```
$ ./TestKitt2012.sh # for 2012
$ ./TestKitt2015.sh # for 2015
```

if you want to change the para of the model, you can change the *.sh file. Such as:
```
$ vi TestStart.sh
```

### File Struct
```
.                          
├── Source # source code                 
│   ├── Basic       
│   ├── Evaluation       
│   └── ...                
├── Dataset # Get it by ./GenPath.sh, you need build folder                   
│   ├── label_scene_flow.txt   
│   ├── trainlist_scene_flow.txt   
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
│   ├── events.out.tfevents.1541751559.ubuntu      
│   └── ...       
├── GetPath.sh
├── Pre-Train.sh
├── TestStart.sh  
├── TrainStart.sh
├── LICENSE
├── requirements.txt
└── README.md               
```
