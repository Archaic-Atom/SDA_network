# SDA_network
This is the implementation of the paper **Synthetic-to-real domain adaptation joint spatial feature transform for stereo matching**, Xing Li, Yangyu Fan, Zhibo Rao, Guoyun Lv, and Shiya Liu.
The code was written by Xing Li and Zhibo Rao.  

We propose a new method that translates the style of synthetic domain dataset to the real domain but maintains content and spatial information. The visual results are as follows.  
<img src="images/1.jpg" width="900"/>  
**Synthetic-to-real domain translated results.** Blue circles emphasize the difference between the synthetic and translated images, including overall tone (blue sky), local color (the leaves), and sunlight reflection (the illuminate direction).

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
    pillow == 8.1.0  


### Training:

Run the TrainSDANet.sh
```
$ sh TrainSDANet.sh
```

### Testing:

Run the TestSDANet.sh
```
$ sh TestSDANet.sh
```

### File Structure
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
