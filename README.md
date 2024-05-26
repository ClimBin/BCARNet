# BCARNet
This is the official code implementation of the paper "ORSI Salient Object Detection via Bidirectional Cross-Attention and Attention Restoration"

## Test
If you want to test our model, you can place the data set and pre-trained parameters into the specified folder, and then run test.py. Please visit the following link to obtain the pre-trained model: [pre-trained model(ckpt)](https://drive.google.com/drive/folders/1CyuFBo8e0jixgskjqNvyJrptD2hDsj4v?usp=sharing)
```
Dataset/
│
├── ORSSD/
│ ├── testset
| |  |___images
| |  |___masks
│ └── trainset
|    |___images
|    |___masks
│ 
├── EORSSD/
│ ├── ...
│ └── ...
│
└── ORS-4199/
    ...
```

## Results
We also released the model prediction results both based on Resnet50 and PVT-b2, please visit: [prediction results](https://drive.google.com/drive/folders/1CyuFBo8e0jixgskjqNvyJrptD2hDsj4v?usp=sharing)

## Others
Because the manuscript is under review, the training code is not disclosed yet. And some codes are temporarily hidden in model.py, please understand.

In addition, because some existing evaluation codes are usually implemented based on Matlab, the evaluation is relatively slow. We have announced the evaluation code for the Python language version, which involves multi-threaded running jobs and enables rapid evaluation. This contains two files: eval.py and metrics.py 




