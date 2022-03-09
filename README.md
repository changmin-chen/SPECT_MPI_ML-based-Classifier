### Acknowledgement
Data source and references for image processing methods: [Kaggle | SPECT MPI](https://www.kaggle.com/selcankaplan/spect-mpi)    

### Objective of this Project  
This project explored three different image processing pipelines, to test that image processing would or wouldn't help for training the machine learning model, which was a convolutional neural network (CNN) based classifier for distinguishing normal/abnormal myocardial perfusion status using solely the SPECT MPI images as its input data.


__Major differences between image processing pipline versions:__   
* __verion 0__  
Remove redundant frame lines, number labels, etc., from the original 2D images, then cropped and concatenated it to become a 3D heart volume. The input data for CNN model will have 2 channels: stress and rest (status of the heart during imaging).   

* __verion 1__  
Further than version 0, also:  
(1) The stress volume was spatially __registrated__ to the rest volume (3 dimensional, rigid-body).  
(2) The excessive signals located at inferior heart wall were __masked__ by the half-circle-mask, whos center was the centroid of the heart wall.  
 

* __verion 2__   
Further than version 0, also:  
(1) The stress volume was spatially __registrated__ to the rest volume (3 dimensional, rigid-body).  

![image](https://github.com/chenchami/SPECT_MPI/blob/master/info/SPECT_MPI_flowchart.png)  

### Model Architecture  
![image](https://github.com/chenchami/SPECT_MPI/blob/master/info/Fake3dNet_structure.png)
