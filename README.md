# Acknowledgement
__資料來源、影像處理方法之參考與其他本計畫相關資訊皆取自於[Kaggle | SPECT MPI](https://www.kaggle.com/selcankaplan/spect-mpi)__    
## Image Processing Information for each Versions  
每個影像處理版本都會移除影像中不必要的框線或數字。  
接著將每個block elements(心臟各個不同切面)，在第三維度做串接以形成3D volume(stress與rest狀態分開處理)。  
最終儲存的檔案型式為nii檔，矩陣大小89x89x40x2(第四維度的第1個是stress、第2個是rest)。  

除上述之外，ver0至ver2具有差異的部分如下:  
__verion 0__  
do nothing further.  

__verion 1__  
(1) The stress volume is registrated to the rest volume(3 dimensional, rigid-body).  
(2) And then, mask the excessive inferior wall signal by using the half-circle mask (block-wise), whos center is the centroid of the heart wall.  

__verion 2__   
The stress volume is registrated to the rest volume(3 dimensional, rigid-body).  

  
![image](https://github.com/chenchami/SPECT_MPI/blob/master/info/SPECT_MPI_flowchart.png)
## Model Architecture  
![image](https://github.com/chenchami/SPECT_MPI/blob/master/info/Fake3dNet_structure.png)
