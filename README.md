# SPECT_MPI
資料來源與本計畫相關資訊: https://www.kaggle.com/selcankaplan/spect-mpi  

## Image Processing Information for each versions:  
每個影像處理版本都會移除影像中不必要的框線或數字。  
接著將每個block elements(心臟各個不同切面)，在第三維度做串接以形成3D volume(stress與rest狀態分開處理)。  
最終儲存的檔案型式為nii檔，矩陣大小89x89x40x2(第四維度的第1個是stress、第2個是rest)。  

除上述之外，ver0至ver2具有差異的部分如下:  
ver 0:  
do nothing further.  

ver 1:  
(1) The stress volume is registrated to the rest volume(3 dimensional, rigid-body).  
(2) And then, mask the excessive inferior wall signal by using the half-circle mask (block-wise), whos center is the centroid of the heart wall.  

ver 2:  
The stress volume is registrated to the rest volume(3 dimensional, rigid-body).  

## Pipeline
根據檔案名稱中的step次序執行檔案  
已經過影像處理完畢的檔案存放於: https://drive.google.com/drive/folders/1EdcS08BG3pkm9ZGedpNDHEJ9-gkUouEI?usp=sharing  
(若已下載影像處理完畢的檔案，則直接從step2開始執行檔案即可)
