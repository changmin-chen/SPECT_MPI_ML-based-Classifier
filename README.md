# Acknowledgement
__資料來源、影像處理方法之參考與其他本計畫相關資訊皆取自於[Kaggle | SPECT MPI](https://www.kaggle.com/selcankaplan/spect-mpi)__    
## Image Processing Information for each versions:  
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

  
![image](https://github.com/chenchami/SPECT_MPI/blob/master/SPECT_MPI_flowchart.png)

## Pipeline
根據檔案名稱中的step次序執行檔案    
建議直接下載[影像處理完畢的檔案](https://drive.google.com/drive/folders/1EdcS08BG3pkm9ZGedpNDHEJ9-gkUouEI?usp=sharing)後直接從step1開始執行程式即可   
(若欲觀看影像處理過程，亦或想從原始影像處理開始執行各步驟，請見"IMGproc_toolbox"資料夾)  

## IMGproc_toolbox
程式碼說明:  
* SPECT_MPI_step0_imageProcessing:  
執行影像處理  
以原始影像作為輸入(預定存放於"data"資料夾)，並輸出影像處理後的影像，存放到"proc_data"資料夾。

* Example_imageProcessing_ver1_3Dregist_circleMasked:  
範例程式  
呈現影像處理過程中各個階段的結果(該程式碼以影像處理ver 1為例)。
