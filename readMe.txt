Image Processing Info.
每個版本都會移除影像中不必要的框框線或數字，然後串成3D volume
儲存的檔案型式為nii檔，矩陣大小89x89x40x2 (第四維度的第1個是stress、第2個是rest)


除此之外，ver0 ~ verN 有些許不同:

ver 0: concatenate to 3D only.

ver 1: The stress volume is registrated to the rest volume. And then, mask the 
	excessive inferior wall signal by using the half-circle mask (block-wise),
	whos center is the centroid of the heart wall.

ver 2: The stress volume is registrated to the rest volume. And then, mask the
	excessive inferior wall signal by using Laplacian mask (block-wise).
	 