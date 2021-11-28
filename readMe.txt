Image Processing Info.

ver 0: concatenate to 3D only.
ver 1: masking the excessive inferior wall signal by using the half-circle mask (block-wise),
	whos center is the centroid of the heart wall. And then, the stress volume is registrated
	to the rest volume (estimation and application are both on masked images).

ver 2: masking the excessive inferior wall signal by using Laplacian edge detection and region growing (block-wise).
	And then, the stress volume is registrated to the rest volume (estimation on masked images, application on original images).
