The 'dataset' folder contains 7 folders (one for each category).
Each category folder contains 4 to 6 folders (one for each video).
Each video folder contains :
  a folder named 'input' containing a separate JPEG file for each frame of the input video
  a folder named 'groundtruth' containing a separate BMP file for each frame of the groundtruth
  files named 'ROI.bmp' and 'ROI.jpg' showing the spatial region of interest
  a file named 'temporalROI.txt' containing two frame numbers. Only the frames in this range will be used to calculate your score

The 'results' folder has the same file tree, except the video folders are empty.
You need to put your results in each video folder.

To process the dataset, we offer matlab and python/c++ code. Take a look at the CODE section on http://changedetection.net/.

Example : the 'highway' sequence	(temporalROI.txt : 470 1700)
You will use your method to process all the frame in the 'input' folder. From in000001.jpg to bin001700.bmp.
You will save the results in '?/results/baseline/highway/'. (bmp, jpeg, png or pgm)
in000001.jpg -> bin000001.bmp (will not be used)
in000002.jpg -> bin000002.bmp (will not be used)
...
in000470.jpg -> bin000470.bmp
...
in001700.jpg -> bin001700.bmp

Repeat for all sequences, zip the 'results' folder and upload 'results.zip' to http://changedetection.net/.