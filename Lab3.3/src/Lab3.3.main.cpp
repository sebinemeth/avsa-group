/* Applied Video Sequence Analysis - Escuela Politecnica Superior - Universidad Autonoma de Madrid
 *
 *	Starter code for Task 3.1a of the assignment "Lab3 - Kalman Filtering for object tracking"
 *
 *	This code has been tested using Ubuntu 18.04, OpenCV 3.4.4 & Eclipse 2019-12
 *
 * Author: Juan C. SanMiguel (juancarlos.sanmiguel@uam.es)
 * Date: March 2022
 */
//includes
#include <opencv2/opencv.hpp> 	//opencv libraries
#include "ShowManyImages.hpp"
#include "blobs.hpp"
#include "Kalman.h"
#include <fstream>

//namespaces
using namespace cv; //avoid using 'cv' to declare OpenCV functions and variables (cv::Mat or Mat)
using namespace std;

//main function
int main(int argc, char ** argv)
{
	int count=0;		 											    // frame counter
	Mat frame;														    // frame of the video sequence
	std::string inputvideo = "../dataset_lab3/lab3.3/abandonedBox_600_1000_clip.mp4"; 	// path for the video to process

	Mat bgMask, opening, detectionhistimg;
	vector<cvBlob> bloblist;
	vector<Point> detectionhistory;
	string results_path = "./results"; // folder needs to be created beforehand

	//alternatively, the videofile can be passed via arguments of the executable
	if (argc == 3) inputvideo = argv[1];
	VideoCapture cap(inputvideo);							// reader to grab frames from video

	//check if videofile exists
	if (!cap.isOpened())
		throw std::runtime_error("Could not open video file " + inputvideo); //throw error if not possible to read videofile

	// initialize detector
	auto MOG2 = createBackgroundSubtractorMOG2();
	MOG2->setVarThreshold(16);
	MOG2->setHistory(100);
	MOG2->setDetectShadows(false);

	// output file
	ofstream resultfile(results_path + "/detections.txt");

	// inititlize kalman
	Kalman* kalman;
	const int KALMAN_STATESIZE = 6;

	//main loop
	for (;;) {

		std::cout << "FRAME " << std::setfill('0') << std::setw(3) << ++count << std::endl; //print frame number

		//get frame & check if we achieved the end of the videofile (e.g. frame.data is empty)
		cap >> frame;
		if (!frame.data)
			break;

		//do measurement extraction
		//PLACE YOUR CODE HERE
		//...

		// bg segmentation
		MOG2->apply(frame, bgMask, 0.0001);
		
		// perform opening morph operation
		int morph_size = 1;
		cv::threshold(bgMask, opening, 200, 255, CV_THRESH_BINARY); // do not consider shadows
		morphologyEx(opening, opening, MORPH_OPEN,
				getStructuringElement(MORPH_RECT, Size(2*morph_size+1,2*morph_size+1), Point(morph_size,morph_size)));

		// blob detection
		extractBlobs(opening, bloblist, 4);
		removeSmallBlobs(bloblist, bloblist, 30, 30);

		// save blobs
		Point cp;
		if (bloblist.size() > 0 && count > 1) {
			cvBlob max_blob = maxBlob(bloblist);
			std::cout << max_blob.w << "x" << max_blob.h << " " <<
					     max_blob.x << "," << max_blob.y << std::endl;
			cp = Point(max_blob.x + max_blob.w / 2, max_blob.y + max_blob.h / 2);
		} else {
			cp = Point(-100,-100);
		}
		cout << cp << endl;
		resultfile << cp.x << " " << cp.y << endl;
		detectionhistory.push_back(cp);

		frame.copyTo(detectionhistimg);
		for (auto p : detectionhistory)
			ellipse(detectionhistimg,p,Size(5,5),0,0,360,Scalar(255,0,0),2,8);
		//...

		if (count == 1)
			kalman = new Kalman(KALMAN_STATESIZE, frame);

		kalman->run(cp);

		//display frame (YOU MAY ADD YOUR OWN VISUALIZATION FOR MEASUREMENTS, AND THE STAGES IMPLEMENTED)
		std::ostringstream str;
		str << std::setfill('0') << std::setw(3) << count;
		putText(frame,"Frame " + str.str(), cvPoint(30,30),FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(255,255,255), 1, CV_AA);
		putText(bgMask,"Background mask", cvPoint(30,30),FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(255,255,255), 1, CV_AA);
		putText(opening,"After opening", cvPoint(30,30),FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(255,255,255), 1, CV_AA);
		putText(detectionhistimg,"Detection history", cvPoint(30,30),FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(255,255,255), 1, CV_AA);
		Mat kalmanImage = kalman->print(detectionhistory, frame);

		Mat manyimages = ShowManyImages("Frames", 6, frame, bgMask, opening, paintBlobImage(frame, bloblist, false), detectionhistimg, kalmanImage);
		stringstream count_ss;
		count_ss << setw(6) << setfill('0') << count;
		string filename = results_path + "/abandoned/02/frame" + count_ss.str() + ".png";
		if (count % 50 == 0)
			imwrite(filename, manyimages);

//		imwrite("./boats.jpg", kalmanImage);

		//cancel execution by pressing "ESC"
		if( (char)waitKey(100) == 27)
			break;

		if(count > 400) {
//			break;
		}
	}

	printf("Finished program.");
	destroyAllWindows(); 	// closes all the windows
	resultfile.close();
	return 0;
}
