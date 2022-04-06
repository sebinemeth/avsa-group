/* Applied Video Sequence Analysis - Escuela Politecnica Superior - Universidad Autonoma de Madrid
 *
 *	Starter code for Task 3.1b of the assignment "Lab3 - Kalman Filtering for object tracking"
 *
 *	This code has been tested using Ubuntu 18.04, OpenCV 3.4.4 & Eclipse 2019-12
 *
 * Author: Juan C. SanMiguel (juancarlos.sanmiguel@uam.es)
 * Date: March 2022
 */
//includes
#include <opencv2/opencv.hpp> 	//opencv libraries

//namespaces
using namespace cv; //avoid using 'cv' to declare OpenCV functions and variables (cv::Mat or Mat)
using namespace std;

//main function
int main(int argc, char ** argv)
{
	int count=0;		 											// frame counter
	Mat frame;														// frame of the video sequence
	std::string inputvideo = "../dataset/lab3.1/singleball.mp4"; 	// path for the video to process

	//alternatively, the videofile can be passed via arguments of the executable
	if (argc == 3) inputvideo = argv[1];
	VideoCapture cap(inputvideo);							// reader to grab frames from video

	//check if videofile exists
	if (!cap.isOpened())
		throw std::runtime_error("Could not open video file " + inputvideo); //throw error if not possible to read videofile

	//preload measurements from txt file
	std::ifstream ifile("./src/meas_singleball.txt"); 	//filename with measurements (each line corresponds to X-Y coordinates of the measurement obtained for each frame)
	std::vector<cv::Point> measList; 					//variable where measurements will be stored
	std::string line; 									// auxiliary variable to read each line of file
	while (std::getline(ifile, line)) // read the current line
	{
		std::istringstream iss{line}; // construct a string stream from line

		// read the tokens from current line separated by comma
		std::vector<std::string> tokens; // here we store the tokens
		std::string token; // current token
		while (std::getline(iss, token, ' '))
			tokens.push_back(token); // add the token to the vector

		measList.push_back(cv::Point(std::stoi(tokens[0]),std::stoi(tokens[1])));
		//std::cout << "Processed point: " << std::stoi(tokens[0]) << " " << std::stoi(tokens[1]) << std::endl; //display read data
	}

	//main loop
	for (int i=0;true;i++) {

		std::cout << "FRAME " << std::setfill('0') << std::setw(3) << ++count << std::endl; //print frame number

		//get frame & check if we achieved the end of the videofile (e.g. frame.data is empty)
		cap >> frame;
		if (!frame.data)
			break;

		//get measurement from preloaded list of measurements
		cv::Point meas = measList[i];

		//do kalman-based tracking
		//PLACE YOUR CODE HERE
		//...
		//...
		
		//display frame (YOU MAY ADD YOUR OWN VISUALIZATION FOR MEASUREMENTS, AND THE STAGES IMPLEMENTED)
		std::ostringstream str;
		str << std::setfill('0') << std::setw(3) << count;
		putText(frame,"Frame " + str.str(), cvPoint(30,30),FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(255,255,255), 1, CV_AA);
		drawMarker(frame, meas, cvScalar(255,0,0), MARKER_CROSS, 20,2); //display measurement
		imshow("Frame ",frame);

		//cancel execution by pressing "ESC"
		if( (char)waitKey(100) == 27)
			break;
	}

	printf("Finished program.");
	destroyAllWindows(); 	// closes all the windows
	return 0;
}
