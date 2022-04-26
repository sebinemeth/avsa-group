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
#include <opencv2/video/tracking.hpp>

//namespaces
using namespace cv; //avoid using 'cv' to declare OpenCV functions and variables (cv::Mat or Mat)
using namespace std;


KalmanFilter KF_init(int measurementsize,int statesize,int contrSize)
{
	KalmanFilter KF(statesize,measurementsize,contrSize);

	if (statesize==4) //constant velocity
	{
		KF.transitionMatrix = (Mat_<float>(statesize, statesize) << //A
							1, 1, 0, 0,
							0, 1, 0, 0,
							0, 0, 1, 1,
							0, 0, 0, 1);


		KF.processNoiseCov = (Mat_<float>(statesize, statesize) << //Q
							25, 0, 0, 0,
							0, 10, 0, 0,
							0, 0, 25, 0,
							0, 0, 0, 10);


		KF.measurementMatrix = (Mat_<float>(measurementsize, statesize) << //H
							1, 0, 0, 0,
							0, 0, 1, 0);
	}

	if (statesize==6) //constant acceleration
	{
		KF.transitionMatrix = (Mat_<float>(statesize, statesize) << //A
					1, 1, 0.5f, 0, 0, 0,
					0, 1, 1, 0, 0, 0,
					0, 0, 1, 0, 0, 0,
					0, 0, 0, 1, 1, 0.5f,
					0, 0, 0, 0, 1, 1,
					0, 0, 0, 0, 0, 1);


		KF.processNoiseCov = (Mat_<float>(statesize, statesize) << //Q
					25, 0, 0, 0, 0, 0,
					0, 10, 0, 0, 0, 0, 0,
					0, 0, 1, 0, 0, 0, 0,
					0, 0, 0, 25, 0, 0,
					0, 0, 0, 0, 0, 10, 0,
					0, 0, 1, 0, 0, 0, 0);


		KF.measurementMatrix = (Mat_<float>(measurementsize, statesize) << //H
					1, 0, 0, 0, 0, 0,
					0, 0, 0, 1, 0, 0);
	}


	setIdentity(KF.measurementNoiseCov, Scalar::all(25)); //R
	setIdentity(KF.errorCovPost, Scalar::all(1e-5));  //P

	return KF;
}


void KF_algorithm(std::vector<cv::Point> &measurements, KalmanFilter &KF, bool &detected, int measurementsize, int statesize, cv::Point meas, Mat frame)
{
	Mat measurement = Mat::zeros(Size(1, measurementsize), CV_32F);

	measurement.at<float>(0)=meas.x;
	measurement.at<float>(1)=meas.y;
	Mat predicted_state = KF.predict(); //state prediction
	Point predictPt(predicted_state.at<float>(0),predicted_state.at<float>(statesize/2));
	Point measPt(measurement.at<float>(0),measurement.at<float>(1));

	// If the ball is not on the frame
	if (!detected){
		predicted_state.at<float>(0)=meas.x;
		predicted_state.at<float>(statesize/2)=meas.y;
		KF.statePost = predicted_state;
		measurements.push_back(measPt);
		//cout<<"Not detected"<<endl;

		if (meas.x > 0 and meas.y > 0){
			detected=true;
		}
	}

	// if the ball was found once on the frame
	if (detected){
		// in case of no observations on the frame
		if ((meas.x < 0 or meas.y < 0) or (meas.x > frame.cols or meas.y > frame.rows)){
			KF.statePost = predicted_state;
			measurements.push_back(predictPt);
			//cout<<"Detected, no observation"<<endl;
		}
		// in case the ball could be measured --> correction
		else{
			Mat estimated = KF.correct(measurement);
			Point estimatedPt(estimated.at<float>(0),estimated.at<float>(statesize/2));
			measurements.push_back(estimatedPt);
			//cout<<"Detected, there is an observation"<<endl;
		}
	}
}


//main function
int main(int argc, char ** argv)
{
	int count=0; // frame counter
	Mat frame;

	//...

	int _statesize=4; //X  ==4 in case of constant velocity, ==6 in case of constant acceleration

	int _measurementsize=2; //Z
	int _contrSize=0;
	bool _detected=false;
	std::vector<cv::Point> _measurements;
	KalmanFilter _KF = KF_init(_measurementsize,_statesize,_contrSize);

	//...

	// frame of the video sequence
	std::string inputvideo = "../dataset_lab3/lab3.1/singleball.mp4"; 	// path for the video to process

	//alternatively, the videofile can be passed via arguments of the executable
	if (argc == 3) inputvideo = argv[1];
	VideoCapture cap(inputvideo); // reader to grab frames from video

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
		//int a;
		//cin>>a;
		++count;
		//std::cout << "FRAME " << std::setfill('0') << std::setw(3) << count << std::endl; //print frame number

		//get frame & check if we achieved the end of the videofile (e.g. frame.data is empty)
		cap >> frame;
		if (!frame.data)
			break;

		//get measurement from preloaded list of measurements
		cv::Point meas = measList[i];


		//do kalman-based tracking
		//...

		KF_algorithm(_measurements, _KF, _detected, _measurementsize, _statesize, meas, frame);

		//...
		
		//display frame (YOU MAY ADD YOUR OWN VISUALIZATION FOR MEASUREMENTS, AND THE STAGES IMPLEMENTED)
		std::ostringstream str;
		str << std::setfill('0') << std::setw(3) << count;
		putText(frame,"Frame " + str.str(), cvPoint(30,30),FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(255,255,255), 1, CV_AA);
		putText(frame,"Measurement z_k ", cvPoint(30,50),FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(255,0,0), 1, CV_AA);
		putText(frame,"Estimated x_k ", cvPoint(30,70),FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0,0,255), 1, CV_AA);
		for (int j=0; j<=i; j++) {
			drawMarker(frame, measList[j], cvScalar(255,0,0), MARKER_CROSS, 20,2); //display measurement
			drawMarker(frame, _measurements[j], cvScalar(0,0,255), MARKER_CROSS, 20,2); //display estimations
		}
		//drawMarker(frame, meas, cvScalar(255,0,0), MARKER_CROSS, 20,2); //display measurement
		//drawMarker(frame, cvPoint(measurement), cvScalar(0,255,0), MARKER_CROSS, 20,2);
		imshow("Tracking results (trajectories) ",frame);

		// Saving the plots
		//...

		string filename = "./results/statesize_"+std::to_string(_statesize)+"/" + std::to_string(i) + ".png";
		imwrite(filename, frame);

		//...

		//cancel execution by pressing "ESC"
		if( (char)waitKey(100) == 27)
			break;
	}

	printf("Finished program.");
	destroyAllWindows(); 	// closes all the windows
	return 0;
}
