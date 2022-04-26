/*
 * Kalman.h
 *
 *  Created on: Apr 26, 2022
 *      Author: avsa
 */

#ifndef SRC_KALMAN_H_
#define SRC_KALMAN_H_

#include <opencv2/opencv.hpp> 	//opencv libraries
#include <opencv2/video/tracking.hpp>

using namespace cv; //avoid using 'cv' to declare OpenCV functions and variables (cv::Mat or Mat)
using namespace std;

class Kalman {
public:
	Kalman(int statesize, Mat &frame);
	void run(cv::Point meas);
	std::vector<cv::Point> out_points;
	Mat print(vector<Point>& meas_history, Mat& frame);
	virtual ~Kalman();

private:
	KalmanFilter kalmanFilter;
	int stateSize;
	bool detected;
	cv::Size frameSize;
};
#endif /* SRC_KALMAN_H_ */

