/*
 * Kalman.cpp
 *
 *  Created on: Apr 26, 2022
 *      Author: avsa
 */

#include "Kalman.h"
#include <opencv2/opencv.hpp> 	//opencv libraries
#include <opencv2/video/tracking.hpp>

using namespace cv;
//avoid using 'cv' to declare OpenCV functions and variables (cv::Mat or Mat)
using namespace std;

Kalman::Kalman(int statesize, Mat &frame) :
		stateSize(statesize), detected(false), frameSize(frame.size()) {
	kalmanFilter = KalmanFilter(stateSize, 2, 0);

	int MEAS_NOISE_COV = 25;
	int PROC_NOISE_COV = 25;
	int PROC_VELOCITY = 10;

	setIdentity(kalmanFilter.errorCovPost, Scalar::all(1e5));  //P

	if (stateSize == 4) //constant velocity
			{
		kalmanFilter.transitionMatrix = (Mat_<float>(stateSize, stateSize) << //A
				1, 1, 0, 0, //
				0, 1, 0, 0, //
				0, 0, 1, 1, //
				0, 0, 0, 1);

		kalmanFilter.processNoiseCov =
				(Mat_<float>(stateSize, stateSize) << //Q
						PROC_NOISE_COV, 0, 0, 0, //
						0, PROC_VELOCITY, 0, 0, //
						0, 0, PROC_NOISE_COV, 0, //
						0, 0, 0, PROC_VELOCITY);

		kalmanFilter.measurementMatrix = (Mat_<float>(2, stateSize) << //H
				1, 0, 0, 0, //
				0, 0, 1, 0);

		//kalmanFilter.errorCovPost.at<float>(0,0) = 25;
		//kalmanFilter.errorCovPost.at<float>(2,2) = 25;
	}

	if (stateSize == 6) //constant acceleration
			{
		kalmanFilter.transitionMatrix = (Mat_<float>(stateSize, stateSize) << //A
				1, 1, 0.5f, 0, 0, 0, //
				0, 1, 1, 0, 0, 0, //
				0, 0, 1, 0, 0, 0, //
				0, 0, 0, 1, 1, 0.5f, //
				0, 0, 0, 0, 1, 1, //
				0, 0, 0, 0, 0, 1);

		kalmanFilter.processNoiseCov =
				(Mat_<float>(stateSize, stateSize) << //Q
						PROC_NOISE_COV, 0, 0, 0, 0, 0, //
						0, PROC_VELOCITY, 0, 0, 0, 0, //
						0, 0, 1, 0, 0, 0, //
						0, 0, 0, PROC_NOISE_COV, 0, 0, //
						0, 0, 0, 0, PROC_VELOCITY, 0, //
						0, 0, 0, 0, 0, 1);

		kalmanFilter.measurementMatrix = (Mat_<float>(2, stateSize) << //H
				1, 0, 0, 0, 0, 0, //
				0, 0, 0, 1, 0, 0);

//		kalmanFilter.errorCovPost.at<float>(0,0) = 15;
//		kalmanFilter.errorCovPost.at<float>(3,3) = 15;
	}

	setIdentity(kalmanFilter.measurementNoiseCov, Scalar::all(MEAS_NOISE_COV)); //R

}

void Kalman::run(cv::Point meas) {
	Mat measurement = Mat::zeros(Size(1, 2), CV_32F);

	measurement.at<float>(0) = meas.x;
	measurement.at<float>(1) = meas.y;
	Mat predicted_state = kalmanFilter.predict(); //state prediction
	Point predictPt(predicted_state.at<float>(0),
			predicted_state.at<float>(stateSize / 2));
	Point measPt(measurement.at<float>(0), measurement.at<float>(1));

	// If the ball is not on the frame
	if (!detected) {
		predicted_state.at<float>(0) = meas.x;
		predicted_state.at<float>(stateSize / 2) = meas.y;
		kalmanFilter.statePost = predicted_state;
		out_points.push_back(measPt);
		//cout<<"Not detected"<<endl;

		if (meas.x > 0 and meas.y > 0) {
			detected = true;
		}
	}

	// if the ball was found once on the frame
	if (detected) {
		// in case of no observations on the frame
		if ((meas.x < 0 or meas.y < 0)
				or (meas.x > frameSize.width or meas.y > frameSize.height)) {
			kalmanFilter.statePost = predicted_state;
			out_points.push_back(predictPt);
			//cout<<"Detected, no observation"<<endl;
		}
		// in case the ball could be measured --> correction
		else {
			Mat estimated = kalmanFilter.correct(measurement);
			Point estimatedPt(estimated.at<float>(0),
					estimated.at<float>(stateSize / 2));
			out_points.push_back(estimatedPt);
			//cout<<"Detected, there is an observation"<<endl;
		}
	}
	cout << out_points.size() << " " << out_points[out_points.size() - 1] << endl;
}

Mat Kalman::print(vector<Point> &meas_history, Mat &frame) {
	Mat out_frame;
	frame.copyTo(out_frame);
	putText(out_frame, "Measurement z_k ", cvPoint(30, 50),
			FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(255, 0, 0), 1, CV_AA);
	putText(out_frame, "Estimated x_k ", cvPoint(30, 70),
			FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 0, 255), 1, CV_AA);
	for (unsigned int j = 0; j < out_points.size(); j++) {
		if (meas_history[j].x > 0 && meas_history[j].y > 0)
			drawMarker(out_frame, meas_history[j], cvScalar(255, 0, 0),
					MARKER_CROSS, 20, 2); //display measurement
		drawMarker(out_frame, out_points[j], cvScalar(0, 0, 255), MARKER_CROSS,
				20, 2); //display estimations
	}
	return out_frame;
}

Kalman::~Kalman() {
	// TODO Auto-generated destructor stub
}

