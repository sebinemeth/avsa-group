/* Applied Video Sequence Analysis - Escuela Politecnica Superior - Universidad Autonoma de Madrid
 *
 *	This source code belongs to the template (sample program)
 *	provided for the assignment LAB 4 "Histogram-based tracking"
 *
 *	This code has been tested using:
 *	- Operative System: Ubuntu 18.04
 *	- OpenCV version: 3.4.4
 *	- Eclipse version: 2019-12
 *
 * Author: Juan C. SanMiguel (juancarlos.sanmiguel@uam.es)
 * Date: April 2020
 */
//includes
#include <stdio.h> 								//Standard I/O library
#include <numeric>								//For std::accumulate function
#include <string> 								//For std::to_string function
#include <opencv2/opencv.hpp>					//opencv libraries
#include "utils.hpp" 							//for functions readGroundTruthFile & estimateTrackingPerformance
#include "ShowManyImages.hpp"

//namespaces
using namespace cv;
using namespace std;

//main function
void converter(Mat src, Mat &output, String channel)
{
	if (channel=="gr")
		cvtColor(src, output, cv::COLOR_RGB2GRAY);
	else if (channel=="h" or channel=="s")
	{
		cvtColor(src, output, cv::COLOR_RGB2HSV);
		vector<Mat> channels;
		split(output, channels);
		if (channel=="h")
			output=channels[0];
		if (channel=="s")
			output=channels[1];
	}
	else if (channel=="r" or channel=="g" or channel=="b")
	{
			vector<Mat> channels;
			split(src, channels);
			if (channel=="r")
				output=channels[0];
			if (channel=="g")
				output=channels[1];
			if (channel=="b")
				output=channels[2];
	}


}

Mat create_histogram(Mat ROI, int n_bins, bool col_features)
{
	if (col_features)
	{
		Mat hist;
		float range[] = { 0, 256 }; //the upper boundary is exclusive
		const float* histRange = { range };
		bool uniform = true, accumulate = false;
		calcHist( &ROI, 1, 0, Mat(), hist, 1, &n_bins, &histRange, uniform, accumulate);

		return hist;
	}
	else
	{
		HOGDescriptor hog;
		Mat ResizedCroppedOutput;
		resize(ROI, ResizedCroppedOutput, Size(64,128));
		hog.nbins=n_bins;
		vector< float > descriptors;
		hog.compute(ResizedCroppedOutput, descriptors);
		Mat hist=Mat(descriptors).clone();

		return hist;
	}

}


Mat visualize_histogram(Mat hist, int histSize)
{
	int hist_w = 512, hist_h = 400;
	int bin_w = cvRound( (double) hist_w/histSize);
	Mat histImage = Mat::zeros( hist_h, hist_w, CV_8UC1);
	normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

	for( int i = 1; i < histSize; i++ )
		{
			line( histImage, Point( bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1)) ),
				  Point( bin_w*(i), hist_h - cvRound(hist.at<float>(i)) ),
				  Scalar( 255, 0, 0), 2, 8, 0  );
		}
	return histImage;
}



int main(int argc, char ** argv)
{
	//PLEASE CHANGE 'dataset_path' & 'output_path' ACCORDING TO YOUR PROJECT
	std::string dataset_path = "../dataset_lab4/";									//dataset location.
	std::string output_path = "../outvideos/";									//location to save output videos

	// dataset paths
	std::string sequences[] = {"bolt1"};//};//,										//test data for lab4.1, 4.3 & 4.5
							   //"sphere","car1",								//test data for lab4.2
							   //"ball2","basketball",						//test data for lab4.4
							   //"bag","ball","road"};						//test data for lab4.6
	std::string image_path = "%08d.jpg"; 									//format of frames. DO NOT CHANGE
	std::string groundtruth_file = "groundtruth.txt"; 						//file for ground truth data. DO NOT CHANGE
	int NumSeq = sizeof(sequences)/sizeof(sequences[0]);					//number of sequences

	//Loop for all sequence of each category
	for (int s=0; s<NumSeq; s++ )
	{
		Mat frame;										//current Frame
		int frame_idx=0;								//index of current Frame
		std::vector<Rect> list_bbox_est, list_bbox_gt;	//estimated & groundtruth bounding boxes
		std::vector<double> procTimes;					//vector to accumulate processing times

		std::string inputvideo = dataset_path + "/" + sequences[s] + "/img/" + image_path; //path of videofile. DO NOT CHANGE
		VideoCapture cap(inputvideo);	// reader to grab frames from videofile

		//check if videofile exists
		if (!cap.isOpened())
			throw std::runtime_error("Could not open video file " + inputvideo); //error if not possible to read videofile

		// Define the codec and create VideoWriter object.The output is stored in 'outcpp.avi' file.
		cv::Size frame_size(cap.get(cv::CAP_PROP_FRAME_WIDTH),cap.get(cv::CAP_PROP_FRAME_HEIGHT));//cv::Size frame_size(700,460);
		VideoWriter outputvideo(output_path+"outvid_" + sequences[s]+".avi",CV_FOURCC('X','V','I','D'),10, frame_size);	//xvid compression (cannot be changed in OpenCV)

		//Read ground truth file and store bounding boxes
		std::string inputGroundtruth = dataset_path + "/" + sequences[s] + "/" + groundtruth_file;//path of groundtruth file. DO NOT CHANGE
		list_bbox_gt = readGroundTruthFile(inputGroundtruth); //read groundtruth bounding boxes

		//main loop for the sequence
		std::cout << "Displaying sequence at " << inputvideo << std::endl;
		std::cout << "  with groundtruth at " << inputGroundtruth << std::endl;
		for (;;) {
			//get frame & check if we achieved the end of the videofile (e.g. frame.data is empty)
			cap >> frame;
			if (!frame.data)
				break;

			//Time measurement
			double t = (double)getTickCount();
			frame_idx=cap.get(cv::CAP_PROP_POS_FRAMES);			//get the current frame

			////////////////////////////////////////////////////////////////////////////////////////////
			//DO TRACKING
			//Change the following line with your own code
			list_bbox_est.push_back(Rect(20,20,40,50));//we use a fixed value only for this demo program. Remove this line when you use your code
			//...
			// ADD YOUR CODE HERE
			//...
			//parameters
			String color="gr";//gr,h,s,r,g,b
			int n_bins =16;//256; modify here the number of bins
			bool col_features = true; //set false in order to obtain gradient features


			//conversion
			Mat output;
			converter(frame, output, color);

			//selecting ROI
			cv::Mat croppedOutput = output(list_bbox_est[frame_idx-1]);
			cv::Mat croppedEstimation= frame(list_bbox_est[frame_idx-1]);
			cv::Mat croppedGroundTruth= frame(list_bbox_gt[frame_idx-1]);


			//histogram creation
			Mat hist = create_histogram(croppedOutput, n_bins, col_features);
			Mat histImage = visualize_histogram(hist, n_bins);




			////////////////////////////////////////////////////////////////////////////////////////////

			//Time measurement
			procTimes.push_back(((double)getTickCount() - t)*1000. / cv::getTickFrequency());
			//std::cout << " processing time=" << procTimes[procTimes.size()-1] << " ms" << std::endl;

			// plot frame number & groundtruth bounding box for each frame
			/*
			putText(frame, std::to_string(frame_idx), cv::Point(10,15),FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255)); //text in red
			rectangle(frame, list_bbox_gt[frame_idx-1], Scalar(0, 255, 0));		//draw bounding box for groundtruth
			rectangle(frame, list_bbox_est[frame_idx-1], Scalar(0, 0, 255));	//draw bounding box (estimation)

			//show & save data
			imshow("Tracking for "+sequences[s]+" (Green=GT, Red=Estimation)", frame);
			outputvideo.write(frame);//save frame to output video
			*/
			putText(frame,"Frame " + std::to_string(frame_idx), cv::Point(10,15),FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
			rectangle(frame, list_bbox_gt[frame_idx-1], Scalar(0, 255, 0));		//draw bounding box for groundtruth
			rectangle(frame, list_bbox_est[frame_idx-1], Scalar(0, 0, 255));	//draw bounding box (estimation)

			Mat manyimages = ShowManyImages("Frames", 4, frame, croppedOutput, croppedEstimation, histImage);

			// Saving the plots
			string est_filename = "./results/"+color+"/" + "est_" + std::to_string(frame_idx-1) + ".png";
			imwrite(est_filename, croppedEstimation);

			string hist_filename = "./results/"+color+"/" + "hist_" + std::to_string(frame_idx-1) + ".png";
			imwrite(hist_filename, histImage);

			string frame_filename = "./results/"+color+"/" + "frame_" + std::to_string(frame_idx-1) + ".png";
			imwrite(frame_filename, frame);

			//exit if ESC key is pressed
			if(waitKey(30) == 27) break;
		}

		//comparison groundtruth & estimation
		vector<float> trackPerf = estimateTrackingPerformance(list_bbox_gt, list_bbox_est);

		//print stats about processing time and tracking performance
		std::cout << "  Average processing time = " << std::accumulate( procTimes.begin(), procTimes.end(), 0.0) / procTimes.size() << " ms/frame" << std::endl;
		std::cout << "  Average tracking performance = " << std::accumulate( trackPerf.begin(), trackPerf.end(), 0.0) / trackPerf.size() << std::endl;

		//release all resources
		cap.release();			// close inputvideo
		outputvideo.release(); 	// close outputvideo
		destroyAllWindows(); 	// close all the windows
	}
	printf("Finished program.");
	return 0;
}
