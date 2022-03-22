/* Applied Video Analysis of Sequences (AVSA)
 *
 *	LAB2: Blob detection & classification
 *	Lab2.0: Sample Opencv project
 * 
 *
 * Authors: José M. Martínez (josem.martinez@uam.es), Paula Moral (paula.moral@uam.es), Juan C. San Miguel (juancarlos.sanmiguel@uam.es)
 */

//system libraries C/C++
#include <stdio.h>
#include <iostream>
#include <sstream>

//opencv libraries
#include <opencv2/opencv.hpp>
#include <opencv2/video/background_segm.hpp>

//Header ShowManyImages
#include "ShowManyImages.hpp"

//include for blob-related functions
#include "blobs.hpp"

//namespaces
using namespace cv; //avoid using 'cv' to declare OpenCV functions and variables (cv::Mat or Mat)
using namespace std;

#define MIN_WIDTH 20
#define MIN_HEIGHT 20

//main function
int main(int argc, char ** argv) 
{
	Mat frame; // current Frame
	Mat fgmask; // foreground mask
	std::vector<cvBlob> bloblist; // list for blobs
	std::vector<cvBlob> bloblistFiltered; // list for blobs

	// STATIONARY BLOBS
	Mat fgmask_history; // STATIONARY foreground mask
	Mat sfgmask; // STATIONARY foreground mask
	std::vector<cvBlob> sbloblist; // list for STATIONARY blobs
	std::vector<cvBlob> sbloblistFiltered; // list for STATIONARY blobs


	double t, acum_t; //variables for execution time
		int t_freq = getTickFrequency();

		//Paths for the dataset
//		// In this example we assume that the dataset is available at
//		// "/home/avsa/datasets/...
		string dataset_path = "../Lab2datasets/"; //SET THIS DIRECTORY according to your download
		string dataset_cat[1] = {""};
		//string baseline_seq[10] = {"AVSS2007/AVSSS07_EASY.mkv","AVSS2007/AVSSS07_HARD.mkv", "ETRI/ETRI_od_a.avi", "PETS2006/PETS2006_S1/PETS2006_S1_C3.mpeg","PETS2006/PETS2006_S4/PETS2006_S4_C3.avi","PETS2006/PETS2006_S5/PETS2006_S5_C3.mpeg","VISOR/visor_Video00.avi","VISOR/visor_Video01.avi","VISOR/visor_Video02.avi","VISOR/visor_Video03.avi"};
		string baseline_seq[1] = {
				"ETRI/ETRI_od_A.avi"
//				"PETS2006/PETS2006_S1/PETS2006_S1_C3.mpeg",
////				"PETS2006/PETS2006_S4/PETS2006_S4_C3.avi",
//				"PETS2006/PETS2006_S5/PETS2006_S5_C3.mpeg",
//				"VISOR/visor_Video00.avi",
//				"VISOR/visor_Video01.avi",
//				"VISOR/visor_Video02.avi",
//				"VISOR/visor_Video03.avi"
		};
		string image_path = ""; //path to images - this format allows to read consecutive images with filename inXXXXXX.jpq (six digits) starting with 000001

//		// "/home/avsa/datasets/dataset2012lite/dataset/baseline/...
//		string dataset_path = "../dataset2012lite/dataset"; //SET THIS DIRECTORY according to your download
//		string dataset_cat[1] = {"baseline"};
//		string baseline_seq[4] = {"highway","office","pedestrians","PETS2006"};
//		string image_path = "/in%06d.jpg"; //path to images - this format allows to read consecutive images with filename inXXXXXX.jpq (six digits) starting with 000001


	//	string dataset_path = "/home/avsa/AVSA2020datasets/AVSASlidesVideos_dataset"; //SET THIS DIRECTORY according to your download
	//	string dataset_cat[1] = {"foregroundSeg"};
	//	string baseline_seq[5] = {"hall.avi", "empty_office.avi","stationary_objects.avi","eps_hotstart.avi","eps_shadows.avi"};
	//	string image_path = ""; //path to images - this format allows to read consecutive images with filename inXXXXXX.jpq (six digits) starting with 000001


		//Paths for the results
		// In this example we assume that the results are stored in the project directory
		// "/home/avsa/eclipse-workspace/Lab1.1AVSA2020/
		string project_root_path = "../"; //SET THIS DIRECTORY according to your project
		string project_name = "Lab2.2"; //SET THIS DIRECTORY according to your project
		string results_path = project_root_path+"/"+project_name+"/results";

		// create directory to store results
		string makedir_cmd = "mkdir "+project_root_path+"/"+project_name;
		system(makedir_cmd.c_str());
		makedir_cmd = "mkdir "+results_path;
		system(makedir_cmd.c_str());


		int NumCat = sizeof(dataset_cat)/sizeof(dataset_cat[0]); //number of categories (have faith ... it works! ;) ... each string size is 32 -at leat for the current values-)

		//Loop for all categories
		for (int c=0; c<NumCat; c++ )
		{
					// create directory to store results for category
			string makedir_cmd = "mkdir "+results_path + "/" + dataset_cat[c];
			system(makedir_cmd.c_str());

			int NumSeq = sizeof(baseline_seq)/sizeof(baseline_seq[0]);  //number of sequences per category ((have faith ... it works! ;) ... each string size is 32 -at leat for the current values-)

			//Loop for all sequence of each category
			for (int s=0; s<NumSeq; s++ )
			{
			VideoCapture cap;//reader to grab videoframes

			//Compose full path of images
			string inputvideo = dataset_path + "/" + dataset_cat[c] + "/" + baseline_seq[s] + image_path;
			cout << "Accessing sequence at " << inputvideo << endl;

			//open the video file to check if it exists
			cap.open(inputvideo);
			if (!cap.isOpened()) {
				cout << "Could not open video file " << inputvideo << endl;
			return -1;
			}

			// create directory to store results for sequence
			string makedir_cmd = "mkdir "+results_path + "/" + dataset_cat[c] + "/" + baseline_seq[s];
			system(makedir_cmd.c_str());

			//MOG2 approach
			Ptr<BackgroundSubtractor> pMOG2 = cv::createBackgroundSubtractorMOG2();

			//main loop
			Mat img; // current Frame

			int it=1;
			acum_t=0;

			for (;;) {

				//get frame
				cap >> img;

				//check if we achieved the end of the file (e.g. img.data is empty)
				if (!img.data)
					break;


				//Time measurement
				t = (double)getTickCount();

				//apply algs
				img.copyTo(frame);
				// Compute fgmask
				double learningrate=-1; //default value (as starting point)
				// The value between 0 and 1 that indicates how fast the background model is
				// learnt. Negative parameter (default -1) value makes the algorithm to use some automatically chosen learning
				// rate. 0 means that the background model is not updated at all, 1 means that the background model
				// is completely reinitialized from the last frame.
				pMOG2->apply(frame, fgmask, learningrate);
				// 0 bkg, 255 fg, 127 (gray) shadows ...

				int connectivity = 8; // 4 or 8

				// Extract the blobs in fgmask
				extractBlobs(fgmask, bloblist, connectivity);
				//		cout << "Num blobs extracted=" << bloblist.size() << endl;
				removeSmallBlobs(bloblist, bloblistFiltered, MIN_WIDTH, MIN_HEIGHT);
				//		cout << "Num small blobs removed=" << bloblist.size()-bloblistFiltered.size() << endl;

				// Clasify the blobs in fgmask
				classifyBlobs(bloblistFiltered);

				// STATIONARY BLOBS
				if (it==1)
					{
					sfgmask = Mat::zeros(Size(fgmask.cols, fgmask.rows), CV_8UC1);
					fgmask_history = Mat::zeros(Size(fgmask.cols, fgmask.rows), CV_32FC1);
					}
				// Extract the STATIC blobs in fgmask
				extractStationaryFG(fgmask, fgmask_history, sfgmask);
				extractBlobs(sfgmask, sbloblist, connectivity);
				//		cout << "Num STATIONARY blobs extracted=" << sbloblist.size() << endl;
				
				int min_width=0;  // to set properly
				int min_height=0; // to set properly
			
				removeSmallBlobs(sbloblist, sbloblistFiltered, MIN_WIDTH, MIN_HEIGHT);
				//		cout << "Num STATIONARY small blobs removed=" << sbloblist.size()-sbloblistFiltered.size() << endl;


				// Clasify the blobs in fgmask
				classifyBlobs(sbloblistFiltered);

		
				//Time measurement
				t = (double)getTickCount() - t;
//		        if (_CONSOLE_DEBUG) cout << "proc. time = " << 1000*t/t_freq << " milliseconds."<< endl;
				acum_t=+t;

				//SHOW RESULTS
				//get the frame number and write it on the current frame

				string title= project_name + " | Frame - FgM - Stat FgM | Blobs - Classes - Stat Classes | BlobsFil - ClassesFil - Stat ClassesFil | ("+dataset_cat[c] + "/" + baseline_seq[s] + ")";

				auto paintedLabelled = paintBlobImage(frame,bloblistFiltered, true);
				stringstream it_ss;
				it_ss << setw(6) << setfill('0') << it;
				string filename = results_path + "/" + it_ss.str() + ".png";
				imwrite(filename, paintedLabelled);
				ShowManyImages(title, 6, frame, fgmask, sfgmask,
						paintBlobImage(frame,bloblistFiltered, false), paintedLabelled, paintBlobImage(frame,sbloblistFiltered, true));

				//exit if ESC key is pressed
				if(waitKey(30) == 27) break;
				it++;
			} //main loop

	cout << it-1 << "frames processed in " << 1000*acum_t/t_freq << " milliseconds."<< endl;


	//release all resources

	cap.release();
	destroyAllWindows();
	waitKey(0); // (should stop till any key is pressed .. doesn't!!!!!)
}
}
return 0;
}




