/* Applied Video Analysis of Sequences (AVSA)
 *
 *	LAB2: Blob detection & classification
 *	Lab2.0: Sample Opencv project
 *
 *
 * Authors: José M. Martínez (josem.martinez@uam.es), Paula Moral (paula.moral@uam.es), Juan C. San Miguel (juancarlos.sanmiguel@uam.es)
 */

#include "blobs.hpp"
#include "math.h"


/**
 *	Draws blobs with different rectangles on the image 'frame'. All the input arguments must be
 *  initialized when using this function.
 *
 * \param frame Input image
 * \param pBlobList List to store the blobs found
 * \param labelled - true write label and color bb, false does not wirite label nor color bb
 *
 * \return Image containing the draw blobs. If no blobs have to be painted
 *  or arguments are wrong, the function returns a copy of the original "frame".
 *
 */
Mat paintBlobImage(cv::Mat frame, std::vector<cvBlob> bloblist, bool labelled) {
	cv::Mat blobImage;
	//check input conditions and return original if any is not satisfied
	//...
	frame.copyTo(blobImage);

	//required variables to paint
	//...

	//paint each blob of the list
	for (int i = 0; i < bloblist.size(); i++) {
		cvBlob blob = bloblist[i]; //get ith blob
		//...
		Scalar color;
		std::string label = "";
		switch (blob.label) {
		case PERSON:
			color = Scalar(255, 0, 0);
			label = "PERSON";
			break;
		case CAR:
			color = Scalar(0, 255, 0);
			label = "CAR";
			break;
		case OBJECT:
			color = Scalar(0, 0, 255);
			label = "OBJECT";
			break;
		default:
			color = Scalar(255, 0, 0);
			label = "UNKOWN";
		}

		Point p1 = Point(blob.x, blob.y);
		Point p2 = Point(blob.x + blob.w, blob.y + blob.h);

		rectangle(blobImage, p1, p2, color, 1, 8, 0);
		if (labelled) {
			rectangle(blobImage, p1, p2, color, 1, 8, 0);
			putText(blobImage, label, p1, FONT_HERSHEY_SIMPLEX, 0.5, color);
		} else
			rectangle(blobImage, p1, p2, Scalar(255, 255, 255), 1, 8, 0);
	}

	//destroy all resources (if required)
	//...

	//return the image to show
	return blobImage;
}

/**
 *	Blob extraction from 1-channel image (binary). The extraction is performed based
 *	on the analysis of the connected components. All the input arguments must be 
 *  initialized when using this function.
 *
 * \param fgmask Foreground/Background segmentation mask (1-channel binary image) 
 * \param bloblist List with found blobs
 *
 * \return Operation code (negative if not succesfull operation) 
 */
Mat extractBlobs(cv::Mat fgmask, std::vector<cvBlob> &bloblist,
		int connectivity) {
	//check input conditions and return -1 if any is not satisfied
	//...		

	if (connectivity != 8 && connectivity != 4) {
		std::cerr << "Unknown connectivity" << std::endl;
		throw Exception();
	}

	//required variables for connected component analysis 
	//...
	Mat aux; // image to be updated each time a blob is detected (blob cleared)
	fgmask.convertTo(aux, CV_32SC1);

	//clear blob list (to fill with this function)
	bloblist.clear();

	//Connected component analysis

	// void creation of a unqie blob in the center
//	cvBlob blob = initBlob(1, fgmask.cols / 2, fgmask.rows / 2, fgmask.cols / 4,
//			fgmask.rows / 4);
//	bloblist.push_back(blob);
	unsigned blobId = 0;
	unsigned blobValue = 1;

	for (int i = 0; i < fgmask.rows; ++i) {
		for (int j = 0; j < fgmask.cols; ++j) {
			auto c = aux.at<int>(i, j);
			if (c == 255) {
				blobId++;
				cv::Rect rect;
				cv::floodFill(aux, cv::Point(j, i), rand() % 256, &rect, Scalar(),
						Scalar(), connectivity);
				cvBlob blob = initBlob(blobId, rect.x, rect.y, rect.width,
						rect.height, i, j);
				bloblist.push_back(blob);
			}
		}
	}

//	std::cout << bkg << " " << fg << " " << sh <<" " << fill << " " << unknown << " "<< bkg+fg+sh+unknown  << " " << fgmask.rows*fgmask.cols << std::endl;
//	std::cout << blob_id << " " << small_blobs << std::endl;

	//destroy all resources
	//...

	//return OK code
	aux.convertTo(aux, CV_8UC1);
	return aux;
}

int removeSmallBlobs(std::vector<cvBlob> bloblist_in,
		std::vector<cvBlob> &bloblist_out, int min_width, int min_height) {
	//check input conditions and return -1 if any is not satisfied

	//required variables
	//...

	//clear blob list (to fill with this function)
	bloblist_out.clear();

	for (int i = 0; i < bloblist_in.size(); i++) {
		cvBlob blob_in = bloblist_in[i]; //get ith blob

		if (blob_in.w >= min_width && blob_in.h >= min_height)
			bloblist_out.push_back(blob_in);

	}
	//destroy all resources
	//...

	//return OK code
	return 1;
}

/**
 *	Blob classification between the available classes in 'Blob.hpp' (see CLASS typedef). All the input arguments must be
 *  initialized when using this function.
 *
 * \param frame Input image
 * \param fgmask Foreground/Background segmentation mask (1-channel binary image)
 * \param bloblist List with found blobs
 *
 * \return Operation code (negative if not succesfull operation)
 */

// ASPECT RATIO MODELS
#define MEAN_PERSON 0.3950
#define STD_PERSON 0.1887

#define MEAN_CAR 1.4736
#define STD_CAR 0.2329

#define MEAN_OBJECT 1.2111
#define STD_OBJECT 0.4470

// end ASPECT RATIO MODELS

// distances
float ED(float val1, float val2) {
	return sqrt(pow(val1 - val2, 2));
}

float WED(float val1, float val2, float std) {
	return sqrt(pow(val1 - val2, 2) / pow(std, 2));
}
// helper struct containing the relative distance from a CLASS mean
struct dfromcls {
	CLASS cls;
	float d;
	// implement comparison operator to be able to run a minimum search on it
	bool operator<(dfromcls dfc) {
		return d < dfc.d;
	}
};
//end distances
int classifyBlobs(std::vector<cvBlob> &bloblist) {
	//check input conditions and return -1 if any is not satisfied
	//...

	//required variables for classification
	//...

	//classify each blob of the list
	for (cvBlob &blob : bloblist) {
		//...
		double ar = blob.w / (double) blob.h; // aspect ration
		std::vector<dfromcls> v = {{UNKNOWN, 1e10f}}; // vector containing distances from valid classes

		if (ED(MEAN_PERSON, ar) <= STD_PERSON)
			v.push_back({PERSON, WED(MEAN_PERSON, ar, STD_PERSON)});
		if (ED(MEAN_CAR, ar) <= STD_CAR)
			v.push_back({CAR, WED(MEAN_CAR, ar, STD_CAR)});
		if (ED(MEAN_OBJECT, ar) <= STD_OBJECT)
			v.push_back({OBJECT, WED(MEAN_OBJECT, ar, STD_OBJECT)});

		auto it = std::min_element(v.begin(), v.end()); // find CLASS with smallest relative distance
		blob.label = it->cls;

		// void implementation (does not change label -at creation UNKNOWN-)
	}

	//destroy all resources
	//...

	//return OK code
	return 1;
}

//stationary blob extraction function
/**
 *	Stationary FG detection
 *
 * \param fgmask Foreground/Background segmentation mask (1-channel binary image)
 * \param fgmask_history Foreground history counter image (1-channel integer image)
 * \param sfgmask Foreground/Background segmentation mask (1-channel binary image)
 *
 * \return Operation code (negative if not succesfull operation)
 *
 *
 * Based on: Stationary foreground detection for video-surveillance based on foreground and motion history images, D.Ortego, J.C.SanMiguel, AVSS2013
 *
 */

#define FPS 25 //check in video - not really critical
#define SECS_STATIONARY 1.5 // to set
#define I_COST 1.0f // to set // increment cost for stationarity detection
#define D_COST 3.0f // to set // decrement cost for stationarity detection
#define STAT_TH 0.5 // to set

int extractStationaryFG(Mat fgmask, Mat &fgmask_history, Mat &sfgmask) {

	int numframes4static = (int) (FPS * SECS_STATIONARY);

	// update fgmask_counter
	for (int i = 0; i < fgmask.rows; i++)
		for (int j = 0; j < fgmask.cols; j++) {
			// ...
			int is_fg = fgmask.at<uchar>(i, j) == 255;
			if (is_fg)
				fgmask_history.at<float>(i, j) += I_COST * is_fg;
			else
				fgmask_history.at<float>(i, j) =
						std::max(0.0f, fgmask_history.at<float>(i, j) - D_COST * (1 - is_fg));
		}

	// update sfgmask
	for (int i = 0; i < fgmask.rows; i++)
		for (int j = 0; j < fgmask.cols; j++) {
			// ...
			auto val = std::min(1.0f, fgmask_history.at<float>(i, j) / numframes4static);
			if (val >= STAT_TH)
				sfgmask.at<uchar>(i, j) = 255;
			else
				sfgmask.at<uchar>(i, j) = 0;
		}
	return 1;
}

cvBlob& maxBlob(std::vector<cvBlob>& bloblist) {
	auto it = std::max_element(bloblist.begin(), bloblist.end());
	return *it;
}

