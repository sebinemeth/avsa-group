/* Applied Video Analysis of Sequences (AVSA)
 *
 *	LAB2: Blob detection & classification
 *	Lab2.0: Sample Opencv project
 *
 *
 * Authors: José M. Martínez (josem.martinez@uam.es), Paula Moral (paula.moral@uam.es), Juan C. San Miguel (juancarlos.sanmiguel@uam.es)
 */

#include "blobs.hpp"

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
 Mat paintBlobImage(cv::Mat frame, std::vector<cvBlob> bloblist, bool labelled)
{
	cv::Mat blobImage;
	//check input conditions and return original if any is not satisfied
	//...
	frame.copyTo(blobImage);

	//required variables to paint
	//...

	//paint each blob of the list
	for(int i = 0; i < bloblist.size(); i++)
	{
		cvBlob blob = bloblist[i]; //get ith blob
		//...
		Scalar color;
		std::string label="";
		switch(blob.label){
		case PERSON:
			color = Scalar(255,0,0);
			label="PERSON";
			break;
		case CAR:
					color = Scalar(0,255,0);
					label="CAR";
					break;
		case OBJECT:
					color = Scalar(0,0,255);
					label="OBJECT";
					break;
		default:
			color = Scalar(255, 255, 255);
			label="UNKOWN";
		}

		Point p1 = Point(blob.x, blob.y);
		Point p2 = Point(blob.x+blob.w, blob.y+blob.h);

		rectangle(blobImage, p1, p2, color, 1, 8, 0);
		if (labelled)
			{
			rectangle(blobImage, p1, p2, color, 1, 8, 0);
			putText(blobImage, label, p1, FONT_HERSHEY_SIMPLEX, 0.5, color);
			}
			else
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
int extractBlobs(cv::Mat fgmask, std::vector<cvBlob> &bloblist, int connectivity)
{	
	//check input conditions and return -1 if any is not satisfied
	//...		

	//required variables for connected component analysis 
	//...
	Mat aux; // image to be updated each time a blob is detected (blob cleared)
	fgmask.convertTo(aux,CV_32SC1);
	
	//clear blob list (to fill with this function)
	bloblist.clear();
			
	//Connected component analysis
		
	
	// void creation of a unqie blob in the center
		cvBlob blob=initBlob(1, fgmask.cols/2, fgmask.rows/2, fgmask.cols/4, fgmask.rows/4);
		bloblist.push_back(blob);

//	std::cout << bkg << " " << fg << " " << sh <<" " << fill << " " << unknown << " "<< bkg+fg+sh+unknown  << " " << fgmask.rows*fgmask.cols << std::endl;
//	std::cout << blob_id << " " << small_blobs << std::endl;

	//destroy all resources
	//...

	//return OK code
	return 1;
}


int removeSmallBlobs(std::vector<cvBlob> bloblist_in, std::vector<cvBlob> &bloblist_out, int min_width, int min_height)
{
	//check input conditions and return -1 if any is not satisfied

	//required variables
	//...

	//clear blob list (to fill with this function)
	bloblist_out.clear();


	for(int i = 0; i < bloblist_in.size(); i++)
	{
		cvBlob blob_in = bloblist_in[i]; //get ith blob

		// ...
		bloblist_out.push_back(blob_in); // void implementation (does not remove)

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
float ED(float val1, float val2)
{
	return sqrt(pow(val1-val2,2));
}

float WED(float val1, float val2, float std)
{
	return sqrt(pow(val1-val2,2)/pow(std,2));
}
//end distances
 int classifyBlobs(std::vector<cvBlob> &bloblist)
 {
 	//check input conditions and return -1 if any is not satisfied
 	//...

 	//required variables for classification
 	//...

 	//classify each blob of the list
 	for(int i = 0; i < bloblist.size(); i++)
 	{
 		cvBlob blob = bloblist[i]; //get i-th blob
 		//...

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
#define SECS_STATIONARY 0 // to set
#define I_COST 0 // to set // increment cost for stationarity detection
#define D_COST 0 // to set // decrement cost for stationarity detection
#define STAT_TH 0.0 // to set

 int extractStationaryFG (Mat fgmask, Mat &fgmask_history, Mat &sfgmask)
 {

	 int numframes4static=(int)(FPS*SECS_STATIONARY);


	 // update fgmask_counter
	 for (int i=0; i<fgmask.rows;i++)
		 for(int j=0; j<fgmask.cols;j++)
		 {
			// ...
			 fgmask_history.at<float>(i,j) = 0; // void implementation (no history)
		 }//for

	// update sfgmask
	for (int i=0; i<fgmask.rows;i++)
		 for(int j=0; j<fgmask.cols;j++)
			 {
			 	 // ...
				 sfgmask.at<uchar>(i,j)=0;// void implementation (no stationary fg)
			 }
 return 1;
 }



