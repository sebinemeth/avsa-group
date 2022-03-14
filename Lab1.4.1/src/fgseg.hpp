/* Applied Video Sequence Analysis (AVSA)
 *
 *	LAB1.0: Background Subtraction - Unix version
 *	fgesg.hpp
 *
 * 	Authors: José M. Martínez (josem.martinez@uam.es), Paula Moral (paula.moral@uam.es) & Juan Carlos San Miguel (juancarlos.sanmiguel@uam.es)
 *	VPULab-UAM 2020
 */


#include <opencv2/opencv.hpp>

#ifndef FGSEG_H_INCLUDE
#define FGSEG_H_INCLUDE

using namespace cv;
using namespace std;

namespace fgseg {


	//Declaration of FGSeg class based on BackGround Subtraction (bgs)
	class bgs{
	public:

		//constructor with parameter "threshold"
		bgs(double threshold, bool rgb);

		bgs(double threshold, double alpha, double selective_bkg_update, bool rgb);

		bgs(double threshold, double alpha, double selective_bkg_update, int threshold_ghosts2, bool rgb);

		bgs(double threshold, double alpha, double selective_bkg_update, int threshold_ghosts2, double alpha_sh, double beta_sh, int sat_th, int hue_th);

		bgs(double threshold, double alpha, double selective_bkg_update, int threshold_ghosts2, double alpha_sh, double beta_sh, int sat_th, int hue_th, bool unimodel, int init_count, double alpha_g, double gauss_th, bool rgb);

		//destructor
		~bgs(void);

		//method to initialize bkg (first frame - hot start)
		void init_bkg(cv::Mat Frame);

		//method to perform BackGroundSubtraction
		void bkgSubtraction(cv::Mat Frame);

		//method to perform BackGroundSubtraction
		void gaussian(cv::Mat Frame, int it);

		//method to detect and remove shadows in the binary BGS mask
		void removeShadows();

		//returns the BG image
		cv::Mat getBG(){return _bkg;};

		//returns the DIFF image
		cv::Mat getDiff(){return _diff;};

		//returns the BGS mask
		cv::Mat getBGSmask(){return _bgsmask;};

		//returns the binary mask with detected shadows
		cv::Mat getShadowMask(){return _shadowmask;};

		//returns the binary FG mask
		cv::Mat getFGmask(){return _fgmask;};


		//ADD ADITIONAL METHODS HERE
		//...
	private:
		cv::Mat _bkg; //Background model
		cv::Mat	_frame; //current frame
		cv::Mat _diff; //abs diff frame
		cv::Mat _bgsmask; //binary image for bgssub (FG)
		cv::Mat _shadowmask; //binary mask for detected shadows
		cv::Mat _fgmask; //binary image for foreground (FG)
		cv::Mat _fgcounter;

		bool _rgb;

		double _threshold;
		//ADD ADITIONAL VARIABLES HERE
		double _alpha;
		bool _selective_bkg_update;
		int _threshold_ghosts2;
		cv::Mat _rgb_bgsmask;

		double _alpha_sh;
		double _beta_sh;
		int _sat_th;
		int _hue_th;

		cv::Mat _frame_hsv;
		cv::Mat _bkg_hsv;

		cv::Mat _mean;
		cv::Mat _var;
		bool _unimodel;

		int _init_count;
		cv::Mat _init_sum;
		cv::Mat _init_sum_sq;

		double _alpha_g;
		double _gauss_th;
		//...

	};//end of class bgs

}//end of namespace

#endif




