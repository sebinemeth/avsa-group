/* Applied Video Sequence Analysis (AVSA)
 *
 *	LAB1.0: Background Subtraction - Unix version
 *	fgesg.cpp
 *
 * 	Authors: José M. Martínez (josem.martinez@uam.es), Paula Moral (paula.moral@uam.es) & Juan Carlos San Miguel (juancarlos.sanmiguel@uam.es)
 *	VPULab-UAM 2020
 */

#include <opencv2/opencv.hpp>
#include "fgseg.hpp"
#include "stdlib.h"

using namespace fgseg;

//default constructor
bgs::bgs(double threshold, bool rgb) :
		_threshold(threshold), _alpha(0.0), _selective_bkg_update(false), _threshold_ghosts2(
				0), _rgb(rgb) {
}

bgs::bgs(double threshold, double alpha, double selective_bkg_update, bool rgb) :
		_threshold(threshold), _alpha(alpha), _selective_bkg_update(
				selective_bkg_update), _threshold_ghosts2(0), _rgb(rgb) {
}

bgs::bgs(double threshold, double alpha, double selective_bkg_update,
		int threshold_ghosts2, bool rgb) :
		_threshold(threshold), _alpha(alpha), _selective_bkg_update(
				selective_bkg_update), _threshold_ghosts2(threshold_ghosts2), _rgb(
				rgb) {
}

bgs::bgs(double threshold, double alpha, double selective_bkg_update,
		int threshold_ghosts2, double alpha_sh, double beta_sh, int sat_th,
		int hue_th) :
		_threshold(threshold), _alpha(alpha), _selective_bkg_update(
				selective_bkg_update), _threshold_ghosts2(threshold_ghosts2), _rgb(
				true), _alpha_sh(alpha_sh), _beta_sh(beta_sh), _sat_th(sat_th), _hue_th(
				hue_th) {
}

//default destructor
bgs::~bgs(void) {
}

//method to initialize bkg (first frame - hot start)
void bgs::init_bkg(cv::Mat Frame) {

	if (!_rgb) {
		cvtColor(Frame, Frame, COLOR_BGR2GRAY); // to work with gray even if input is color

		_bkg = Mat::zeros(Size(Frame.cols, Frame.rows), CV_8UC1); // void function for Lab1.0 - returns zero matrix
		//ADD YOUR CODE HERE
		//...
		_bkg = Frame.clone();
		_fgcounter = Mat::zeros(Size(Frame.cols, Frame.rows), CV_16UC1);
		//...
	} else {
		_bkg = Frame.clone();
		_fgcounter = Mat::zeros(Size(Frame.cols, Frame.rows), CV_16UC1);
	}

}

//method to perform BackGroundSubtraction
void bgs::bkgSubtraction(cv::Mat Frame) {

	if (!_rgb) {
		cvtColor(Frame, Frame, COLOR_BGR2GRAY); // to work with gray even if input is color
		Frame.copyTo(_frame);

		_diff = Mat::zeros(Size(Frame.cols, Frame.rows), CV_8UC1); // void function for Lab1.0 - returns zero matrix
		_bgsmask = Mat::zeros(Size(Frame.cols, Frame.rows), CV_8UC1); // void function for Lab1.0 - returns zero matrix
		//ADD YOUR CODE HERE
		//...
		cv::absdiff(_frame, _bkg, _diff);
		cv::threshold(_diff, _bgsmask, _threshold, 255, THRESH_BINARY);
		if (_selective_bkg_update) {
			for (int i = 0; i < _frame.rows; ++i) {
				for (int j = 0; j < _frame.cols; ++j) {
					if (_bgsmask.at<uchar>(i, j) == 0) {
						_bkg.at<uchar>(i, j) = _frame.at<uchar>(i, j) * _alpha
								+ _bkg.at<uchar>(i, j) * (1 - _alpha);
					}
				}
			}
		} else {
			cv::add(_frame * _alpha, _bkg * (1.0 - _alpha), _bkg);
		}

		if (_threshold_ghosts2 > 0) {
			for (int i = 0; i < _frame.rows; ++i) {
				for (int j = 0; j < _frame.cols; ++j) {
					int m = _bgsmask.at<uchar>(i, j) == 255;

					_fgcounter.at<ushort>(i, j) += m;
					_fgcounter.at<ushort>(i, j) *= m;

					if (_fgcounter.at<ushort>(i, j) >= _threshold_ghosts2) {
						_bkg.at<uchar>(i, j) = _frame.at<uchar>(i, j);
						_fgcounter.at<ushort>(i, j) = 0;
					}
				}
			}
		}

		//...
	} else {
		Frame.copyTo(_frame);
		_diff = Mat::zeros(Size(Frame.cols, Frame.rows), CV_8UC3);
		_bgsmask = Mat::zeros(Size(Frame.cols, Frame.rows), CV_8UC1);

		_rgb_bgsmask = Mat::zeros(Size(Frame.cols, Frame.rows), CV_8UC3);

		cv::absdiff(_frame, _bkg, _diff);
		cv::threshold(_diff, _rgb_bgsmask, _threshold, 255, THRESH_BINARY);

		for (int i = 0; i < _frame.rows; ++i) {
			for (int j = 0; j < _frame.cols; ++j) {
				Vec3b maskPixel = _rgb_bgsmask.at<Vec3b>(i, j);
				if (maskPixel[0] || maskPixel[1] || maskPixel[2]) {
					_bgsmask.at<uchar>(i, j) = 255;
				}
			}
		}

		if (_selective_bkg_update) {
			for (int i = 0; i < _frame.rows; ++i) {
				for (int j = 0; j < _frame.cols; ++j) {
					if (_bgsmask.at<uchar>(i, j) == 0) {
						_bkg.at<Vec3b>(i, j) = _frame.at<Vec3b>(i, j) * _alpha
								+ _bkg.at<Vec3b>(i, j) * (1 - _alpha);
					}
				}
			}
		} else {
			cv::add(_frame * _alpha, _bkg * (1.0 - _alpha), _bkg);
		}

		if (_threshold_ghosts2 > 0) {
			for (int i = 0; i < _frame.rows; ++i) {
				for (int j = 0; j < _frame.cols; ++j) {
					int m = _bgsmask.at<uchar>(i, j) == 255;

					_fgcounter.at<ushort>(i, j) += m;
					_fgcounter.at<ushort>(i, j) *= m;

					if (_fgcounter.at<ushort>(i, j) >= _threshold_ghosts2) {
						_bkg.at<Vec3b>(i, j) = _frame.at<Vec3b>(i, j);
						_fgcounter.at<ushort>(i, j) = 0;
					}
				}
			}
		}
	}

}

//method to detect and remove shadows in the BGS mask to create FG mask
void bgs::removeShadows() {
	// init Shadow Mask (currently Shadow Detection not implemented)
	_shadowmask = Mat::zeros(Size(_frame.cols, _frame.rows), CV_8UC1);

	//ADD YOUR CODE HERE
	//...

	if (!_rgb) {
		cout << "not rgb" << endl;
		exit(1);
	}

	cvtColor(_frame, _frame_hsv, COLOR_BGR2HSV);
	cvtColor(_bkg, _bkg_hsv, COLOR_BGR2HSV);

	for (int i = 0; i < _frame.rows; ++i) {
		for (int j = 0; j < _frame.cols; ++j) {
			auto p_bkg = _bkg_hsv.at<Vec3b>(i, j);
			auto p_frame = _frame_hsv.at<Vec3b>(i, j);

			auto bh = p_bkg[0], bs = p_bkg[1], bv = p_bkg[2];
			auto fh = p_frame[0], fs = p_frame[1], fv = p_frame[2];

			auto dh = min(abs(fh - bh), 360 - abs(fh - bh));

			if (bv != 0 && (fv / (double) bv) >= _alpha_sh
					&& (fv / (double) bv) <= _beta_sh && abs(fs - bs) <= _sat_th
					&& dh <= _hue_th
					&& _bgsmask.at<uchar>(i, j) == 255)
				_shadowmask.at<uchar>(i, j) = 255;
		}
	}
	//...

	absdiff(_bgsmask, _shadowmask, _fgmask); // eliminates shadows from bgsmask
}

//ADD ADDITIONAL FUNCTIONS HERE

