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

bgs::bgs(double threshold, double alpha, double selective_bkg_update,
		int threshold_ghosts2, double alpha_sh, double beta_sh, int sat_th,
		int hue_th, bool unimodel, int init_count, double alpha_g,
		double gauss_th, bool rgb) :
		_threshold(threshold), _alpha(alpha), _selective_bkg_update(
				selective_bkg_update), _threshold_ghosts2(threshold_ghosts2), _alpha_sh(
				alpha_sh), _beta_sh(beta_sh), _sat_th(sat_th), _hue_th(hue_th), _unimodel(
				unimodel), _init_count(init_count), _alpha_g(alpha_g), _gauss_th(
				gauss_th), _rgb(rgb) {
}

bgs::bgs(double threshold, double alpha, double selective_bkg_update,
		int threshold_ghosts2, double alpha_sh, double beta_sh, int sat_th,
		int hue_th, bool unimodel, int init_count, double alpha_g,
		double gauss_th, int K, double w_th, double initial_var, bool rgb) :
		_threshold(threshold), _alpha(alpha), _selective_bkg_update(
				selective_bkg_update), _threshold_ghosts2(threshold_ghosts2), _alpha_sh(
				alpha_sh), _beta_sh(beta_sh), _sat_th(sat_th), _hue_th(hue_th), _unimodel(
				unimodel), _init_count(init_count), _alpha_g(alpha_g), _gauss_th(
				gauss_th), _K(K), _w_th(w_th), _initial_var(initial_var), _rgb(rgb) {
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
		cv::absdiff(Frame, _bkg, _diff);
		_fgcounter = Mat::zeros(Size(Frame.cols, Frame.rows), CV_16UC1);

		_var = Mat::zeros(Size(Frame.cols, Frame.rows), CV_64F);
		_mean = Mat::zeros(Size(Frame.cols, Frame.rows), CV_64F);
		_init_sum = Mat::zeros(Size(Frame.cols, Frame.rows), CV_64F);
		_init_sum_sq = Mat::zeros(Size(Frame.cols, Frame.rows), CV_64F);

		_w_mm = vector<Mat>(_K,
				Mat::zeros(Size(Frame.cols, Frame.rows), CV_64F));
		_mean_mm = vector<Mat>(_K,
				Mat::zeros(Size(Frame.cols, Frame.rows), CV_64F));
		_var_mm = vector<Mat>(_K,
				Mat::zeros(Size(Frame.cols, Frame.rows), CV_64F));

		//...
	} else {
		_bkg = Frame.clone();
		cv::absdiff(Frame, _bkg, _diff);
		_fgcounter = Mat::zeros(Size(Frame.cols, Frame.rows), CV_16UC1);

		_var = Mat::zeros(Size(Frame.cols, Frame.rows), CV_64FC3);
		_mean = Mat::zeros(Size(Frame.cols, Frame.rows), CV_64FC3);
		_init_sum = Mat::zeros(Size(Frame.cols, Frame.rows), CV_64FC3);
		_init_sum_sq = Mat::zeros(Size(Frame.cols, Frame.rows), CV_64FC3);
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

	//ADD YOUR CODE HERE
	//...

	if (!_rgb) {
		_bgsmask.copyTo(_shadowmask);

		absdiff(_bgsmask, _bgsmask, _shadowmask);
	} else {
		_shadowmask = Mat::zeros(Size(_frame.cols, _frame.rows), CV_8UC1);

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
						&& (fv / (double) bv) <= _beta_sh
						&& abs(fs - bs) <= _sat_th && dh <= _hue_th
						&& _bgsmask.at<uchar>(i, j) == 255)
					_shadowmask.at<uchar>(i, j) = 255;
			}
		}
	}
	//...

	absdiff(_bgsmask, _shadowmask, _fgmask); // eliminates shadows from bgsmask
}

//ADD ADDITIONAL FUNCTIONS HERE
void bgs::gaussian(cv::Mat Frame, int it) {
	_bgsmask = Mat::zeros(Size(Frame.cols, Frame.rows), CV_8UC1);

	if (!_rgb) {
		cvtColor(Frame, Frame, COLOR_BGR2GRAY); // to work with gray even if input is color
		Frame.copyTo(_frame);
		_frame.convertTo(_frame, CV_64F);
		if (_unimodel) {
			if (it <= _init_count) {
				// Initialization
				_init_sum += _frame;
				_init_sum_sq += _frame.mul(_frame);

				_mean = _init_sum / (double) _init_count;
				_var = (_init_sum_sq / (double) _init_count) - _mean.mul(_mean);
			} else {
				for (int i = 0; i < _frame.rows; ++i) {
					for (int j = 0; j < _frame.cols; ++j) {
						auto im = _frame.at<double>(i, j);
						auto m = _mean.at<double>(i, j);
						auto s2 = _var.at<double>(i, j);
						auto d = _frame.at<double>(i, j)
								- _mean.at<double>(i, j);

						if (fabs(d) < _gauss_th * sqrt(s2)) {
							_bgsmask.at<uchar>(i, j) = 0;
							m = im * _alpha_g + m * (1.0 - _alpha_g);
							s2 = pow(d, 2) * _alpha_g + s2 * (1 - _alpha_g);
						} else {
							_bgsmask.at<uchar>(i, j) = 255;
						}
					}
				}
			}
		} else {
			if (it == 1) {
				_mean_mm[0] = _frame;
				_var_mm[0] = Mat::ones(Size(Frame.cols, Frame.rows), CV_64F) * _initial_var;
				_w_mm[0] = Mat::ones(Size(Frame.cols, Frame.rows), CV_64F);
			}
			_M_mm = vector<Mat>(_K,
					Mat::zeros(Size(Frame.cols, Frame.rows), CV_8UC1));
			_w_mm_tmp = vector<Mat>(_K,
					Mat::zeros(Size(Frame.cols, Frame.rows), CV_64FC1));

			for (int k = 0; k < _K; ++k) {
				for (int i = 0; i < _frame.rows; ++i) {
					for (int j = 0; j < _frame.cols; ++j) {
						auto im = _frame.at<double>(i, j);
						auto m = _mean_mm[k].at<double>(i, j);
						auto s2 = _var_mm[k].at<double>(i, j);
						auto d = _frame.at<double>(i, j)
								- _mean_mm[k].at<double>(i, j);

						if (fabs(d) < _gauss_th * sqrt(s2)) {
							_M_mm[k].at<uchar>(i, j) = 1;
							m = im * _alpha_g + m * (1.0 - _alpha_g);
							s2 = pow(d, 2) * _alpha_g + s2 * (1 - _alpha_g);
						}
						_w_mm_tmp[k].at<double>(i, j) = _w_mm[k].at<double>(i,
								j) * (1 - _alpha_g)
								+ ((int) _M_mm[k].at<uchar>(i, j)) * _alpha_g;
					}
				}
			}

			for (int i = 0; i < _frame.rows; ++i) {
				for (int j = 0; j < _frame.cols; ++j) {
					double sum = 0.0;
					cout << _w_mm_tmp[0].at<double>(i, j) << " "
							<< _w_mm_tmp[1].at<double>(i, j) << " "
							<< _w_mm_tmp[2].at<double>(i, j) << " " << sum
							<< endl;
					for (int k = 0; k < _K; ++k)
						sum += _w_mm_tmp[k].at<double>(i, j);
					for (int k = 0; k < _K; ++k)
						_w_mm_tmp[k].at<double>(i, j) /= sum;
					cout << _w_mm_tmp[0].at<double>(i, j) << " "
							<< _w_mm_tmp[1].at<double>(i, j) << " "
							<< _w_mm_tmp[2].at<double>(i, j) << " " << sum
							<< endl;
				}
			}

			for (int i = 0; i < _frame.rows; ++i) {
				for (int j = 0; j < _frame.cols; ++j) {
					int min_pos = -1;
					double min_val = 1.0;

					for (int k = 0; k < _K; ++k) {
						if (_M_mm[k].at<uchar>(i, j) == 1
								&& _w_mm_tmp[k].at<double>(i, j)
										>= (1 - _w_th)) {
							_bgsmask.at<uchar>(i, j) = 0;
							break;
						}
						_bgsmask.at<uchar>(i, j) = 255;
						if (_w_mm_tmp[k].at<double>(i, j) < min_val) {
							min_pos = k;
							min_val = _w_mm_tmp[k].at<double>(i, j);
						}
					}
					if (_bgsmask.at<uchar>(i, j) == 255) {
						bool has_match = false;
						for (int k = 0; k < _K; ++k)
							if (_M_mm[k].at<uchar>(i, j) == 1)
								has_match = true;
						if (!has_match) {
							_w_mm_tmp[min_pos].at<double>(i, j) = 0.05;
							_mean_mm[min_pos].at<double>(i, j) = _frame.at<
									double>(i, j);
							_var_mm[min_pos].at<double>(i, j) = _initial_var;
						}

						double sum = 0.0;
						for (int k = 0; k < _K; ++k)
							sum += _w_mm_tmp[k].at<double>(i, j);
						for (int k = 0; k < _K; ++k)
							_w_mm_tmp[k].at<double>(i, j) /= sum;
					}
				}
			}

			for (int k = 0; k < _K; ++k)
				_w_mm[k] = _w_mm_tmp[k];
		}
		_frame.convertTo(_frame, CV_8UC1);
	} else {
		if (_unimodel) {
			Frame.copyTo(_frame);
			_frame.convertTo(_frame, CV_64FC3);
			if (it <= _init_count) {
				// Initialization
				_init_sum += _frame;
				_init_sum_sq += _frame.mul(_frame);

				_mean = _init_sum / (double) _init_count;
				_var = (_init_sum_sq / (double) _init_count) - _mean.mul(_mean);
			} else {
				for (int i = 0; i < _frame.rows; ++i) {
					for (int j = 0; j < _frame.cols; ++j) {
						auto im = _frame.at<Vec3d>(i, j);
						auto m = _mean.at<Vec3d>(i, j);
						auto s2 = _var.at<Vec3d>(i, j);
						auto d = _frame.at<Vec3d>(i, j) - _mean.at<Vec3d>(i, j);

						bool bg = true;
						for (int c = 0; c < 3; ++c) {
							if (fabs(d[c]) < _gauss_th * sqrt(s2[c])) {
								m[c] = im[c] * _alpha_g
										+ m[c] * (1.0 - _alpha_g);
								s2[c] = pow(d[c], 2) * _alpha_g
										+ s2[c] * (1 - _alpha_g);
							} else {
								bg = false;
							}
						}
						if (bg)
							_bgsmask.at<uchar>(i, j) = 0;

						else
							_bgsmask.at<uchar>(i, j) = 255;

					}
				}

			}
		} else {
			cout << "not unimodel" << endl;
			exit(1);
		}
		_frame.convertTo(_frame, CV_8UC3);
	}
}
