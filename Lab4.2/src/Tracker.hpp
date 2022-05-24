/*
 * RegionProposal.hpp
 *
 *  Created on: May 10, 2022
 *      Author: avsa
 */

#ifndef SRC_TRACKER_HPP_
#define SRC_TRACKER_HPP_

#include <vector>
#include <opencv2/opencv.hpp>

enum ColorFeature {
	GRAY,
	HUE,
	SAT,
	RED,
	GREEN,
	BLUE
};

class Tracker {
public:
	Tracker(cv::Size frameSize, int count, int step, int n_bins,
			bool col_features, ColorFeature colorFeature);
	void updateModelHist(cv::Mat ROI);
	void generateCandidates();
	cv::Mat convert(cv::Mat &src);
	cv::Mat createHistogram(cv::Mat ROI);
	cv::Mat visualizeHistogram(cv::Mat hist);
	void matchCandidates(cv::Mat&);
	void selectMinMatch();
	virtual ~Tracker();

	const std::vector<float>& getCandidateScores() const {
		return _candidateScores;
	}
	const std::vector<cv::Rect>& getCandidates() const {
		return _candidates;
	}

	const cv::Rect& getBbox() const {
		return _bbox;
	}
	void setBbox(const cv::Rect &bbox) {
		_bbox = bbox;
	}

	bool isColFeatures() const {
		return _col_features;
	}

	ColorFeature getColorFeature() const {
		return _colorFeature;
	}

	int getCount() const {
		return _count;
	}

	int getBins() const {
		return _n_bins;
	}

	int getStep() const {
		return _step;
	}

private:
	cv::Size _frameSize;
	int _count;
	int _step;
	int _n_bins;
	bool _col_features;
	ColorFeature _colorFeature;
	std::vector<cv::Rect> _candidates;
	std::vector<float> _candidateScores;
	cv::Mat _modelHist;
	cv::Rect _bbox;
};

#endif /* SRC_TRACKER_HPP_ */
