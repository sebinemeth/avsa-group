/*
 * RegionProposal.cpp
 *
 *  Created on: May 10, 2022
 *      Author: avsa
 */

#include "Tracker.hpp"

Tracker::Tracker(cv::Size frameSize, int count, int step, int n_bins,
		bool col_features, ColorFeature colorFeature) :
		_frameSize(frameSize), _count(count), _step(step), _n_bins(n_bins), _col_features(
				col_features), _colorFeature(colorFeature) {

}

void Tracker::updateModelHist(cv::Mat ROI) {
	_modelHist = createHistogram(ROI);
}

void Tracker::generateCandidates() {
	_candidates.clear();
	for (int i = -_count; i <= _count; i++) {
		for (int j = -_count; j <= _count; j++) {
			int x = _bbox.x + i * _step, y = _bbox.y + j * _step, w =
					_bbox.width, h = _bbox.height;
			if (x >= 0 && x + w < _frameSize.width && y >= 0
					&& y + h < _frameSize.height)
				_candidates.push_back(cv::Rect(x, y, w, h));
		}
	}
	return;
}

cv::Mat Tracker::convert(cv::Mat &src) {
	cv::Mat output;
	if (_colorFeature == GRAY)
		cvtColor(src, output, cv::COLOR_RGB2GRAY);
	else if (_colorFeature == HUE or _colorFeature == SAT) {
		cvtColor(src, output, cv::COLOR_RGB2HSV);
		std::vector<cv::Mat> channels;
		cv::split(output, channels);
		if (_colorFeature == HUE)
			output = channels[0];
		if (_colorFeature == SAT)
			output = channels[1];

	} else if (_colorFeature == RED or _colorFeature == GREEN
			or _colorFeature == BLUE) {
		std::vector<cv::Mat> channels;
		cv::split(src, channels);
		if (_colorFeature == RED)
			output = channels[0];
		if (_colorFeature == GREEN)
			output = channels[1];
		if (_colorFeature == BLUE)
			output = channels[2];
	}
	return output;
}

cv::Mat Tracker::createHistogram(cv::Mat ROI) {
	if (_col_features) {
		cv::Mat hist;
		float range[] = { 0, 256 }; //the upper boundary is exclusive
		const float *histRange = { range };
		bool uniform = true, accumulate = false;
		cv::calcHist(&ROI, 1, 0, cv::Mat(), hist, 1, &_n_bins, &histRange,
				uniform, accumulate);

		return hist / cv::sum(hist);
	} else {
		cv::HOGDescriptor hog;
		cv::Mat ResizedCroppedOutput;
		cv::resize(ROI, ResizedCroppedOutput, cv::Size(64, 128));
		hog.nbins = _n_bins;
		std::vector<float> descriptors;
		hog.compute(ResizedCroppedOutput, descriptors);
		cv::Mat hist = cv::Mat(descriptors).clone();

		return hist / cv::sum(hist);
	}
}

cv::Mat Tracker::visualizeHistogram(cv::Mat hist) {
	int hist_w = 512, hist_h = 400;
	int bin_w = cvRound((double) hist_w / _n_bins);
	cv::Mat histImage = cv::Mat::zeros(hist_h, hist_w, CV_8UC1);
	cv::normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX, -1,
			cv::Mat());

	if (_col_features) {
		for (int i = 1; i < _n_bins; i++) {
			cv::line(histImage,
					cv::Point(bin_w * (i - 1),
							hist_h - cvRound(hist.at<float>(i - 1))),
					cv::Point(bin_w * (i), hist_h - cvRound(hist.at<float>(i))),
					cv::Scalar(255, 0, 0), 2, 8, 0);
		}
	}
	return histImage;
}

void Tracker::matchCandidates(cv::Mat &image) {
	_candidateScores.clear();
	for (size_t i = 0; i < _candidates.size(); ++i) {
		cv::Mat candHist = createHistogram(image(_candidates[i]));
		// float score = candHist.dot(_modelHist) / (cv::norm(candHist) * cv::norm(_modelHist));
		float score = cv::compareHist(candHist, _modelHist,
				cv::HISTCMP_BHATTACHARYYA);
		_candidateScores.push_back(score);
	}
	return;
}

void Tracker::selectMinMatch() {
	size_t argmin = 0;
	for (size_t i = 0; i < _candidates.size(); i++)
		if (_candidateScores[i] < _candidateScores[argmin])
			argmin = i;
	_bbox = _candidates[argmin];
}

Tracker::~Tracker() {
	// TODO Auto-generated destructor stub
}

