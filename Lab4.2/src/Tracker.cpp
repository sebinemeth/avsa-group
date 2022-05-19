/*
 * RegionProposal.cpp
 *
 *  Created on: May 10, 2022
 *      Author: avsa
 */

#include "Tracker.hpp"

Tracker::Tracker(cv::Size frameSize, int count, int step, int n_bins, bool col_features, cv::String color) : _frameSize(frameSize),
		_count(count), _step(step), _n_bins(n_bins), _col_features(col_features), _color(color) {

}

void Tracker::updateModelHist(cv::Mat ROI) {
	_modelHist = createHistogram(ROI);
}

void Tracker::generateCandidates() {
	_candidates.clear();
	for (int i = -_count; i <= _count; i++)
		for (int j = -_count; j <= _count; j++) {
			int x = _bbox.x + i * _step, y = _bbox.y + j * _step, w = _bbox.width, h =
					_bbox.height;
			_candidates.push_back(
					cv::Rect(std::max(0, std::min(_frameSize.width - 1, x)),
							std::max(0, std::min(_frameSize.height - 1, y)),
							std::max(0, std::min(_frameSize.width - 1 - x, w)),
							std::max(0, std::min(_frameSize.height - 1 - y, h))));
		}
	return;
}

cv::Mat Tracker::convert(cv::Mat &src) {
	cv::Mat output;
	if (_color == "gr")
		cvtColor(src, output, cv::COLOR_RGB2GRAY);
	else if (_color == "h" or _color == "s") {
		cvtColor(src, output, cv::COLOR_RGB2HSV);
		std::vector<cv::Mat> channels;
		cv::split(output, channels);
		if (_color == "h")
			output = channels[0];
		if (_color == "s")
			output = channels[1];

	} else if (_color == "r" or _color == "g" or _color == "b") {
		std::vector<cv::Mat> channels;
		cv::split(src, channels);
		if (_color == "r")
			output = channels[0];
		if (_color == "g")
			output = channels[1];
		if (_color == "b")
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

		return hist;
	} else {
		cv::HOGDescriptor hog;
		cv::Mat ResizedCroppedOutput;
		cv::resize(ROI, ResizedCroppedOutput, cv::Size(64, 128));
		hog.nbins = _n_bins;
		std::vector<float> descriptors;
		hog.compute(ResizedCroppedOutput, descriptors);
		cv::Mat hist = cv::Mat(descriptors).clone();

		return hist;
	}
}

cv::Mat Tracker::visualizeHistogram(cv::Mat hist)
{
	int hist_w = 512, hist_h = 400;
	int bin_w = cvRound( (double) hist_w/_n_bins);
	cv::Mat histImage = cv::Mat::zeros( hist_h, hist_w, CV_8UC1);
	cv::normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );

	for( int i = 1; i < _n_bins; i++ )
		{
		cv::line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1)) ),
				cv::Point( bin_w*(i), hist_h - cvRound(hist.at<float>(i)) ),
				cv::Scalar( 255, 0, 0), 2, 8, 0  );
		}
	return histImage;
}


void Tracker::matchCandidates(cv::Mat& image) {
	_candidateScores.clear();

	for (size_t i = 0; i < _candidates.size(); ++i) {
		cv::Mat candHist = createHistogram(image(_candidates[i]));
		// match candHist and _modelHist
		// store in _candidateScores
	}
	return;
}

void Tracker::selectMinMatch() {
	_bbox = _candidates[0];
	// select minimal score from _candidateScores
	// store corresponding bbox in _bbox
}

Tracker::~Tracker() {
	// TODO Auto-generated destructor stub
}

