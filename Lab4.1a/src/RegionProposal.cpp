/*
 * RegionProposal.cpp
 *
 *  Created on: May 10, 2022
 *      Author: avsa
 */

#include "RegionProposal.hpp"

RegionProposal::RegionProposal(int count, int step) :
		_count(count), _step(step) {

}

void RegionProposal::generateCandidates(cv::Rect_<int> box,
		cv::Size img_size) {
	_candidates.clear();
	for (int i = -_count; i <= _count; i++)
		for (int j = -_count; j <= _count; j++) {
			int x = box.x + i * _step, y = box.y + j * _step, w = box.width, h =
					box.height;
			_candidates.push_back(
					cv::Rect(std::max(0, std::min(img_size.width - 1, x)),
							std::max(0, std::min(img_size.height - 1, y)),
							std::max(0, std::min(img_size.width - 1 - x, w)),
							std::max(0, std::min(img_size.height - 1 - y, h))));
		}
	return;
}

void RegionProposal::matchCandidates(cv::Mat image, cv::Rect templ) {
	_candidateScores.clear();
	cv::cvtColor(image, image, cv::COLOR_RGB2GRAY);
	image.convertTo(image, CV_32FC1);

	for (size_t i = 0; i < _candidates.size(); ++i) {
		cv::Mat score;
		cv::Mat template_patch = image(templ);
		cv::Mat candidate_patch = image(_candidates[i]);
		score.create(1, 1, CV_32FC1);
		cv::matchTemplate(candidate_patch, template_patch, score, CV_TM_CCOEFF_NORMED);
		_candidateScores.push_back(score.at<float>(0));
	}
	return;
}

cv::Rect RegionProposal::getMax() {
	size_t argmax = 0;
	for (size_t i=0; i<_candidates.size(); i++)
		if (_candidateScores[i] > _candidateScores[argmax])
			argmax = i;
	return _candidates[argmax];
}


RegionProposal::~RegionProposal() {
	// TODO Auto-generated destructor stub
}

