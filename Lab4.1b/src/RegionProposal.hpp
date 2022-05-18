/*
 * RegionProposal.hpp
 *
 *  Created on: May 10, 2022
 *      Author: avsa
 */

#ifndef SRC_REGIONPROPOSAL_HPP_
#define SRC_REGIONPROPOSAL_HPP_

#include <vector>
#include <opencv2/opencv.hpp>

class RegionProposal {
public:
	RegionProposal(int, int);
	void generateCandidates(cv::Rect, cv::Size);
	void matchCandidates(cv::Mat, cv::Rect);
	virtual ~RegionProposal();

	const std::vector<float>& getCandidateScores() const { return _candidateScores; }
	const std::vector<cv::Rect>& getCandidates() const { return _candidates; }

	cv::Rect getMax();

private:
	int _count;
	int _step;
	std::vector<cv::Rect> _candidates;
	std::vector<float> _candidateScores;
};

#endif /* SRC_REGIONPROPOSAL_HPP_ */
