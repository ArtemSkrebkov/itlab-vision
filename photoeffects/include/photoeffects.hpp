#pragma once

#include <opencv2/core/core.hpp>

int sepia(cv::InputArray src, cv::OutputArray dst);

int filmGrain(cv::InputArray src, cv::OutputArray dst);

int fadeColor(cv::Mat& image,cv::Mat& result);

cv::Mat tint(cv::Mat& image, int hue, float sat);

int glow(cv::Mat& image);

int boost_color(cv::Mat& image);

cv::Mat antique(cv::Mat& image);
