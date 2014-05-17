#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdlib.h>
#include <stdio.h>
#define TIMER_START(name) int64 t_##name = getTickCount()
#define TIMER_END(name) printf("TIMER_" #name ":\t%6.2fms\n", \
            1000.f * ((getTickCount() - t_##name) / getTickFrequency()))

using namespace cv;

int main(int argc, char **argv)
{
	TIMER_START(readImg);
	Mat img = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	TIMER_END(readImg);
	
	int countProc = atoi(argv[2]);
	cv::setNumThreads(countProc);
	
	Mat hlsImg;
	TIMER_START(toHLS);
	cvtColor(img, hlsImg, CV_BGR2HSV);
	TIMER_END(toHLS);

	Mat rgbImg;
	TIMER_START(toBGR);
	cvtColor(hlsImg, rgbImg, CV_HSV2BGR);
	TIMER_END(toBGR);
	//imshow("1", img);
	//imshow("2", rgbImg);
}