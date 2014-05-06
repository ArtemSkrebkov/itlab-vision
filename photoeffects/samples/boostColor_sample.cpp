#include "photoeffects.hpp"
#include <stdlib.h>
#include <iostream>

#include <stdio.h>
#define TIMER_START(name) int64 t_##name = getTickCount()
#define TIMER_END(name) printf("TIMER_" #name ":\t%6.2fms\n", \
            1000.f * ((getTickCount() - t_##name) / getTickFrequency()))


using namespace cv;
using namespace std;

const char *helper =
"./boostColor_sample <img> <intensity>\n\
\t<img> - file name contained the source image, must be 3-channel, RGB-image\n\
\t<intensity> - intensity of boost color filter, must be real number from 0.0 to 1.0\n\
";

int processArguments(int argc, char **argv, Mat &img, float &intensity);

int main(int argc, char **argv)
{
    const char *srcImgWinName = "Initial image", *dstImgWinName = "Processed image";
    Mat img, dstImg;
    float intensity;
    if (processArguments(argc, argv, img, intensity) != 0)
    {
        cout << helper << endl;
        return 1;
    }
	cv::setNumThreads(4);
	TIMER_START(ALL);
    boostColor(img, dstImg, intensity);
	TIMER_END(ALL);
    namedWindow(srcImgWinName);
    namedWindow(dstImgWinName);
    imshow(srcImgWinName, img);
    imshow(dstImgWinName, dstImg);
    waitKey();
    destroyAllWindows();

    return 0;
}

int processArguments(int argc, char **argv, Mat &img, float &intensity)
{
    if (argc < 3)
    {
        return 1;
    }
    img = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    intensity = (float)atof(argv[2]);
    return 0;
}