#include <opencv2/core/core.hpp>
#include <opencv2/core/internal.hpp>
#include "photoeffects.hpp"

#include <stdio.h>
#define TIMER_START(name) int64 t_##name = getTickCount()
#define TIMER_END(name) printf("TIMER_" #name ":\t%6.2fms\n", \
            1000.f * ((getTickCount() - t_##name) / getTickFrequency()))



using namespace cv;

namespace
{
const int MAX_INTENSITY = 255;

class BoostColorInvoker
{
public:
    BoostColorInvoker(const Mat& src, Mat& dst, unsigned char intensity)
        : src_(src),
          dst_(dst),
          intensity_(intensity),
          height_(src.cols) {}

    void operator()(const BlockedRange& rows) const
    {
        Mat srcStripe = src_.rowRange(rows.begin(), rows.end());

		Mat srcHlsStripe;
        cvtColor(srcStripe, srcHlsStripe, CV_BGR2HLS);
        int stripeWidht = srcHlsStripe.rows;
		
        for (int y = 0; y < stripeWidht; y++)
        {
            unsigned char* row = srcHlsStripe.row(y).data;
            for (int x = 0; x < height_*3; x += 3)
            {
				row[x + 2] = min(row[x + 2] + intensity_, MAX_INTENSITY);
            }
        }
		
        //Mat dstStripe = dst_.rowRange(rows.begin(), rows.end());
		dst_ = srcHlsStripe;
        //cvtColor(srcHlsStripe, dstStripe, CV_HLS2BGR);
    }

private:
    const Mat& src_;
    Mat& dst_;
    unsigned char intensity_;
    const int height_;

    BoostColorInvoker& operator=(const BoostColorInvoker&);
};
}

int boostColor(InputArray src, OutputArray dst, float intensity)
{
    Mat srcImg = src.getMat();

    CV_Assert(srcImg.channels() == 3);
    CV_Assert(intensity >= 0.0f && intensity <= 1.0f);

    if (srcImg.type() != CV_8UC3)
    {
        srcImg.convertTo(srcImg, CV_8UC3);
    }

    dst.create(srcImg.size(), srcImg.type());
    Mat dstMat = dst.getMat();
	TIMER_START(parallel);
    parallel_for(BlockedRange(0, srcImg.rows), BoostColorInvoker(srcImg, dstMat, intensity * MAX_INTENSITY));
	TIMER_END(parallel);

	TIMER_START(toBGR);
	cvtColor(dstMat, dstMat, CV_HLS2BGR);
	TIMER_END(toBGR);
	TIMER_START(toSrcType);
	dstMat.convertTo(dst, srcImg.type());
	TIMER_END(toSrcType);
    return 0;
}
