#pragma once

#ifndef __RETINAFACE_NCNN_H__
#define __RETINAFACE_NCNN_H__

#include "net.h"

#include "stdio.h"
#include "opencv.h"
#include "platform.h"

#if NCNN_VULKAN
#include "gpu.h"
#endif // NCNN_VULKAN

//#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <time.h>
#include <algorithm>
#include <map>
#include <iostream>
#include <math.h>

//using namespace cv;
using namespace std;


//struct Bbox
//{
//    float score;
//    int x1;
//    int y1;
//    int x2;
//    int y2;
//    float area;
//    float ppoint[10];
//    float regreCoord[4];
//};

template<typename _Tp>
struct Rect_
{
    Rect_() : x(0), y(0), width(0), height(0) {}
    Rect_(_Tp _x, _Tp _y, _Tp _w, _Tp _h) : x(_x), y(_y), width(_w), height(_h) {}

    _Tp x;
    _Tp y;
    _Tp width;
    _Tp height;

    // area
    _Tp area() const
    {
        return width * height;
    }
};
template<typename _Tp> static inline Rect_<_Tp>& operator &= ( Rect_<_Tp>& a, const Rect_<_Tp>& b )
{
    _Tp x1 = std::max(a.x, b.x), y1 = std::max(a.y, b.y);
    a.width = std::min(a.x + a.width, b.x + b.width) - x1;
    a.height = std::min(a.y + a.height, b.y + b.height) - y1;
    a.x = x1; a.y = y1;
    if( a.width <= 0 || a.height <= 0 )
        a = Rect_<_Tp>();
    return a;
}
template<typename _Tp> static inline Rect_<_Tp>& operator |= ( Rect_<_Tp>& a, const Rect_<_Tp>& b )
{
    _Tp x1 = std::min(a.x, b.x), y1 = std::min(a.y, b.y);
    a.width = std::max(a.x + a.width, b.x + b.width) - x1;
    a.height = std::max(a.y + a.height, b.y + b.height) - y1;
    a.x = x1; a.y = y1;
    return a;
}
template<typename _Tp> static inline Rect_<_Tp> operator & (const Rect_<_Tp>& a, const Rect_<_Tp>& b)
{
    Rect_<_Tp> c = a;
    return c &= b;
}

typedef Rect_<int> Rect;
typedef Rect_<float> Rect2f;


template<typename _Tp>
struct Point_
{
    Point_() : x(0), y(0) {}
    Point_(_Tp _x, _Tp _y) : x(_x), y(_y) {}

    _Tp x;
    _Tp y;
};

typedef Point_<int> Point;
typedef Point_<float> Point2f;

struct Mat
{
    Mat() : data(0), refcount(0), rows(0), cols(0), c(0) {}

    Mat(int _rows, int _cols, int flags) : data(0), refcount(0)
    {
        create(_rows, _cols, flags);
    }

    // copy
    Mat(const Mat& m) : data(m.data), refcount(m.refcount)
    {
        if (refcount)
            NCNN_XADD(refcount, 1);

        rows = m.rows;
        cols = m.cols;
        c = m.c;
    }

    Mat(int _rows, int _cols, int flags, void* _data) : data((unsigned char*)_data), refcount(0)
    {
        rows = _rows;
        cols = _cols;
        c = flags;
    }

    ~Mat()
    {
        release();
    }

    // assign
    Mat& operator=(const Mat& m)
    {
        if (this == &m)
            return *this;

        if (m.refcount)
            NCNN_XADD(m.refcount, 1);

        release();

        data = m.data;
        refcount = m.refcount;

        rows = m.rows;
        cols = m.cols;
        c = m.c;

        return *this;
    }

    void create(int _rows, int _cols, int flags)
    {
        release();

        rows = _rows;
        cols = _cols;
        c = flags;

        if (total() > 0)
        {
            // refcount address must be aligned, so we expand totalsize here
            size_t totalsize = (total() + 3) >> 2 << 2;
            data = (unsigned char*)ncnn::fastMalloc(totalsize + (int)sizeof(*refcount));
            refcount = (int*)(((unsigned char*)data) + totalsize);
            *refcount = 1;
        }
    }

    void release()
    {
        if (refcount && NCNN_XADD(refcount, -1) == 1)
            ncnn::fastFree(data);

        data = 0;

        rows = 0;
        cols = 0;
        c = 0;

        refcount = 0;
    }

    Mat clone() const
    {
        if (empty())
            return Mat();

        Mat m(rows, cols, c);

        if (total() > 0)
        {
            memcpy(m.data, data, total());
        }

        return m;
    }


    bool empty() const { return data == 0 || total() == 0; }

    int channels() const { return c; }

    size_t total() const { return cols * rows * c; }

    const unsigned char* ptr(int y) const { return data + y * cols * c; }

    unsigned char* ptr(int y) { return data + y * cols * c; }

    // roi
    Mat operator()( const Rect& roi ) const
    {
        if (empty())
            return Mat();

        Mat m(roi.height, roi.width, c);

        int sy = roi.y;
        for (int y = 0; y < roi.height; y++)
        {
            const unsigned char* sptr = ptr(sy) + roi.x * c;
            unsigned char* dptr = m.ptr(y);
            memcpy(dptr, sptr, roi.width * c);
            sy++;
        }

        return m;
    }

    unsigned char* data;

    // pointer to the reference counter;
    // when points to user-allocated data, the pointer is NULL
    int* refcount;

    int rows;
    int cols;

    int c;

};

//retinaface
struct FaceObject
{
    //cv::Rect_<float> rect;
    //cv::Point2f landmark[5];

    Rect2f rect;
    Point2f landmark[5];

	float prob;
};

class RetinaFaceNet {
//
public:
    RetinaFaceNet(const string &model_path);//init
    ~RetinaFaceNet();
    void SetNumThreads(int numThreads);
	void detect(ncnn::Mat& img_, std::vector<FaceObject>& faceobjects);

private:

	inline float intersection_area(const FaceObject& a, const FaceObject& b);
	void qsort_descent_inplace(std::vector<FaceObject>& faceobjects, int left, int right);
	void qsort_descent_inplace(std::vector<FaceObject>& faceobjects);
	void nms_sorted_bboxes(const std::vector<FaceObject>& faceobjects, std::vector<int>& picked, float nms_threshold);
	ncnn::Mat generate_anchors(int base_size, const ncnn::Mat& ratios, const ncnn::Mat& scales);
	void generate_proposals(const ncnn::Mat& anchors, int feat_stride, const ncnn::Mat& score_blob, const ncnn::Mat& bbox_blob, const ncnn::Mat& landmark_blob, float prob_threshold, std::vector<FaceObject>& faceobjects);

	ncnn::Net retinaface;
    ncnn::Mat img;
    //ncnn::Mat bgr;

private://部分可调参数
	const float threshold[3] = { 0.8f, 0.8f, 0.6f };
	int minsize = 40;
	const float pre_facetor = 0.709f;
	int count = 10;
	int num_threads = 4;
};


#endif //__RETINAFACE_NCNN_H__
