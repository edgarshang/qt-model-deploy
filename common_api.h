#ifndef COMMON_API_H
#define COMMON_API_H

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>


class Common_API
{
public:
    static std::vector<std::string> readClassNames(std::string classNamePath);
};

class Show
{
public:
    virtual void imageshow(cv::Mat &image) = 0;
};

#endif // COMMON_API_H
