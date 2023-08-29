#ifndef COMMON_API_H
#define COMMON_API_H

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <QString>

#define YOLOV5  "YOLOv5"
#define YOLOV8  "YOLOv8"
#define YOLOV5_SEG "YOLOv5_Seg"
#define YOLOV8_SEG "YOLOv8_Seg"

enum DeployMode { OnnxRunTime, Openvino };
typedef struct
{
    QString modelType;
    QString filePath;
    DeployMode deploymode;

}modelTypeInfo_;


class Common_API
{
public:
    static std::vector<std::string> readClassNames(std::string classNamePath);
    static float sigmoid_function(float a);
    static void draw_pose_keyPoint(const float* data, cv::Mat &input_image);
};


class Show
{
public:
    virtual void imageshow(cv::Mat &image) = 0;
};

class ImageProcessor
{
public:
    virtual void processor(modelTypeInfo_ &info) = 0;
};

class ModelProcessor
{
public:
    virtual void modelRunner() = 0;
};





#endif // COMMON_API_H
