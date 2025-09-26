#ifndef COMMON_API_H
#define COMMON_API_H

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <QString>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <QObject>

#define YOLOV5  "YOLOv5"
#define YOLOV8  "YOLOv8"
#define YOLOV11 "YOLOv11"
#define YOLOV5_SEG "YOLOv5_Seg"
#define YOLOV8_SEG "YOLOv8_Seg"
#define YOLOV11_SEG "YOLOv11_Seg"

using namespace nvinfer1;
using namespace nvonnxparser;

class Logger : public ILogger
{
    void  log(Severity severity, const char* msg) noexcept
    {
        if (severity <= Severity::kINFO)
        {
            std::cout << msg << std::endl;
        }
    }
};


enum DeployMode { OnnxRunTime, Openvino, TensorRT };
enum VedioMode {OpenCV, FFmpeg};
typedef struct
{
    QString modelType;
    QString filePath;
    DeployMode deploymode;
    float scores;
    float conf;
    VedioMode vedioTypeMode;

}modelTypeInfo_;

typedef struct
{
    std::string modelPath;
    std::string imagePath;
    std::string label_text;
    std::string modelType;
    float scoresThreshold;
    float confienceThreshold;
}modelConfInfo_;



class Common_API
{
public:
    static std::vector<std::string> readClassNames(std::string classNamePath);
    static float sigmoid_function(float a);
    static void draw_pose_keyPoint(const float* data, cv::Mat &input_image);
    static int load_tensorRT_model(char **trtMode, const char* modelPath);
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

class ModelProcessor : public QObject
{
    Q_OBJECT
public:
    virtual void modelRunner() = 0;
    virtual void modelStop()
    {

    }

signals:
    void FrameReady(cv::Mat &iamge);
};

class DeCode : public QObject
{
    Q_OBJECT
public:
    virtual void deCodeImage() = 0;
    DeCode(QString path):QObject(nullptr)
    {
        videoPath = path;
    }
signals:
    void frameReady(cv::Mat &frame);

public:
    QString videoPath;
};





#endif // COMMON_API_H
