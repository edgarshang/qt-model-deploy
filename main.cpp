#include <QApplication>
#include "uideploy.h"
#include "ModelHandler.h"

#include <openvino/openvino.hpp>


using namespace cv;
using namespace std;
using namespace ov;

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <NvOnnxParser.h>

using namespace nvinfer1;
using namespace nvonnxparser;
using namespace cv;

class Logger :public ILogger
{
    void  log(Severity severity, const char* msg) noexcept
    {
        if (severity != Severity::kINFO)
        {
            std::cout << msg << std::endl;
        }
    }
}gLogger;


int main(int argc, char *argv[])
{

//    printf("hello, wrold");

//        auto builder = createInferBuilder(gLogger);
//        builder->getLogger()->log(nvinfer1::ILogger::Severity::kERROR, "Create Builder...");

//        return 0;
    QApplication a(argc, argv);
    Deploy w;

    ModelHandler modelHandle(&w);
    w.setImageProcesser(&modelHandle);

    w.show();

    return a.exec();
}
