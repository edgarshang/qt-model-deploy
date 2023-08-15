#include <QApplication>
#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>
#include <onnxruntime_cxx_api.h>
#include <cpu_provider_factory.h>
#include <fstream>
#include "uideploy.h"
#include "ort_tutorial.h"

using namespace InferenceEngine;
using namespace cv;
using namespace std;


int main(int argc, char *argv[])
{

    QApplication a(argc, argv);
    Deploy w;
    ort_tutorial test("D:/project/ort-deploy/resnet18.onnx", "D:/project/OpenCV/opencv_tutorial_data/images/space_shuttle.jpg", "D:/project/ort-deploy/imagenet_classes.txt","resnet18");
//       test.process();
    test.set_Show_image(&w);
    w.show();
     test.process();

    return a.exec();


//    std::cout << "hello, world" << std::endl;
//    ort_tutorial test("D:/project/ort-deploy/resnet18.onnx", "D:/project/OpenCV/opencv_tutorial_data/images/space_shuttle.jpg", "D:/project/ort-deploy/imagenet_classes.txt","resnet18");
//    test.process();
//    return 0;


}
