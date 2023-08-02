#include "deploy.h"
#include <QApplication>
#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>
#include <onnxruntime_cxx_api.h>

using namespace InferenceEngine;
using namespace cv;

int main(int argc, char *argv[])
{
//      Mat src = Mat::zeros(Size(300,500), CV_8UC3);
//      circle(src, Point(50,50), 30,Scalar(255,0,0),2,8,0);
//      imshow("test", src);
//      waitKey(0);
//      destroyAllWindows();

//      return 0;
    QApplication a(argc, argv);
    Deploy w;
    w.show();

    return a.exec();
}
