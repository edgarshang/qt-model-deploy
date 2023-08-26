#include <QApplication>
#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>
#include <onnxruntime_cxx_api.h>
#include <cpu_provider_factory.h>
#include <fstream>
#include "uideploy.h"
#include "ort_tutorial.h"
#include "ModelHandler.h"

using namespace InferenceEngine;
using namespace cv;
using namespace std;


int main(int argc, char *argv[])
{

    QApplication a(argc, argv);
    Deploy w;

    ModelHandler modelHandle(&w);
    w.setImageProcesser(&modelHandle);

    w.show();

    return a.exec();
}
