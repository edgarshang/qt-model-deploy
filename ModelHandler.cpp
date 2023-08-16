#include "ModelHandler.h"
#include <QDebug>
//#include "ort_tutorial.h"

ModelHandler::ModelHandler(Show *imageDisplay)
{
    display = imageDisplay;
}

void ModelHandler::processor(modelTypeInfo_ &info)
{
    qDebug() << "imageProcess...";
    qDebug() << info.modelType;
//    qDebug() << info.deploymode;
    qDebug() << info.filePath;
    if( info.deploymode == OnnxRunTime)
    {
        qDebug() << "onnxruntime";
        if( info.modelType == "resnet18")
        {
//            ort_tutorial test("D:/project/ort-deploy/resnet18.onnx", "D:/project/OpenCV/opencv_tutorial_data/images/space_shuttle.jpg", "D:/project/ort-deploy/imagenet_classes.txt","resnet18");
//            test.set_Show_image(display);
//            modelInference = test;
//            test.process();
        }
    }else if (info.deploymode == Openvino)
    {
        qDebug() << "openvino";
    }
}

//void ModelHandler::run()
//{

//}
