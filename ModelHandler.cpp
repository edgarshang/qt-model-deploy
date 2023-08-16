#include "ModelHandler.h"
#include <QDebug>
#include "ort_tutorial.h"

ModelHandler::ModelHandler(Show *imageDisplay)
{
    display = imageDisplay;
//    connect(this, SIGNAL(finished()), this, SLOT(QObject::deleteLater));
}


void ModelHandler::processor(modelTypeInfo_ &info)
{
//    qDebug() << "imageProcess...";
//    qDebug() << info.modelType;
//    qDebug() << info.deploymode;
//    qDebug() << info.filePath;
    if( info.deploymode == OnnxRunTime)
    {
        qDebug() << "onnxruntime";
        if( info.modelType == "resnet18")
        {
            ort_test = std::make_shared<ort_tutorial>("D:/project/ort-deploy/resnet18.onnx", info.filePath.toStdString(), "D:/project/ort-deploy/imagenet_classes.txt");
            ort_test->set_Show_image(display);
            modelInference = ort_test;
            this->start();
        }else if( info.modelType == "YOLOv5" )
        {
            yolov5_onnx_deploy = std::make_shared<Yolov5_Onnx_Deploy>("D:/project/ort-deploy/yolov5s.onnx", info.filePath.toStdString(), "D:/project/ort-deploy/classes.txt");
            yolov5_onnx_deploy->set_Show_image(display);
            modelInference = yolov5_onnx_deploy;
            this->start();
        }else if( info.modelType == "YOLOv8" )
        {

        }else if( info.modelType == "FasterRcnn" )
        {

        }else if( info.modelType == "MaskRcnn" )
        {

        }else if( info.modelType == "Unet" )
        {

        }else if( info.modelType == "resnet18" )
        {

        }
    }else if (info.deploymode == Openvino)
    {
        qDebug() << "openvino";
    }
}

void ModelHandler::run()
{
     modelInference->modelRunner();
     qDebug() << "run()...";
}
