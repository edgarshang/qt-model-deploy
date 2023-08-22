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
        }else if( info.modelType == "YOLOv5" || info.modelType == "YOLOv8")
        {
            yolov5_onnx_deploy = std::make_shared<Yolov5_Onnx_Deploy>((info.modelType == "YOLOv5" ? "D:/project/ort-deploy/yolov5s.onnx":"D:/project/ort-deploy/yolov8n.onnx"),
                                                                      info.filePath.toStdString(), "D:/project/ort-deploy/classes.txt", info.modelType.toStdString());
            yolov5_onnx_deploy->set_Show_image(display);
            modelInference = yolov5_onnx_deploy;
            this->start();
        }else if( info.modelType == "FasterRcnn" || info.modelType == "RetinaNet")
        {
            qDebug() << "info.modelType = " << info.modelType;
            faster_rcnn_deploy = std::make_shared<FasterRcnn>((info.modelType == "FasterRcnn" ? "D:/project/ort-deploy/faster_rcnn.onnx" : "D:/project/ort-deploy/retinanet_resnet50_fpn.onnx"),
                                                              info.filePath.toStdString(), "D:/project/ort-deploy/classes.txt", info.modelType.toStdString());
            faster_rcnn_deploy->set_Show_image(display);
            modelInference = faster_rcnn_deploy;

            this->start();
        }else {
            qDebug() << "models deploys not supported!!!";
        }
    }else if (info.deploymode == Openvino)
    {
        qDebug() << "openvino";
    }
}

void ModelHandler::run()
{
    if(modelInference != nullptr)
    {
        modelInference->modelRunner();
    }
}
