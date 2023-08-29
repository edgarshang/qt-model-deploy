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
        }else if( info.modelType == YOLOV5 || info.modelType == YOLOV8)
        {
            yolov5_onnx_deploy = std::make_shared<Yolov5_Onnx_Deploy>((info.modelType == YOLOV5 ? "D:/project/ort-deploy/yolov5s.onnx":"D:/project/ort-deploy/yolov8n.onnx"),
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
        }else if(info.modelType == YOLOV5_SEG || info.modelType == YOLOV8_SEG){
            qDebug() << "info.modelType : " << info.modelType;
            yolov5_seg_onnx_deploy = std::make_shared<Yolov5_Seg_Onnx>((info.modelType == YOLOV5_SEG ? "D:/project/ort-deploy/yolov5s-seg.onnx":"D:/project/ort-deploy/yolov8n-seg.onnx"),
                                                                      info.filePath.toStdString(), "D:/project/ort-deploy/classes.txt", info.modelType.toStdString());
            yolov5_seg_onnx_deploy->set_Show_image(display);
            modelInference = yolov5_seg_onnx_deploy;
            this->start();
        }else if(info.modelType == "MaskRcnn")
        {
            qDebug() << "info.modelType : " << info.modelType;
            maskRcnn_Seg_onnx_deploy = std::make_shared<MaskRcnn_Seg_Onnx>("D:/project/ort-deploy/mask_rcnn.onnx", info.filePath.toStdString(),"D:/project/ort-deploy/classes.txt");
            maskRcnn_Seg_onnx_deploy->set_Show_image(display);
            modelInference = maskRcnn_Seg_onnx_deploy;
            this->start();
        }else if("DeepLabV3" == info.modelType)
        {
            qDebug() << "info.modelType: " << info.modelType;
            deepLabV3_onnx_deploy = std::make_shared<DeepLabV3>("D:/project/ort-deploy/deeplabv3_mobilenet.onnx", info.filePath.toStdString(), "D:/project/ort-deploy/classes.txt", info.modelType.toStdString());
            deepLabV3_onnx_deploy->set_Show_image(display);
            modelInference = deepLabV3_onnx_deploy;
            this->start();
        }else if("Unet" == info.modelType)
        {
            qDebug() << "info.modelType: " << info.modelType;
            unet_onnx_deploy = std::make_shared<Unet>("D:/project/ort-deploy/unet_road.onnx", info.filePath.toStdString(), "D:/project/ort-deploy/classes.txt", info.modelType.toStdString());
            unet_onnx_deploy->set_Show_image(display);
            modelInference = unet_onnx_deploy;
            this->start();
        }else if("keyPointRcnn" == info.modelType)
        {
            qDebug() << "info.modelType: " << info.modelType;
            keyPointRcnn_onnx_deploy = std::make_shared<keyPointRcnn>("D:/project/ort-deploy/keypointrcnn_resnet50_fpn.onnx", info.filePath.toStdString(),
                                                                      "D:/project/ort-deploy/classes.txt", info.modelType.toStdString());
            keyPointRcnn_onnx_deploy->set_Show_image(display);
            modelInference = keyPointRcnn_onnx_deploy;
            this->start();
        }
        else {
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
