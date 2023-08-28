#ifndef MODELHANDLER_H
#define MODELHANDLER_H

#include <QString>
#include <QThread>
#include <iostream>
#include <memory>
#include "common_api.h"
#include "ort_tutorial.h"
#include "Yolov5_Onnx_Deploy.h"
#include "Yolov5_Seg_Onnx.h"
#include "FasterRcnn.h"
#include "MaskRcnn_Seg_Onnx.h"
#include "DeepLabV3.h"
#include "Unet.h"


class ModelHandler : public QThread,  public ImageProcessor
{
public:
    ModelHandler(Show *imageDisplay);
    virtual void processor(modelTypeInfo_ &info);
    Show *display;

//    ort_tutorial *ort_test;
    std::shared_ptr<ort_tutorial> ort_test;
    std::shared_ptr<Yolov5_Onnx_Deploy> yolov5_onnx_deploy;
    std::shared_ptr<Yolov5_Seg_Onnx> yolov5_seg_onnx_deploy;
    std::shared_ptr<FasterRcnn> faster_rcnn_deploy;
    std::shared_ptr<ModelProcessor> modelInference;
    std::shared_ptr<MaskRcnn_Seg_Onnx> maskRcnn_Seg_onnx_deploy;
    std::shared_ptr<DeepLabV3> deepLabV3_onnx_deploy;
    std::shared_ptr<Unet> unet_onnx_deploy;

protected:
    void run();



};

#endif // MODELHANDLER_H
