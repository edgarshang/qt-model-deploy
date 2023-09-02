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
#include "keyPointRcnn.h"
#include "Yolov8_KeyPoint.h"
#include "Yolov6_Face.h"
#include "Unet_Road_Openvino.h"
#include "Yolov5_Openvino_Deploy.h"
#include "Yolov5_Seg_Openvino_Deploy.h"
#include "Yolov8_KeyPoint_Openvino.h"
#include "Yolov6_Face_Openvino.h"
#include "MaskRcnn_Seg_Openvino_Deploy.h"
#include "keyPointRcnn_Openvino_Deploy.h"
#include "FastRcnn_Openvino_Deploy.h"
#include "DeepLabV3_Openvino_Deploy.h"


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
    std::shared_ptr<keyPointRcnn> keyPointRcnn_onnx_deploy;
    std::shared_ptr<Yolov8_KeyPoint> yolov8_pose_deploy;
    std::shared_ptr<Yolov6_Face> yolov6_face_deploy;
    std::shared_ptr<Unet_Road_Openvino> unet_openvino_deploy;
    std::shared_ptr<Yolov5_Openvino_Deploy> yolov5_openvino_deploy;
    std::shared_ptr<Yolov5_Seg_Openvino_Deploy> yolov5_seg_openvino_deploy;
    std::shared_ptr<Yolov8_KeyPoint_Openvino> yolov8_keypoint_openvino_deploy;
    std::shared_ptr<Yolov6_Face_Openvino_Deploy> yolov6_face_openvino_deploy;
    std::shared_ptr<MaskRcnn_Seg_Openvino_Deploy> maskrcnn_seg_openvino_deploy;
    std::shared_ptr<keyPointRcnn_Openvino_Deploy> keypointrcnn_openvino_deploy;
    std::shared_ptr<FastRcnn_Openvino_Deploy> fasterrcnn_openvino_deploy;
    std::shared_ptr<DeepLabV3_Openvino_Deploy> deeplabv3_openvino_deploy;

    modelConfInfo_ modelInfo;

protected:
    void run();



};

#endif // MODELHANDLER_H
