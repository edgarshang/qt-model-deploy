#-------------------------------------------------
#
# Project created by QtCreator 2023-08-02T14:58:12
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = qt-deploy
TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0


SOURCES += \
        main.cpp \
    resnet18_TensorRT.cpp \
    uideploy.cpp \
    ort_tutorial.cpp \
    common_api.cpp \
    ModelHandler.cpp \
    Yolov5_Onnx_Deploy.cpp \
    FasterRcnn.cpp \
    Yolov5_Seg_Onnx.cpp \
    MaskRcnn_Seg_Onnx.cpp \
    DeepLabV3.cpp \
    Unet.cpp \
    keyPointRcnn.cpp \
    Yolov8_KeyPoint.cpp \
    Yolov6_Face.cpp \
    Unet_Road_Openvino.cpp \
    Yolov5_Openvino_Deploy.cpp \
    Yolov5_Seg_Openvino_Deploy.cpp \
    Yolov8_KeyPoint_Openvino.cpp \
    Yolov6_Face_Openvino.cpp \
    MaskRcnn_Seg_Openvino_Deploy.cpp \
    keyPointRcnn_Openvino_Deploy.cpp \
    FastRcnn_Openvino_Deploy.cpp \
    DeepLabV3_Openvino_Deploy.cpp \
    Resnet18_Openvino_Deploy.cpp

HEADERS += \
    resnet18_TensorRT.h \
    uideploy.h \
    ort_tutorial.h \
    common_api.h \
    ModelHandler.h \
    Yolov5_Onnx_Deploy.h \
    FasterRcnn.h \
    Yolov5_Seg_Onnx.h \
    MaskRcnn_Seg_Onnx.h \
    DeepLabV3.h \
    Unet.h \
    keyPointRcnn.h \
    Yolov8_KeyPoint.h \
    Yolov6_Face.h \
    Unet_Road_Openvino.h \
    Yolov5_Openvino_Deploy.h \
    Yolov5_Seg_Openvino_Deploy.h \
    Yolov8_KeyPoint_Openvino.h \
    Yolov6_Face_Openvino.h \
    MaskRcnn_Seg_Openvino_Deploy.h \
    keyPointRcnn_Openvino_Deploy.h \
    FastRcnn_Openvino_Deploy.h \
    DeepLabV3_Openvino_Deploy.h \
    Resnet18_Openvino_Deploy.h

INCLUDEPATH += $$quote(D:\project\OpenCV\opencvcode\build\include) \
               $$quote(D:\project\OpenCV\opencvcode\build\include\opencv2) \
               $$quote(C:\Program Files (x86)\Intel\openvino_2022.3\runtime\include) \
               $$quote(C:\Program Files (x86)\Intel\openvino_2022.3\runtime\include\ie) \
               $$quote(C:\Program Files (x86)\Intel\openvino_2022.3\runtime\include\ngraph) \
               $$quote(C:\Program Files (x86)\Intel\openvino_2022.3\runtime\include\openvino) \
               $$quote(C:\Program Files (x86)\Intel\openvino_2022.3\runtime\3rdparty\tbb\include) \
               $$quote(D:\project\onnxruntime-win-x64-1.13.1\include) \
               $$quote(D:\software-pack\TensorRT-10.6.0.26.Windows.win10.cuda-12.6\TensorRT-10.6.0.26\include) \
               $$quote(C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\include)

LIBS += -L$$quote(D:\project\OpenCV\opencvcode\build\x64\vc15\lib) \
        -lopencv_world454



LIBS += -L$$quote(D:\project\onnxruntime-win-x64-1.13.1\lib) \
        -lonnxruntime \
        -lonnxruntime_providers_shared

LIBS += -L$$quote(C:\Program Files (x86)\Intel\openvino_2022.3\runtime\3rdparty\tbb\lib) \
        -ltbb \
        -ltbb_preview \
        -ltbbbind \
        -ltbbmalloc \
        -ltbbmalloc_proxy \
        -ltbbproxy

LIBS += -L$$quote(C:\Program Files (x86)\Intel\openvino_2022.3\runtime\lib\intel64\Release) \
        -lopenvino \
        -lopenvino_c \
        -lopenvino_onnx_frontend

LIBS += -L$$quote(D:\software-pack\TensorRT-10.6.0.26.Windows.win10.cuda-12.6\TensorRT-10.6.0.26\lib) \
        -lnvinfer_10 \
        -lnvinfer_dispatch_10 \
        -lnvinfer_lean_10 \
        -lnvinfer_plugin_10 \
        -lnvinfer_vc_plugin_10 \
        -lnvonnxparser_10

LIBS += -L$$quote(C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\lib\x64) \
        -lcublas \
        -lcublasLt \
        -lcuda \
        -lcudadevrt \
        -lcudart \
        -lcudart_static \
        -lcufft \
        -lcufftw \
        -lcufilt \
        -lcurand \
        -lcusolver \
        -lcusolverMg \
        -lcusparse \
        -lnppc \
        -lnppial \
        -lnppicc \
        -lnppidei \
        -lnppif \
        -lnppig \
        -lnppim \
        -lnppist \
        -lnppisu \
        -lnppitc \
        -lnpps \
        -lnvblas \
        -lnvfatbin \
        -lnvfatbin_static \
        -lnvJitLink \
        -lnvJitLink_static \
        -lnvjpeg \
        -lnvml \
        -lnvptxcompiler_static \
        -lnvrtc-builtins_static \
        -lnvrtc \
        -lnvrtc_static \
        -lOpenCL



