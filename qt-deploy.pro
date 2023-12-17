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
               $$quote(D:\project\TensorRT-8.6.0.12\include) \
               $$quote(C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\include) \
               $$quote(D:\appSoftware\ffmpeg-6.1-full_build-shared\include)

#LIBS += -L$$quote(C:\Program Files (x86)\Intel\openvino_2021.4.752\opencv\lib) \
#        -lopencv_calib3d453 \
#        -lopencv_core453 \
#        -lopencv_dnn453 \
#        -lopencv_features2d453 \
#        -lopencv_flann453 \
#        -lopencv_gapi453 \
#        -lopencv_highgui453 \
#        -lopencv_imgcodecs453 \
#        -lopencv_imgproc453 \
#        -lopencv_ml453 \
#        -lopencv_objdetect453 \
#        -lopencv_photo453 \
#        -lopencv_stitching453 \
#        -lopencv_video453 \
#        -lopencv_videoio453
LIBS += -L$$quote(D:\project\OpenCV\opencvcode\build\x64\vc15\lib) \
        -lopencv_world454

#LIBS += -L$$quote(C:\Program Files (x86)\Intel\openvino_2021.4.752\deployment_tools\inference_engine\lib\intel64\Release) \
#        -linference_engine \
#        -linference_engine_c_api \
#        -linference_engine_transformations

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

LIBS += -L$$quote(D:\project\TensorRT-8.6.0.12\lib) \
        -lnvinfer \
        -lnvinfer_dispatch \
        -lnvinfer_lean \
        -lnvinfer_plugin \
        -lnvinfer_vc_plugin \
        -lnvonnxparser \
        -lnvparsers

LIBS += -L$$quote(C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\lib\x64) \
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
        -lnvjpeg \
        -lnvml \
        -lnvptxcompiler_static \
        -lnvrtc-builtins_static \
        -lnvrtc \
        -lnvrtc_static \
        -lOpenCL

LIBS += -L$$quote(D:\appSoftware\ffmpeg-6.1-full_build-shared\lib) \
        -lavcodec \
        -lavdevice \
        -lavfilter \
        -lavformat \
        -lavutil \
        -lpostproc \
        -lswresample \
        -lswscale







