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
        deploy.cpp

HEADERS += \
        deploy.h

INCLUDEPATH += $$quote(C:\Program Files (x86)\Intel\openvino_2021.4.752\opencv\include) \
               $$quote(C:\Program Files (x86)\Intel\openvino_2021.4.752\opencv\include\opencv2) \
               $$quote(C:\Program Files (x86)\Intel\openvino_2021.4.752\deployment_tools\inference_engine\include) \
               D:\project\onnxruntime-win-x64-1.13.1\include

LIBS += -L$$quote(C:\Program Files (x86)\Intel\openvino_2021.4.752\opencv\lib) \
        -lopencv_calib3d453 \
        -lopencv_core453 \
        -lopencv_dnn453 \
        -lopencv_features2d453 \
        -lopencv_flann453 \
        -lopencv_gapi453 \
        -lopencv_highgui453 \
        -lopencv_imgcodecs453 \
        -lopencv_imgproc453 \
        -lopencv_ml453 \
        -lopencv_objdetect453 \
        -lopencv_photo453 \
        -lopencv_stitching453 \
        -lopencv_video453 \
        -lopencv_videoio453

LIBS += -L$$quote(C:\Program Files (x86)\Intel\openvino_2021.4.752\deployment_tools\inference_engine\lib\intel64\Release) \
        -linference_engine \
        -linference_engine_c_api \
        -linference_engine_transformations

LIBS += -LD:\project\onnxruntime-win-x64-1.13.1\lib \
        -lonnxruntime \
        -lonnxruntime_providers_shared

