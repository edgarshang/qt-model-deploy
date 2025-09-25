#include <QApplication>
#include "uideploy.h"
#include "ModelHandler.h"

#include <openvino/openvino.hpp>


using namespace cv;
using namespace std;
using namespace ov;


int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    int ret;


    ModelHandler modelHandle;
    modelHandle.setDisplayer(&modelHandle.m_ui);
    modelHandle.m_ui.setImageProcesser(&modelHandle);
//    w.setImageProcesser(&modelHandle);

//    w.show();
    modelHandle.m_ui.show();
    ret = a.exec();
    return ret;
}
