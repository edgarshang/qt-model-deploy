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
    Deploy w;

    ModelHandler modelHandle(&w);
    w.setImageProcesser(&modelHandle);

    w.show();

    ret = a.exec();
    return ret;
}
