#include "common_api.h"
#include <string>
#include <fstream>

void Common_API::draw_pose_keyPoint(const float* data, cv::Mat &inputimage)
{
    const float* kpts_data = data;

    // draw key points
    // nose -> left_eye -> left_ear.(0, 1), (1, 3)
    cv::line(inputimage, cv::Point(kpts_data[0], kpts_data[1]), cv::Point(kpts_data[3], kpts_data[4]), cv::Scalar(255, 255, 0), 2, 8, 0);
    cv::line(inputimage, cv::Point(kpts_data[3], kpts_data[4]), cv::Point(kpts_data[9], kpts_data[10]), cv::Scalar(255, 255, 0), 2, 8, 0);

    // nose -> right_eye -> right_ear.(0, 2), (2, 4)
    cv::line(inputimage, cv::Point(kpts_data[0], kpts_data[1]), cv::Point(kpts_data[6], kpts_data[7]), cv::Scalar(255, 255, 0), 2, 8, 0);
    cv::line(inputimage, cv::Point(kpts_data[6], kpts_data[7]), cv::Point(kpts_data[12], kpts_data[13]), cv::Scalar(255, 255, 0), 2, 8, 0);

    // nose -> left_shoulder -> left_elbow -> left_wrist.(0, 5), (5, 7), (7, 9)
    cv::line(inputimage, cv::Point(kpts_data[0], kpts_data[1]), cv::Point(kpts_data[15], kpts_data[16]), cv::Scalar(255, 255, 0), 2, 8, 0);
    cv::line(inputimage, cv::Point(kpts_data[15], kpts_data[16]), cv::Point(kpts_data[21], kpts_data[22]), cv::Scalar(255, 255, 0), 2, 8, 0);
    cv::line(inputimage, cv::Point(kpts_data[21], kpts_data[22]), cv::Point(kpts_data[27], kpts_data[28]), cv::Scalar(255, 255, 0), 2, 8, 0);

    // nose -> right_shoulder -> right_elbow -> right_wrist.(0, 6), (6, 8), (8, 10)
    cv::line(inputimage, cv::Point(kpts_data[0], kpts_data[1]), cv::Point(kpts_data[18], kpts_data[19]), cv::Scalar(255, 255, 0), 2, 8, 0);
    cv::line(inputimage, cv::Point(kpts_data[18], kpts_data[19]), cv::Point(kpts_data[24], kpts_data[25]), cv::Scalar(255, 255, 0), 2, 8, 0);
    cv::line(inputimage, cv::Point(kpts_data[24], kpts_data[25]), cv::Point(kpts_data[30], kpts_data[31]), cv::Scalar(255, 255, 0), 2, 8, 0);

    // left_shoulder -> left_hip -> left_knee -> left_ankle.(5, 11), (11, 13), (13, 15)
    cv::line(inputimage, cv::Point(kpts_data[15], kpts_data[16]), cv::Point(kpts_data[33], kpts_data[34]), cv::Scalar(255, 255, 0), 2, 8, 0);
    cv::line(inputimage, cv::Point(kpts_data[33], kpts_data[34]), cv::Point(kpts_data[39], kpts_data[40]), cv::Scalar(255, 255, 0), 2, 8, 0);
    cv::line(inputimage, cv::Point(kpts_data[39], kpts_data[40]), cv::Point(kpts_data[45], kpts_data[46]), cv::Scalar(255, 255, 0), 2, 8, 0);

    // right_shoulder -> right_hip -> right_knee -> right_ankle.(6, 12), (12, 14), (14, 16)
    cv::line(inputimage, cv::Point(kpts_data[18], kpts_data[19]), cv::Point(kpts_data[36], kpts_data[37]), cv::Scalar(255, 255, 0), 2, 8, 0);
    cv::line(inputimage, cv::Point(kpts_data[36], kpts_data[37]), cv::Point(kpts_data[42], kpts_data[43]), cv::Scalar(255, 255, 0), 2, 8, 0);
    cv::line(inputimage, cv::Point(kpts_data[42], kpts_data[43]), cv::Point(kpts_data[48], kpts_data[49]), cv::Scalar(255, 255, 0), 2, 8, 0);

    // render all key point circles
    for (int c = 0; c < 17; c++) {
        cv::circle(inputimage, cv::Point(kpts_data[c * 3], kpts_data[c * 3 + 1]), 4, cv::Scalar(0, 255, 0), 3, 8, 0);
    }
}

std::vector<std::string> Common_API::readClassNames(std::string classNamePath)
{
    std::vector<std::string> classNames;
    std::ifstream fp(classNamePath);
    if (!fp.is_open())
    {
        printf("could not open file...\n");
        exit(-1);
    }

    std::string name;
    while (!fp.eof())
    {
        getline(fp, name);
        if (name.length())
        {
            classNames.push_back(name);
        }
    }
    fp.close();

    return classNames;
}

float Common_API::sigmoid_function(float a)
{
    float b = 1. / (1. + exp(-a));
    return b;
}
