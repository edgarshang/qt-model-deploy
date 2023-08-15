#include "common_api.h"
#include <string>
#include <fstream>

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
