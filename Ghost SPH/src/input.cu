#include "surface.h"
#include "input.h"
#include <fstream>
#include <memory>

inline std::string toUpper(const std::string &str) {
    std::string res = str;
    for (auto &ch : res)
        ch = toupper(ch);

    return std::move(res);
}

std::shared_ptr<Surface> getSurfaceInformation(std::istream &input,  std::string type) {
    if(type == std::string("SPHERE")){
        float x, y, z, r;
        input >> x >> y >> z >> r;
        return std::shared_ptr<Sphere>(new Sphere(vec3(x, y, z), r));
    }
    else if (type == std::string("CUBE")) {
        vec3 points[4];
        for (int i = 0; i < 4; ++i)
            input >> points[i].x >> points[i].y >> points[i].z;
        return std::shared_ptr<Cube>(new Cube(points[0], points[1] , points[2], points[3]));
    }
    else if (type == std::string("TRIANGLE")) {
        vec3 points[3];
        for (int i = 0; i < 3; ++i)
            input >> points[i].x >> points[i].y >> points[i].z;
        return std::shared_ptr<Triangle>(new Triangle(points[0], points[1], points[2]));
    }
    return nullptr;
}

bool parseInputFile(std::string path, ParticleManager &pm) {
    std::ifstream input(path);
    std::vector<std::shared_ptr<Surface>> water, solid;
    while (input) {
        std::string surface_type;
        int num;
        input >> surface_type >> num;
        surface_type = toUpper(surface_type);
        if (surface_type == std::string("SOLID")) {
            for (int i = 0; i < num; ++i) {
                input >> surface_type;
                surface_type = toUpper(surface_type);
                solid.push_back(getSurfaceInformation(input,surface_type));
            }
        }
        else if (surface_type == std::string("WATER")) {
            for (int i = 0; i < num; ++i) {
                input >> surface_type;
                surface_type = toUpper(surface_type);
                water.push_back(getSurfaceInformation(input, surface_type));
            }
        }
    }

    pm.initSamples(water, solid);

    return true;
}