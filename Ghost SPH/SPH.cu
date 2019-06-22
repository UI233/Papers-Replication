// SPH.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <fstream>
#include "glm/glm.hpp"
#include "CUDA_compute.h"
#include "include/input.h"
#include <ctime>
void outputAsCSV(ParticleManager &pm, int index) {
    auto str = std::to_string(index / 10);
    if (index % 10 == 0) {
        std::ofstream out("simulation" + str + ".csv");
        out << "x,y,z,v_x,v_y,v_z" << std::endl;
        for (auto &flow : pm.fluid) {
            auto &p = flow.second.p;
            auto &v = flow.second.v;
            out << p.x << "," << p.y << "," << p.z <<
                "," << v.x << "," << v.y << "," << v.z << std::endl;
        }
        for (auto &flow : pm.solid) {
            auto &p = flow.second.p;
            auto &v = flow.second.v;
            out << p.x << "," << p.y << "," << p.z <<
                "," << v.x << "," << v.y << "," << v.z << std::endl;
        }
        out.close();
    }
/*    vec3 res(0.0f);
    float x = 0.0f, y = 0.0f, z = 0.0f;
    for (auto &par : pm.fluid) {
        auto &gp=   par.second.gradpre;
        x = std::min(x, gp.x);
        y = std::min(y, gp.y);
        z = std::min(z, gp.z);
    }
    res = res / (float)pm.fluid.size();
/   std::cout << x << " " << y << " " << z << std::endl;*/
}

//void sampleTest() {
//    Grid grid(100, 100, 200, 0.005);
//    Cube sp(vec3(-0.4f, -0.4f, 0.0f), vec3(.83f, .0f, .0f), vec3(.0f, .83f, .0f), vec3(.0f, .0f, 0.80f));
//    auto vec = grid.sampleSurface(sp, 0.08, 20, 1.0845);
//    std::ofstream out("cube_surface.csv");
//    out << "x,y,z" << std::endl;
//    for (auto &pos : vec)
//        out << pos.x << "," << pos.y << "," << pos.z << std::endl;
//    out.close();
//    std::vector<vec3> seeds{ vec[0] };
//    vec = grid.sampleVolume([](vec3 p) {return true; }, sp, seeds, 0.08, 20);
//    out.open("cube_inner.csv", std::ios::out);
//    out << "x,y,z" << std::endl;
//    for (auto &pos : vec)
//        out << pos.x << "," << pos.y << "," << pos.z << std::endl;
//    out.close();
//}

void output(ParticleManager &pm, std::string path, int type) {
    std::ofstream out(path);
    out << "x,y,z\n";
    if (type == 0)
    for (auto &par : pm.p_solid) {
        auto pos = par.p;
        out << pos.x << "," << pos.y << "," << pos.z << "\n";
    }

    if (type == 1)
    for (auto &par : pm.p_fluid) {
        auto pos = par.p;
        out << pos.x << "," << pos.y << "," << pos.z << "\n";
    }

    if (type == 2)
    for (auto &par : pm.p_air) {
        auto pos = par.p;
        out << pos.x << "," << pos.y << "," << pos.z << "\n";
    }
    out.close();
}

int main()
{
    ParticleManager pm(0.008);
    unsigned int step;
    std::string path;
    std::cout << "Please specify the path to input file: ";
    std::cin >> path;
    std::cout << "Please specify the number of steps: ";
    std::cin >> step;

    parseInputFile(path, pm);
    output(pm, "air.csv", 2);
    output(pm, "water.csv", 1);
    output(pm, "fluid.csv", 0);
    auto points = cudaCompute(pm, step);
    //sampleTest();
    return 0;
}
