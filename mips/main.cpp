#define OM_STATIC_BUILD
#include "MIPS.h"
#include "OpenMesh/Core/IO/MeshIO.hh"
#include <iostream>
#include <fstream>
#include <string>

int main(int argc, char *argv[]) {
    std::string path;
    if (argc < 2) {
        std::cout << "Please specify the file name" << std::endl;
        std::cin >> path;
    }
    else path = argv[1];

    // Import mesh
    Mesh mesh;
    OpenMesh::IO::read_mesh(mesh, path);
    Mesh::Point point;
    mesh.triangulate();
    // for (auto it = mesh.vertices_begin(); it != mesh.vertices_end(); ++it) {
    //     point = mesh.point(*it);
    //     std::cout << point[0] << " " << point[1] << " " << point[2] << std::endl;
    // }

    using vh = OpenMesh::VertexHandle;
    std::cout << mesh.is_boundary(vh(10099)) << mesh.is_boundary(vh(10199)) << mesh.is_boundary(vh(10200)) << std::endl;
    auto param = paratimization(mesh);

    // Output the result of parameterization
    std::ofstream out("param.csv");
    out << "x,y" << std::endl;
    for (int i = 0; i < param.size(); ++i) {
        // if (mesh.is_boundary(OpenMesh::VertexHandle(i))) 
        out << param[i][0] << "," << param[i][1] << "\n";
        // else out << "0,0\n";
    }
    out.close();

    out.open("edge.csv");
    out << "from,to\n";
    for (auto it = mesh.vertices_begin(); it != mesh.vertices_end(); ++it) {
        for (auto adjc = mesh.vv_begin(it); adjc.is_valid(); ++ adjc) {
            if (it.handle().idx() < adjc.handle().idx())
                out << it.handle().idx() << "," << adjc.handle().idx() << "\n";
        }
    }
    out.close();
    return 0;
}