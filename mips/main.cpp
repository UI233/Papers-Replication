#define OM_STATIC_BUILD
#include "MIPS.h"
#include "OpenMesh/Core/IO/MeshIO.hh"
#include <iostream>


int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << "Please specify the file name" << std::endl;
        return 0;
    }

    Mesh mesh;
    OpenMesh::IO::read_mesh(mesh, argv[1]);
    Mesh::Point point;
    for (auto it = mesh.vertices_begin(); it != mesh.vertices_end(); ++it) {
        point = mesh.point(*it);
        std::cout << point[0] << " " << point[1] << " " << point[2] << std::endl;
    }

    auto param = paratimization(mesh);
    return 0;
}