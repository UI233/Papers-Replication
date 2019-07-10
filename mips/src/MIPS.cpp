#include "MIPS.h"
#include "Eigen/SparseLU"
#include "Eigen/Core"
#include <set>

std::vector<OpenMesh::Vec2f> init(Mesh& mesh) {
    std::set<Mesh::VertexHandle> boundaries;
    std::vector<OpenMesh::Vec2f> res;
    std::vector<int> inner;
    size_t vertices_sz = 0;
    size_t inner_count = -1;
    constexpr int coord = 2;

    // select the boundary
    for (auto cit = mesh.vertices_begin(); cit != mesh.vertices_end(); ++cit) {
        if (mesh.is_boundary(cit)) {
            boundaries.insert(*cit);
        }
        else inner_count++;
        inner.push_back(inner_count);
        ++vertices_sz;
    }
    res.resize(vertices_sz);
    Eigen::SparseMatrix<double> sparse(coord * (vertices_sz - boundaries.size()), coord * (vertices_sz - boundaries.size()));

    // compute the boundary points
    if (boundaries.size() > 0) {
        std::vector<bool> visited;
        visited.resize(vertices_sz);
        std::fill(visited.begin(), visited.end(), 0);
        float angle = 0.0f;
        float length = 0.0f;
        // compute the total length
        for (auto st = *boundaries.begin(); !visited[st.idx()];) {
            visited[st.idx()] = true;
            decltype(mesh.vv_begin(st)) nx;
            for (nx = mesh.vv_begin(st); nx.is_valid(); ++nx){ 
                if (boundaries.count(nx.handle()) && !visited[nx.handle().idx()]) {
                    length += (mesh.point(nx) - mesh.point(st)).norm();
                    st = nx;
                    break;
                }
            }
            if (visited[st.idx()]) {
                auto start = boundaries.begin();
                auto i1 = mesh.point(st), i2 = mesh.point(*start);
                length += (i1 - i2).norm();
            }
        }

        // compute the boundary point
        std::fill(visited.begin(), visited.end(), 0);
        res[(*boundaries.begin()).idx()] = OpenMesh::Vec2f(0.0f, 0.0f);
        for (auto st = *boundaries.begin(); !visited[st.idx()];) {
            visited[st.idx()] = true;
            for (auto nx = mesh.vv_begin(st); nx.is_valid(); ++nx){ 
                if (boundaries.count(nx.handle()) && !visited[nx.handle().idx()]) {
                    angle += 2.0 * M_PI * (mesh.point(nx) - mesh.point(st)).norm() / length;
                    st = nx;
                    res[nx.handle().idx()] = OpenMesh::Vec2f(cosf(angle), sinf(angle));
                    break;
                }
            }
        }
    }
    if (boundaries.size() != vertices_sz) {
        // initialize the boundary point and the column
        Eigen::VectorXd B(coord * (vertices_sz - boundaries.size())); 
        B.setZero();
        for (auto bd: boundaries) {
            auto point = mesh.point(bd);
            auto p = res[bd.idx()];
            res[bd.idx()] = p;
            for (auto it = mesh.vv_begin(bd); it.is_valid(); ++it) 
                if (!mesh.is_boundary(*it)) {
                    int ii = inner[it.handle().idx()];
                    auto dis = (point  - mesh.point(it)).norm();
                    sparse.coeffRef(coord * ii,  coord * ii) += 1.0 / dis;
                    sparse.coeffRef(coord * ii + 1,  coord * ii + 1) += 1.0 / dis;
                    B[coord * ii] += p[0] / dis;
                    B[coord * ii + 1] += p[1] / dis;
                }
        }
        // build the sparse matrix
        for (auto i = mesh.vertices_begin(); i != mesh.vertices_end(); ++i) {
            if(!boundaries.count(i)) {
                for (auto j = mesh.vv_iter(i); j.is_valid(); ++j) {
                    if (!boundaries.count(j) && *i != *j) {
                        int ii = inner[i.handle().idx()], ij = inner[j.handle().idx()];
                        double dis = (mesh.point(i) - mesh.point(j)).norm();
                        for (int cnt = 0; cnt < coord; ++cnt) {
                            sparse.coeffRef(coord * ii + cnt, coord * ij + cnt) -= 0.5 / dis;
                            sparse.coeffRef(coord * ij + cnt, coord * ii + cnt) -= 0.5 / dis;
                            sparse.coeffRef(coord * ii + cnt, coord * ii + cnt) += 1.0 / dis;
                        }
                    }
                }
            }
        }

        // solve the system to get the initial points set
        Eigen::SparseLU<decltype(sparse)> solver;    
        solver.analyzePattern(sparse);
        solver.compute(sparse);
        auto X = solver.solve(B);
        int i = 0;
        for (int j = 0; j < vertices_sz; ++j)
            if (!boundaries.count(Mesh::VertexHandle(j))){
                res[j][0] = X[i]; 
                res[j][1] = X[i + 1]; 
                i += 2;
            }
    }
    return std::move(res);
}

// !Todo
void newton(Mesh& mesh, std::vector<OpenMesh::Vec2f> &vecs, double error) {
    Eigen::VectorXd x, gradx;
    Eigen::SparseMatrix<double> Heissein;

    for (auto it = mesh.vertices_begin(); it != mesh.vertices_end(); ++it) {
        // 1-ring neighborhood
        auto st = mesh.vv_begin(it);
        for (auto iv = mesh.vv_begin(it); iv.is_valid(); ) {
            auto nx = (++iv).is_valid() ? iv : st;
            auto e1 = mesh.point(iv) - mesh.point(it);
            auto e2 = mesh.point(nx) - mesh.point(it);

            auto a = e1.norm();
            auto b = OpenMesh::dot(e1, e2) / a;
            b = sqrtf(e2.norm() * e2.norm() - b * b);
        } 
    }
}

std::vector<OpenMesh::Vec2f> paratimization(Mesh& mesh, double error) {
    auto init_param = init(mesh); // the initial parameterization for the mesh
    // newton(mesh, init_param, error); // optimize the parameterization using Quasi Newton Method
    return std::move(init_param);
}

