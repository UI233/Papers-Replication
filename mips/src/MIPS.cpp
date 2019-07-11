#include "MIPS.h"
#include "Eigen/SparseCholesky"
#include "Eigen/Core"
#include <algorithm>
#include <iostream>
#include <set>

// Project the point onto the unit square from a circle
OpenMesh::Vec2f project2Square(float theta) {
    //int quard = theta / (0.5 * M_PI);
    //quard = std::min(quard, 3);
    //theta -= 0.25 * M_PI;
    //switch (quard)
    //{
    //case 0:
    //   return OpenMesh::Vec2f(1.0f, tanf(theta));
    //case 1:
    //    return OpenMesh::Vec2f(1.0f / tanf(theta), 1.0f);
    //case 2:
    //    return OpenMesh::Vec2f(-1.0f, -tanf(theta));
    //case 3:
    //    return OpenMesh::Vec2f(-1.0f / tanf(theta), -1.0f);
    //}
    return OpenMesh::Vec2f(cosf(theta), sinf(theta));
}

// Get the initial parameterization of the mesh via
// minimizing the energy E = 1/2 \sum{|| p_i - p_j ||} for the inner point while the boundary points are projected onto the unit circle
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
        res[(*boundaries.begin()).idx()] = project2Square(0.0f);
        for (auto st = *boundaries.begin(); !visited[st.idx()];) {
            visited[st.idx()] = true;
            for (auto nx = mesh.vv_begin(st); nx.is_valid(); ++nx){ 
                if (boundaries.count(nx.handle()) && !visited[nx.handle().idx()]) {
                    angle += 2.0 * M_PI * (mesh.point(nx) - mesh.point(st)).norm() / length;
                    st = nx;
                    res[nx.handle().idx()] = project2Square(angle);
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
        Eigen::SimplicialLDLT<decltype(sparse)> solver;    
        auto X = solver.compute(sparse).solve(B);
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

auto calcGrad(const Mesh& mesh, const std::vector<OpenMesh::Vec2f> &vecs) -> std::vector<OpenMesh::Vec2f>{
    std::vector<OpenMesh::Vec2f> res;
    res.resize(vecs.size());
    for (auto v = mesh.vertices_begin(); v != mesh.vertices_end(); ++v) {
        res[v.handle().idx()][0] = 0.0f;
        res[v.handle().idx()][1] = 0.0f;
        auto st = mesh.cvv_begin(v);
        for (auto face: mesh.vf_range(v)){
            int cnt = 0;
            OpenMesh::VertexHandle nv, nx;
            for (auto adjv: mesh.fv_range(face)) {
                if (adjv != v) {
                    if (cnt++ == 0)
                        nv = adjv;
                    else nx = adjv;
                }
            }
            auto E1 = mesh.point(nv) - mesh.point(v);
            auto E2 = mesh.point(nx) - mesh.point(v);

            float x = E1.norm();
            float e2_comp_x = OpenMesh::dot(E1, E2) / x;
            float y = sqrtf(E2.norm() * E2.norm() - e2_comp_x * e2_comp_x);

            // The Matrix A = ((p1 - p0) / x, ((p2 - p0) - (p1 - p0) * e2_comp_x / x) / y)
            // calculate the row vector of Matrix A
            const OpenMesh::Vec2f& p0 = vecs[v.handle().idx()];
            const OpenMesh::Vec2f& p1 = vecs[nv.idx()];
            const OpenMesh::Vec2f& p2 = vecs[nx.idx()];
            const OpenMesh::Vec2f e0 = p1 - p0;
            const OpenMesh::Vec2f e1 = p2 - p0;
            const OpenMesh::Vec2f c0 = (e0 / x);
            const OpenMesh::Vec2f c1 = (e1 - e0 * e2_comp_x / x) / y;
            // E(A) = trace(AA') / det(A)
            // calculate the \partial{E(A)} / \partial{p0.x} and \partial{E(A)}/ \partial{p0.y}
            float mdet = c0[0] * c1[1] - c0[1] * c1[0];
            float traTa = OpenMesh::dot(c0, c0) + OpenMesh::dot(c1, c1);
            float dtraTa_dx = 2 * c0[0] / (-x) + 2 * c1[0] * (e2_comp_x / x - 1.0) / y;
            float dtraTa_dy = 2 * c0[1] / (-x) + 2 * c1[1] * (e2_comp_x / x - 1.0) / y;
            float dDet_dx = c1[1] / (-x) - c0[1] * (e2_comp_x / x - 1.0) / y;
            float dDet_dy = c0[0] * (e2_comp_x / x - 1.0) / y - c1[0] / (-x);
            if (mdet < 0.0f) {
                mdet = -mdet;
                dDet_dx = -dDet_dx;
                dDet_dy = -dDet_dy;
            }
            res[v.handle().idx()][0] += (dtraTa_dx * mdet - traTa * dDet_dx)/ (mdet * mdet);
            res[v.handle().idx()][1] += (dtraTa_dy * mdet - traTa * dDet_dy)/ (mdet * mdet);
        }
    }

    return std::move(res);
}

double energy(const Mesh &mesh, std::vector<OpenMesh::Vec2f>& vecs) {
    double E = 0.0;
    for (auto face = mesh.faces_begin(); face != mesh.faces_end(); ++face) {
        OpenMesh::Vec3f p[3];
        int handles[3];
        int cnt = 0;
        
        for (auto v: mesh.fv_range(face))  {
            handles[cnt] = v.idx();
            p[cnt] = mesh.point(v);
            ++cnt;
        }

        auto E1 = p[1] - p[0];
        auto E2 = p[2] - p[0];

        float x = E1.norm();
        float e2_comp_x = OpenMesh::dot(E1, E2) / x;
        float y = sqrtf(E2.norm() * E2.norm() - e2_comp_x * e2_comp_x);

        // The Matrix A = ((p1 - p0) / x, ((p2 - p0) - (p1 - p0) * e2_comp_x / x) / y)
        // calculate the row vector of Matrix A
        const OpenMesh::Vec2f& p0 = vecs[handles[0]];
        const OpenMesh::Vec2f& p1 = vecs[handles[1]];
        const OpenMesh::Vec2f& p2 = vecs[handles[2]];
        const OpenMesh::Vec2f e0 = p1 - p0;
        const OpenMesh::Vec2f e1 = p2 - p0;
        const OpenMesh::Vec2f c0 = (e0 / x);
        const OpenMesh::Vec2f c1 = (e1 - e0 * e2_comp_x / x) / y;
        // E(A) = trace(AA') / det(A)
        // calculate the \partial{E(A)} / \partial{p0.x} and \partial{E(A)}/ \partial{p0.y}
        float mdet = c0[0] * c1[1] - c0[1] * c1[0];
        if (mdet < 0.0f)
            mdet = -mdet;
        float traTa = OpenMesh::dot(c0, c0) + OpenMesh::dot(c1, c1);
        float ene = traTa / mdet;
        
        if (std::isinf(ene)) {
            std::cout << "stop";
        }

        E += traTa / mdet;
    }
    
    return E;
}

void gradientDescent(const Mesh& mesh, std::vector<OpenMesh::Vec2f>& init, float step = 0.005f) {
    constexpr float threshold = 0.0005f;
    auto res = init;
    auto norm = [](const decltype(init) vecs) {
        return std::accumulate(vecs.begin(), vecs.end(), 0.0f, [](float accum,  const OpenMesh::Vec2f& e){ return accum + OpenMesh::dot(e,e);});
    };
    int op_step = 0;
    std::vector<OpenMesh::Vec2f> grad;
    do {
        std::cout << op_step++ << " Energy: " << energy(mesh, init) << std::endl;
        grad = calcGrad(mesh, init);
        int i = 0;
        std::for_each(init.begin(), init.end(), [&i, &grad, &step](OpenMesh::Vec2f& e){e -= step * grad[i++];});
    } while(norm(grad) >= threshold);

    return ;
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
    mesh.triangulate();
    auto init_param = init(mesh); // the initial parameterization for the mesh
    // newton(mesh, init_param, error); // optimize the parameterization using Quasi Newton Method
    std::cout << "Initialization complete.\n";
    gradientDescent(mesh, init_param);
    std::cout << "Optimization complete.\n";
    return std::move(init_param);
}
