#include "MIPS.h"
#include "Eigen/SparseCholesky"
#include "Eigen/Core"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <set>
#include <numeric>

// Project the point onto the unit square from a circle
OpenMesh::Vec2f project2Circle(double theta) {
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
        double angle = 0.0f;
        double length = 0.0f;
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
        res[(*boundaries.begin()).idx()] = project2Circle(0.0f);
        for (auto st = *boundaries.begin(); !visited[st.idx()];) {
            visited[st.idx()] = true;
            for (auto nx = mesh.vv_begin(st); nx.is_valid(); ++nx){ 
                if (boundaries.count(nx.handle()) && !visited[nx.handle().idx()]) {
                    angle += 2.0 * M_PI * (mesh.point(nx) - mesh.point(st)).norm() / length;
                    st = nx;
                    res[nx.handle().idx()] = project2Circle(angle);
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

            double x = E1.norm();
            double e2_comp_x = OpenMesh::dot(E1, E2) / x;
            double y = sqrtf(E2.norm() * E2.norm() - e2_comp_x * e2_comp_x);

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
            double mdet = c0[0] * c1[1] - c0[1] * c1[0];
            double traTa = OpenMesh::dot(c0, c0) + OpenMesh::dot(c1, c1);
            double dtraTa_dx = 2 * c0[0] / (-x) + 2 * c1[0] * (e2_comp_x / x - 1.0) / y;
            double dtraTa_dy = 2 * c0[1] / (-x) + 2 * c1[1] * (e2_comp_x / x - 1.0) / y;
            double dDet_dx = c1[1] / (-x) - c0[1] * (e2_comp_x / x - 1.0) / y;
            double dDet_dy = c0[0] * (e2_comp_x / x - 1.0) / y - c1[0] / (-x);
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

auto calcGrad(const Mesh &mesh, const Eigen::VectorXd &vecs) -> Eigen::VectorXd {
    Eigen::VectorXd res(vecs.size() * 2);
    constexpr int coord = 2;
    for (auto v = mesh.vertices_begin(); v != mesh.vertices_end(); ++v) {
        res[coord * v.handle().idx()] = 0.0f;
        res[coord * v.handle().idx() + 1] = 0.0f;
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

            double x = E1.norm();
            double e2_comp_x = OpenMesh::dot(E1, E2) / x;
            double y = sqrtf(E2.norm() * E2.norm() - e2_comp_x * e2_comp_x);

            // The Matrix A = ((p1 - p0) / x, ((p2 - p0) - (p1 - p0) * e2_comp_x / x) / y)
            // calculate the row vector of Matrix A
            const OpenMesh::Vec2f& p0 = OpenMesh::Vec2f(vecs[coord * v.handle().idx()], vecs[coord * v.handle().idx() + 1]);
            const OpenMesh::Vec2f& p1 = OpenMesh::Vec2f(vecs[coord * nv.idx()], vecs[coord * nv.idx() + 1]);
            const OpenMesh::Vec2f& p2 = OpenMesh::Vec2f(vecs[coord * nx.idx()], vecs[coord * nx.idx() + 1]);
            const OpenMesh::Vec2f e0 = p1 - p0;
            const OpenMesh::Vec2f e1 = p2 - p0;
            const OpenMesh::Vec2f c0 = (e0 / x);
            const OpenMesh::Vec2f c1 = (e1 - e0 * e2_comp_x / x) / y;
            // E(A) = trace(AA') / det(A)
            // calculate the \partial{E(A)} / \partial{p0.x} and \partial{E(A)}/ \partial{p0.y}
            double mdet = c0[0] * c1[1] - c0[1] * c1[0];
            double traTa = OpenMesh::dot(c0, c0) + OpenMesh::dot(c1, c1);
            double dtraTa_dx = 2 * c0[0] / (-x) + 2 * c1[0] * (e2_comp_x / x - 1.0) / y;
            double dtraTa_dy = 2 * c0[1] / (-x) + 2 * c1[1] * (e2_comp_x / x - 1.0) / y;
            double dDet_dx = c1[1] / (-x) - c0[1] * (e2_comp_x / x - 1.0) / y;
            double dDet_dy = c0[0] * (e2_comp_x / x - 1.0) / y - c1[0] / (-x);
            if (mdet < 0.0f) {
                mdet = -mdet;
                dDet_dx = -dDet_dx;
                dDet_dy = -dDet_dy;
            }
            res[coord * v.handle().idx()] += (dtraTa_dx * mdet - traTa * dDet_dx)/ (mdet * mdet);
            res[coord * v.handle().idx() + 1] += (dtraTa_dy * mdet - traTa * dDet_dy)/ (mdet * mdet);
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

        double x = E1.norm();
        double e2_comp_x = OpenMesh::dot(E1, E2) / x;
        double y = sqrtf(E2.norm() * E2.norm() - e2_comp_x * e2_comp_x);

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
        double mdet = c0[0] * c1[1] - c0[1] * c1[0];
        if (mdet < 0.0f)
            mdet = -mdet;
        double traTa = OpenMesh::dot(c0, c0) + OpenMesh::dot(c1, c1);
        double ene = traTa / mdet;
        
        if (std::isinf(ene)) {
            std::cout << "stop";
        }

        E += traTa / mdet;
    }
    
    return E;
}

double energy(const Mesh &mesh, const Eigen::VectorXd& vecs) {
    double E = 0.0;
    for (auto face: mesh.all_faces()) {
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

        double x = E1.norm();
        double e2_comp_x = OpenMesh::dot(E1, E2) / x;
        double y = sqrtf(E2.norm() * E2.norm() - e2_comp_x * e2_comp_x);

        // The Matrix A = ((p1 - p0) / x, ((p2 - p0) - (p1 - p0) * e2_comp_x / x) / y)
        // calculate the row vector of Matrix A
        constexpr int coord = 2;
        const OpenMesh::Vec2f& p0 = OpenMesh::Vec2f(vecs[coord * handles[0]], vecs[coord * handles[0] + 1]);
        const OpenMesh::Vec2f& p1 = OpenMesh::Vec2f(vecs[coord * handles[1]], vecs[coord * handles[1] + 1]);
        const OpenMesh::Vec2f& p2 = OpenMesh::Vec2f(vecs[coord * handles[2]], vecs[coord * handles[2] + 1]);
        const OpenMesh::Vec2f e0 = p1 - p0;
        const OpenMesh::Vec2f e1 = p2 - p0;
        const OpenMesh::Vec2f c0 = (e0 / x);
        const OpenMesh::Vec2f c1 = (e1 - e0 * e2_comp_x / x) / y;
        // E(A) = trace(AA') / det(A)
        // calculate the \partial{E(A)} / \partial{p0.x} and \partial{E(A)}/ \partial{p0.y}
        double mdet = c0[0] * c1[1] - c0[1] * c1[0];
        if (mdet < 0.0f)
            mdet = -mdet;
        double traTa = OpenMesh::dot(c0, c0) + OpenMesh::dot(c1, c1);
        double ene = traTa / mdet;
        
        if (std::isinf(ene) || std::isnan(ene)) {
            return std::numeric_limits<double>().infinity();
        }

        E += traTa / mdet;
    }
    
    return E;
}

void gradientDescent(const Mesh& mesh, std::vector<OpenMesh::Vec2f>& init, double step = 0.0000001f) {
    constexpr double threshold = 0.000005f;
    auto res = init;
    auto norm = [](const decltype(init) vecs) {
        return std::accumulate(vecs.begin(), vecs.end(), 0.0f, [](double accum,  const OpenMesh::Vec2f& e){ return accum + OpenMesh::dot(e,e);});
    };
    int op_step = 0;
    std::vector<OpenMesh::Vec2f> grad;
    grad = calcGrad(mesh, init);
    std::ofstream grado("grad.csv");
    grado << "x,y\n";
    for (auto v: grad)
        grado << v[0] << "," << v[1] << "\n";
    double en0 = energy(mesh, init);
    do {
        std::cout << op_step++ << " Energy: " << en0 << std::endl;
        grad = calcGrad(mesh, init);
        for (int i = 0; i < init.size(); ++i)
            init[i] -= step * grad[i];
        double en1 = energy(mesh, init);
        if (abs(en1 - en0) < threshold) 
            return ;
        en0 = en1;
    } while(true);

    return ;
}

// choose the step length satisfying the Wolfe condition
double enei;
double getStepLength(double alpha0, const Mesh& mesh, const Eigen::VectorXd& params, const Eigen::VectorXd& p, const Eigen::VectorXd& grad) {
    constexpr double c1 = 1e-4;
    double ene0 = energy(mesh, params);
    enei = energy(mesh, params + alpha0 * p);
    double dphi = p.dot(grad);
    while (enei > ene0) {
        alpha0 /= 2.0;
        enei = energy(mesh, params + alpha0 * p);
    }
    double inite = enei, inite0 = ene0;
    double init = alpha0;
    if (enei < ene0 + c1 * alpha0 * dphi)
        return  alpha0;
    double alpha = - dphi * alpha0 * alpha0 / (2.0f * (enei - ene0 - dphi * alpha0));
    double enei_0 = enei; // energy_{i-1}
    enei = energy(mesh, params + alpha * p);
    std::cout << alpha << " " << enei << std::endl;
    while (!(enei < ene0 + c1 * alpha * dphi)) {
        // calculate the coeff of the cubic interpolation
        double dim0 = enei - ene0 - dphi * alpha;
        double dim1 = enei_0 - ene0 - dphi * alpha0;
        double numerator = 1.0f / (alpha0 * alpha0 * alpha * alpha * (alpha - alpha0));

        double a = (alpha0 * alpha0 * dim0 - alpha * alpha * dim1) * numerator;
        double b = (-alpha0 * alpha0 * alpha0 * dim0 + alpha * alpha * alpha * dim1) * numerator;
        // calculate the aplha
        alpha0 = alpha;
        alpha = -b + sqrtf(b * b - 3 * a *dphi) / (3 * a);
        // update 
        enei_0 = enei;
        enei = energy(mesh, params + alpha * p);
    }
    return alpha;
}

// !Todo
// Use Quasi-Newton method to minimize the energy
void quasiNewton(Mesh& mesh, std::vector<OpenMesh::Vec2f> &vecs, double error = 0.0005) {
    const size_t dim = vecs.size() * 2;
    Eigen::VectorXd x(dim), x0(dim), gradx(dim), gradx0(dim), sk(dim), yk(dim), p(dim);
    Eigen::MatrixXd Bk(dim, dim);
    size_t step = 0;
    // initialize the data
    double step_len = 1e-5f;
    Bk.setIdentity();
    for (int i = 0; i < vecs.size(); ++i) {
        x0[2 * i] = vecs[i][0];
        x0[2 * i + 1] = vecs[i][1];
    }
    gradx0 = calcGrad(mesh, x0);

    do {
        p = -Bk * gradx0;
        step_len = getStepLength(step_len, mesh, x0, p, gradx0);
        sk = step_len * p;
        x = x0 + sk;
        gradx = calcGrad(mesh, x);
        yk = gradx - gradx0;
        // update the H0
        std::cout << "updating" << Bk.size()  << " " << vecs.size() << std::endl;
        if (step == 0) {
           double entry = yk.dot(sk) / yk.dot(yk); 
           for (int i = 0; i < vecs.size(); ++i)
                Bk.coeffRef(i,i) = entry;
        }
        // update Bk
        std::cout << "updating2" << std::endl;
        auto skTB_k = sk.transpose() * Bk;
        auto num = (skTB_k * sk);
        Bk = Bk - Bk * sk * skTB_k / num  + (yk * yk.transpose()) / (yk.dot(sk));
        std::swap(x, x0);
        std::swap(gradx, gradx0);
        ++step;
        std::cout << "Steps: " << step << " " << "energy: " << enei << " Length: " << step_len << std::endl;
    } while (gradx0.norm() > error);

    // send it back to the caller
    for (int i = 0; i < vecs.size(); ++i) {
         vecs[i][0] = x0[2 * i];
         vecs[i][1] = x0[2 * i + 1];
    }
    return ;
}

std::vector<OpenMesh::Vec2f> paratimization(Mesh& mesh, double error) {
    mesh.triangulate();
    auto init_param = init(mesh); // the initial parameterization for the mesh
    std::cout << "Initialization complete.\n";
    // gradientDescent(mesh, init_param);
    quasiNewton(mesh, init_param);
    std::cout << "Optimization complete.\n";
    return std::move(init_param);
}
