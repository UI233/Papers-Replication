#include <memory>
#include <fstream>
#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <ctime>
#include <cmath>
#include "CUDA_compute.h"
#include <miscellaneous.h>
#define CUDA_FUNC __host__ __device__

using namespace glm;

static constexpr float PI = 3.1415926f;
static constexpr float rho0 = 1000.0f;
static constexpr float time_step = 0.001f;

__host__ void outputCSV(const std::string &path, PointInfo *points_set, ParticleType type, size_t points_num) {
    std::ofstream out(path);
    
    out << "x,y,z,rho,vx,vy,vz\n";
    for (int i = 0; i < points_num; ++i) 
        if (points_set[i].type == type) {
            vec3 pos = points_set[i].pos;
            float rho = points_set[i].rho;
            vec3 v = points_set[i].v;
            out << pos.x << "," << pos.y << "," << pos.z << "," << rho << "," << v.x << "," << v.y << "," << v.z << "\n";
        }
}

__host__ void outputVTK(ParticleManager &pm, int idx) {
    if (idx % 10 == 0) {
        int i = idx / 10;
        std::ofstream out("cubes" + std::to_string(i) + ".vtu");
        out << "# vtk DataFile Version 2.0" << std::endl
            << "Unstructured Grid Example" << std::endl
            << "ASCII" << std::endl
            << "DATASET UNSTRUCTURED_GRID" << std::endl;
        unsigned int size = pm.p_fluid.size();

        out << "POINTS " << size << " float" << std::endl;

        for (auto &flow : pm.p_fluid) {
            auto p = flow.p;
            out << p.x << " " << p.y << " " << p.z << " ";
        }

        /*for (auto &flow : pm.p_solid) {
            auto p = flow.p;
            out << p.x << " " << p.y << " " << p.z << " ";
        }*/

        out << std::endl;
        out << "CELLS " << size << " " << 2 * size << std::endl;
        for (int i = 0; i < size; ++i)
            out << 1 << " " << i << std::endl;
        out << "CELL_TYPES" << " " << size << std::endl;
        for (int i = 0; i < size; ++i)
            out << 1 << std::endl;
        out << "POINT_DATA " << size << std::endl;
        out << "SCALARS scalars float 1" << std::endl
            << "LOOKUP_TABLE default" << std::endl;
        for (auto &flow : pm.p_fluid) {
            auto p = flow.rho;
            out << p << " ";
        }

        //for (auto &flow : pm.p_solid) {
        //    auto p = flow.rho;
        //    out << p << " ";
        //}

        out << std::endl;
    }
}

CUDA_FUNC  bool less_msb(const unsigned int &x, const unsigned int &y) {
    return x < y && (x < (x ^ y));
}

__constant__ vec3 bias;
__constant__ float m;
__constant__ float h2;

CUDA_FUNC bool pointComp(const PointInfo &lhs, const PointInfo &rhs) {
//    unsigned int ida[3] = { (lhs.pos.x - bias.x) / h ,  (lhs.pos.y - bias.y) / h, (lhs.pos.z - bias.z) / h},
//        idb[3] = { (rhs.pos.x - bias.x) / h ,  (rhs.pos.y - bias.y) / h , (rhs.pos.z - bias.z) / h  };
     
   /* int msd = 0;
    for (int i = 1; i < 3; ++i)
        if (less_msb(ida[msd] ^ idb[msd], ida[i] ^ idb[i]))
            msd = i;
    return ida[msd] < idb[msd];*/

    ivec3 ida((lhs.pos - bias) / h2), idb((rhs.pos - bias) / h2);
    //:printf("%f\n", h2);
    return ida.x == idb.x ? (ida.y == idb.y ? ida.z < idb.z : ida.y < idb.y) : ida.x < idb.x;
}

inline __forceinline__ CUDA_FUNC float W(const vec3 &x, float h) {
    h *= 2.0f;
    float r = length(x);
    float h_9 = h * h * h * h * h * h * h * h * h;
    if (r >= h)
        return 0.0f;
    return (365.0f / (64.0f * PI * h_9)) * (h * h - r * r)  * (h * h - r * r) * (h * h - r * r);
}


using ull = unsigned long long;
__device__ ull computeZOrder(ivec3 idx, size_t sz) {
    ull res = 0;
    ull mask = 1u;
    
    for (size_t i = 0; i < sz; ++i) {
        size_t bits = 2u * i;
        res |= ((idx.x & (mask << i)) << (bits));
        res |= ((idx.y & (mask << i)) << (bits + 1));
        res |= ((idx.z & (mask << i)) << (bits + 2));
    }
    return res;
}

__global__ void getAverageDistance(PointInfo *points, size_t points_num, float *sum) {
    int point_idx = (gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x) * blockDim.x * blockDim.y * blockDim.z + threadIdx.z *
        blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

    if (point_idx >= points_num)
        return;

    PointInfo point = points[point_idx];

    float dis = 0.0f;
    for (int i = 0; i < points_num; ++i)
        dis += length(points[i].pos - point.pos);
    
    atomicAdd(sum, dis / points_num);
}

__global__ void constructCellList(Cell *cells, PointInfo *points, size_t point_num, size_t sz) {
    int point_idx = (gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x) * blockDim.x + threadIdx.x;
    if (point_idx >= point_num)
        return ;
    vec3 pos = points[point_idx].pos;
    ivec3 idx = ivec3((pos - bias) / h2);
    size_t cell_idx = computeZOrder(idx, sz);
    //if (cell_idx >= (1u << (3 * sz)))
        //printf("%d: (%d, %d, %d), (%f, %f, %f)\n", point_idx, idx.x, idx.y, idx.z, pos.x, pos.y, pos.z);
    atomicMax(&cells[cell_idx].max, point_idx);
    atomicAdd(&cells[cell_idx].count, 1);
}

__global__ void updateFluidDensity(Cell *cells, PointInfo *points, size_t points_num, size_t sz) {
    int point_idx = (gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x) * blockDim.x * blockDim.y * blockDim.z + threadIdx.z *
        blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    if (point_idx >= points_num)
        return;

    PointInfo point = points[point_idx];
    ivec3 idx = ivec3((point.pos - bias) / h2);
    ivec3 now;
    Cell cell;
    size_t dim = 1u << sz;

    if (point.type == FLUID) {
        point.rho = 0.0f;
        int cnt = 0;
        for (int i = -1; i <= 1; ++i)
            for (int j = -1; j <= 1; ++j)
                for (int k = -1; k <= 1; ++k) {
                    now = idx + ivec3(i, j, k);
                    if (now.x >= 0 && now.x < dim && now.y >= 0 && now.y < dim && now.z >= 0 && now.z < dim) {
                        cell = cells[computeZOrder(now, sz)];
                        //cnt += cell.count;
                        for (int st = cell.max; st != cell.max - cell.count; st--) {
                            PointInfo neighbor = points[st];
                            float w = W(neighbor.pos - point.pos, h2 / 2.0f);
                            point.rho += w * m;
                        }
                    }
                }
    }
    else point.rho = rho0;

    points[point_idx].rho = point.rho;
}

__global__ void updateSolidDensity(Cell *cells, PointInfo *points, size_t points_num, size_t sz) {
    int point_idx = (gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x) * blockDim.x * blockDim.y * blockDim.z + threadIdx.z *
        blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    if (point_idx >= points_num)
        return;

    PointInfo point = points[point_idx];
    ivec3 idx = ivec3((point.pos - bias) / h2);
    ivec3 now;
    Cell cell;
    size_t dim = 1u << sz;

    if (point.type == SOLID) {
        float mindis = 1e30f;
        for (int i = -1; i <= 1; ++i)
            for (int j = -1; j <= 1; ++j)
                for (int k = -1; k <= 1; ++k) {
                    now = idx + ivec3(i, j, k);
                    if (now.x >= 0 && now.x < dim && now.y >= 0 && now.y < dim && now.z >= 0 && now.z < dim) {
                        cell = cells[computeZOrder(now, sz)];
                        for (int st = cell.max; st != cell.max - cell.count; st--) {
                            PointInfo neighbor = points[st];
                            float dis = length(neighbor.pos - point.pos);
                            if (dis < mindis && neighbor.type == FLUID) {
                                point.rho = neighbor.rho;
                                mindis = dis;
                            }
                        }
                    }
                }
    }

    float t = point.rho / rho0;
    point.pre = 2000.0f * ((t * t * t * t) * (t * t) * t - 1.0f);
    points[point_idx] = point;
}

__global__ void updatePressure(Cell *cells, PointInfo *points, size_t points_num, size_t sz) {
    int point_idx = (gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x) * blockDim.x * blockDim.y * blockDim.z + threadIdx.z *
        blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    if (point_idx >= points_num)
        return;

    PointInfo point = points[point_idx];
    ivec3 idx = (ivec3)((point.pos -  bias) / h2);
    ivec3 now;
    point.gradp = vec3(0.0f);
    Cell cell;
    size_t dim = 1u << sz;
    PointInfo neighbor;
    //vec3 gradw, delta;

    if (point.type == FLUID) {
        for (int i = -1; i <= 1; ++i)
            for (int j = -1; j <= 1; ++j)
                for (int k = -1; k <= 1; ++k) {
                    now = idx + ivec3(i, j, k);
                    auto cell_idx = computeZOrder(now, sz);
                    if (cell_idx >= 0 && cell_idx < dim * dim * dim) {
                        cell = cells[cell_idx];
                        for (int st = cell.max; st != cell.max - cell.count; st--) {
                            //if (st < 0 || st >= points_num)
                            //   printf("%d ", st);
                            neighbor = points[st];

                            vec3 gradw(0.0f);
                            float r = length(point.pos - neighbor.pos);
                            if (r != 0.0f && r < h2) {
                                float factor = 45.0f / ( PI * h2 * h2 * h2 * h2 * h2 * h2 );
                                gradw = -factor * (h2  - r) * (h2 - r) *  (point.pos - neighbor.pos) / r;
                            }
                            point.gradp -= m * m * (point.pre / (point.rho * point.rho) + neighbor.pre / (neighbor.rho * neighbor.rho)) * gradw;
                            //if (isnan(point.gradp.x))
                            //    printf("%f %f %d %d\n", point.rho, neighbor.rho, point.type, neighbor.type);
                        }
                    }
                }

        // udpate the Velocity
        // point.v += time_step * (vec3(0.0f, 0.0f, -9.8f) + point.gradp / point.rho);
        point.v += time_step * (vec3(0.0f, 0.0f, -9.8f) + point.gradp );
    }

    points[point_idx] = point;
}

__global__ void updateSolidVelocity(Cell *cells, PointInfo *points, size_t point_num, size_t sz) {
    int point_idx = (gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x) * blockDim.x * blockDim.y * blockDim.z + threadIdx.z *
        blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    if (point_idx >= point_num)
         return;
    PointInfo point = points[point_idx];
    ivec3 idx = (ivec3)((point.pos - bias) / h2);
    ivec3 now;
    Cell cell;
    float mind = 1e30f;
    size_t dim = 1u << sz;

    if (point.type == SOLID) {
        for (int i = -1; i <= 1; ++i)
            for (int j = -1; j <= 1; ++j)
                for (int k = -1; k <= 1; ++k) {
                    now = idx + ivec3(i, j, k);
                    if (now.x >= 0 && now.x < dim && now.y >= 0 && now.y < dim && now.z >= 0 && now.z < dim) {
                        cell = cells[computeZOrder(now, sz)];
                        for (int st = cell.max; st != cell.max - cell.count; st--) {
                            PointInfo neighbor = points[st];

                            if (neighbor.type == FLUID) {
                                float d = length(neighbor.pos - point.pos);
                                if (d < mind) {
                                    point.old_v = neighbor.v - dot(neighbor.v, point.normal) * point.normal + dot(point.v, point.normal) * point.normal;
                                    mind = d;
                                }
                            }
                        }
                    }
                }
    }
    else point.old_v = point.v;
    points[point_idx] = point;
}

__global__ void addViscocity(Cell *cells, PointInfo *points, size_t point_num, size_t sz) {
    constexpr float miu = 3.5f;
    int point_idx = (gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x) * blockDim.x * blockDim.y * blockDim.z + threadIdx.z *
        blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
     if (point_idx >= point_num)
         return;
    PointInfo point = points[point_idx];
    ivec3 idx = (ivec3)((point.pos -  bias) / h2);
    ivec3 now;
    Cell cell;
    size_t dim = 1u << sz;

    if (point.type == FLUID) {
        for (int i = -1; i <= 1; ++i)
            for (int j = -1; j <= 1; ++j)
                for (int k = -1; k <= 1; ++k) {
                    now = idx + ivec3(i, j, k);
                    if (now.x >= 0 && now.x < dim && now.y >= 0 && now.y < dim && now.z >= 0 && now.z < dim) {
                        cell = cells[computeZOrder(now, sz)];
                        for (int st = cell.max; st != cell.max - cell.count; st--) {
                            PointInfo neighbor = points[st];
                            if (st != point_idx) {
                                point.v += 0.5f * m / (neighbor.rho)  * W(neighbor.pos - point.pos, h2 / 2.0f) * (neighbor.old_v - point.old_v);
                                // float W = 45.0f / (PI * h2 * h2 * h2 * h2 * h2 * h2) * (h2 - length(neighbor.pos - point.pos));
                                // point.v += time_step * W * miu * (neighbor.old_v - point.old_v) / (neighbor.rho * point.rho);
                            }
                        }
                    }
                }
        points[point_idx] = point;
    }
}

__global__ void updateAirVelocity(Cell *cells, PointInfo *points, size_t point_num, size_t sz) {
    int point_idx = (gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x) * blockDim.x * blockDim.y * blockDim.z + threadIdx.z *
        blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
     if (point_idx >= point_num)
         return;
    PointInfo point = points[point_idx];
    ivec3 idx = (ivec3)((point.pos - bias) / h2);
    ivec3 now;
    Cell cell;
    float mind = 1e30f;

    if (point.type == AIR) {
        for (int i = -1; i <= 1; ++i)
            for (int j = -1; j <= 1; ++j)
                for (int k = -1; k <= 1; ++k) {
                    now = idx + ivec3(i, j, k);
                    cell = cells[computeZOrder(now, sz)];
                    for (int st = cell.max; st != cell.max - cell.count; st--) {
                        PointInfo neighbor = points[st];
                        if (neighbor.type == FLUID) {
                            float d = length(neighbor.pos - point.pos);
                            if (d < mind) {
                                point.v = neighbor.v;
                                mind = d;
                            }
                        }
                    }
                }
        points[point_idx].v = point.v;
    }
}

__global__ void updatePosition(PointInfo *points, size_t point_num, vec3 *min, vec3 *max) {
    int point_idx = (gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x) * blockDim.x * blockDim.y * blockDim.z + threadIdx.z *
        blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    if (point_idx >= point_num)
        return ;

    vec3 pos = points[point_idx].pos;
    vec3 v = points[point_idx].v;
    pos += time_step * v;
    points[point_idx].pos = pos;
    /*if (length(points[point_idx].v) > 4.0f) {
        auto &gradp = points[point_idx].gradp;
        printf("p: %f %f %f %f %d\n", gradp.x, gradp.y, gradp.z, points[point_idx].rho, points[point_idx].type);
    }*/

    atomicMinFloat(&min->x, pos.x);
    atomicMinFloat(&min->y, pos.y);
    atomicMinFloat(&min->z, pos.z);

    atomicMaxFloat(&max->x, pos.x);
    atomicMaxFloat(&max->y, pos.y);
    atomicMaxFloat(&max->z, pos.z);
}

__host__ cudaError updateVelocity(Cell *cells, PointInfo * points, size_t point_num, size_t sz) {
    //int point_idx = (gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x) * blockDim.x + threadIdx.x;
    dim3 grid_dim(point_num / (4 * 4 * 4) + 1), block_dim(4, 4, 4);
    updateSolidVelocity<<<grid_dim, block_dim>>>(cells, points, point_num, sz);
    auto error = cudaDeviceSynchronize();
    addViscocity<<<grid_dim, block_dim>>>(cells, points, point_num,  sz );
    error = cudaDeviceSynchronize();
    updateAirVelocity<<<grid_dim, block_dim>>>(cells, points, point_num, sz );
    error = cudaDeviceSynchronize();

    return error;
}

__host__ cudaError updateDensity(Cell *cells, PointInfo * points, size_t points_num, size_t sz) {
    dim3 grid_dim(points_num / (4 * 4 * 4) + 1), block_dim(4, 4, 4);
    updateFluidDensity <<<grid_dim, block_dim >>> (cells, points, points_num, sz );
    auto error = cudaDeviceSynchronize();
    updateSolidDensity <<<grid_dim, block_dim >>> (cells, points, points_num, sz );
    error = cudaDeviceSynchronize();
    return error;
}

__host__ PointInfo* cudaCompute(ParticleManager &pm, size_t step) {
    const vec3 pos_inf(INFINITY, INFINITY, INFINITY), neg_inf(-INFINITY, -INFINITY, -INFINITY);
    // get basic data
    float h = pm.getH();

    float *h_gpu;
    cudaMalloc(&h_gpu, sizeof(float));

    PointInfo *points;
    Cell *cells;
    size_t points_num = pm.p_air.size() + pm.p_fluid.size() + pm.p_solid.size();
    size_t capacity = points_num;
    auto AABB = pm.getAABB();
    ivec3 bound = (AABB.second - AABB.first) / (2.0f * h);
    size_t max_num = std::max(bound.x, std::max(bound.y, bound.z));
    // vec3 bias = AABB.first;
    float _one = 1.0f;
    cudaMemcpyToSymbol(bias, &AABB.first, sizeof(vec3));
    cudaMemcpyToSymbol(m, &_one, sizeof(float));
    vec3 *min, *max;
    cudaMalloc(&min, sizeof(vec3));
    cudaMalloc(&max, sizeof(vec3));

    // sort the points
    float h2_cpu = 2 * h;
    cudaMemcpyToSymbol(h2, &h2_cpu, sizeof(float));

    size_t sz = 0;

    while (max_num) {
        ++sz;
        max_num >>= 1u;
    }

    size_t cell_num = (1u << (3u * sz));
    std::unique_ptr<PointInfo[]> point_set(new PointInfo[points_num]);
    // Initialize the Particle information
    {
        int num = 0;
        for (auto &point : pm.p_air) {
            point_set[num].type = AIR;
            point_set[num].pos = point.p;
            point_set[num].rho = rho0;
            point_set[num].v = vec3(0.0f);
            ++num;
        }

        for (auto &point : pm.p_fluid) {
            point_set[num].type = FLUID;
            point_set[num].pos = point.p;
            point_set[num].rho = point.rho;
            point_set[num].v = vec3(0.0f);
            ++num;
        }        

        for (auto &point : pm.p_solid) {
            point_set[num].type = SOLID;
            point_set[num].pos = point.p;
            point_set[num].rho = point.rho;
            point_set[num].v = vec3(0.0f);
            point_set[num].normal = point.normal;
            ++num;
        }    
    }
    auto error = cudaMalloc(&points, sizeof(PointInfo) * points_num);
    error = cudaMemcpy(points, point_set.get(), sizeof(PointInfo) * points_num, cudaMemcpyHostToDevice);
    error = cudaMalloc(&cells, sizeof(Cell) * cell_num);

    for (size_t i = 0; i < step; ++i) {
        auto st = clock(); 
        cudaMemcpy(min, &pos_inf, sizeof(vec3), cudaMemcpyHostToDevice);
        cudaMemcpy(max, &neg_inf, sizeof(vec3), cudaMemcpyHostToDevice);
        error = cudaMemset(cells, 0, sizeof(Cell) * cell_num);
        // This is problemetic, the index strategy does not work  
        thrust::device_ptr<PointInfo> dev_points(points);
        thrust::sort(dev_points, dev_points + points_num, [=] __device__(const PointInfo &a, const PointInfo &b) {
            return pointComp(a, b);
        });
        error = cudaDeviceSynchronize();
        // cudaMemcpy(point_set.get(), points, sizeof(PointInfo) * points_num, cudaMemcpyDeviceToHost);
        
        constructCellList <<<dim3(points_num / 256 + 1), 256 >>> (cells, points, points_num, sz );
        error = cudaDeviceSynchronize();

        updateDensity(cells, points, points_num, sz );
        if (i == 0) {
            error = cudaMemcpy(point_set.get(), points, sizeof(PointInfo) * points_num, cudaMemcpyDeviceToHost);
            int fluid_num = 0;
            float density_sum = 0.0f;
            for (int i = 0; i < points_num; ++i) {
                if (point_set[i].type == FLUID) {
                    ++fluid_num;
                    density_sum += point_set[i].rho;
                }
            }

            float ave_density = density_sum / fluid_num;
            float ave_m = rho0 / ave_density;
            cudaMemcpyToSymbol(m, &ave_m, sizeof(float));
            updateDensity(cells, points, points_num, sz );
        }
        dim3 grid_dim(points_num / (4 * 4 * 4) + 1), block_dim(4, 4, 4);
        updatePressure << <grid_dim, block_dim >> > (cells, points, points_num, sz );
        updateVelocity(cells, points, points_num, sz );
        error = cudaDeviceSynchronize();
        updatePosition <<<grid_dim, block_dim >>> (points, points_num, min, max);
        error = cudaDeviceSynchronize();

        cudaMemcpyToSymbol(bias, min, sizeof(vec3), 0, cudaMemcpyDeviceToDevice);
        cudaMemcpy(&AABB.first, min, sizeof(vec3), cudaMemcpyDeviceToHost);
        cudaMemcpy(&AABB.second, max, sizeof(vec3), cudaMemcpyDeviceToHost);
        bound = (AABB.second - AABB.first) / (2.0f * h);
        max_num = std::max(bound.x, std::max(bound.y, bound.z));

        auto ed = clock();
        std::cout << "Step: " << i << " Time: " << ed - st << " Error: " << error << std::endl;


        error = cudaMemcpy(point_set.get(), points, sizeof(PointInfo) * points_num, cudaMemcpyDeviceToHost);

        auto vi = [](int i) {
            return vec3(0.0f);
        };
        for (int j = 0; j < points_num; ++j) {
                auto &point = point_set[j];
                if (point.type == FLUID) {
                    for (auto &level : pm.boundary)
                        if (level->operator()(point.pos) < 0.0f)
                            point.pos = level->project(point.pos);
                }
                else if (point.type == SOLID)
                    point.v = vi(i);
        }

        // Recalculation the AABB box for the scene
        size_t sz_before = sz;
        sz = 0;
        while (max_num) {
            ++sz;
            max_num >>= 1u;
        }

        if (sz > 9) {
            cudaMemset(h_gpu, 0, sizeof(float));

            getAverageDistance << <grid_dim, block_dim >> > (points, points_num, h_gpu);
            error = cudaDeviceSynchronize();

            float h_cpu;
            cudaMemcpy(&h_cpu, h_gpu, sizeof(float), cudaMemcpyDeviceToHost);
            h2_cpu = 2 * h_cpu / points_num;
            cudaMemcpyToSymbol(h2, &h2_cpu, sizeof(float));

            bound = (AABB.second - AABB.first) / (2.0f * h);
            max_num = std::max(bound.x, std::max(bound.y, bound.z));
            sz = 0;
            while (max_num) {
                ++sz;
                max_num >>= 1u;
            }
        }

        if (sz > sz_before) {
            cell_num = (1u << (3u * sz));
            cudaFree(cells);

            error = cudaMalloc(&cells, sizeof(Cell) * cell_num);
            error = cudaMemset(cells, 0, sizeof(Cell) * cell_num);
        }

        if (i % 5 == 0)
            outputCSV("water" + std::to_string(i / 5) + ".csv", point_set.get(), FLUID, points_num);
        if (i % 5 == 0)
            outputCSV("air" + std::to_string(i / 5) + ".csv", point_set.get(), AIR, points_num);
        if (i % 5 == 0)
            outputCSV("solid" + std::to_string(i / 5) + ".csv", point_set.get(), SOLID, points_num);
        if (i % 100 == 0) {
            pm.p_fluid.clear();
            pm.p_air.clear(); 
            
            for (int i = 0; i < points_num; ++i)
                if (point_set[i].type == FLUID)
                    pm.p_fluid.push_back(Particle<FLUID>(point_set[i].v, point_set[i].pos));
            pm.reSample(true);
            // Initialize the particles
            {
                size_t num = 0;
                if (pm.p_air.size() + pm.p_solid.size() + pm.p_fluid.size() > capacity) {
                    capacity = pm.p_air.size() + pm.p_solid.size() + pm.p_fluid.size();
                    point_set.reset(new PointInfo[capacity]);
                    cudaFree(points);
                    cudaMalloc(&points, capacity * sizeof(PointInfo));
                }

                for (auto &point : pm.p_solid) {
                    point_set[num].type = SOLID;
                    point_set[num].pos = point.p;
                    point_set[num].rho = point.rho;
                    point_set[num].v = vi(i);
                    point_set[num].normal = point.normal;
                    ++num;
                }

                for (auto &point : pm.p_fluid) {
                    point_set[num].type = FLUID;
                    point_set[num].pos = point.p;
                    point_set[num].rho = point.rho;
                    point_set[num].v = point.v;
                    ++num;
                }

                for (auto &point : pm.p_air) {
                    point_set[num].type = AIR;
                    point_set[num].pos = point.p;
                    point_set[num].rho = rho0;
                    ++num;
                } 

                points_num = num;
            }

            // cudaMemcpy(bias_ptr, &bias, sizeof(vec3), cudaMemcpyHostToDevice);
        }
        error = cudaMemcpy(points, point_set.get(), points_num * sizeof(PointInfo), cudaMemcpyHostToDevice);
    }
    cudaFree(cells);
    cudaFree(points);
    return nullptr;
}