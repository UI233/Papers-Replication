#ifndef CUDA_COMPUTE_H
#define CUDA_COMPUTE_H
#include <cuda_runtime.h>
#include "particle.h"
struct Cell {
    int max, count;
};

struct PointInfo {
    ParticleType type;
    float rho, pre;
    vec3 pos, gradp, normal, v, old_v;
};

PointInfo* cudaCompute(ParticleManager &pm, size_t step);
#endif // !CUDA_COMPUTE_H
