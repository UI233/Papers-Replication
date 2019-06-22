#ifndef MISCELLANEOUS_H
#define MISCELLANEOUS_H
#include "glm/glm.hpp"
#include <iostream>
#include <string>
#include <device_atomic_functions.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>

namespace std {
    template<>
    class hash<glm::ivec3>{
    public:
        using argument_type = glm::ivec3;
        using result_type = long long;
        result_type operator()(argument_type const& s) const{
            return (s.x << 31LL) + (s.y << 12LL) + s.z;
        }
    };

    template<>
    class less<glm::ivec3> {
    public:
        bool operator()(const glm::ivec3 &left, const glm::ivec3 &right) const {
            return left.x == right.x ? (
                left.y == left.y ? left.z < right.z : left.y < right.y
                ) : left.x < right.x;
        }
    };
}

__device__ __forceinline__ float atomicMinFloat(float * addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMin((int *)addr, __float_as_int(value))) :
        __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));

    return old;
}

__device__ __forceinline__ float atomicMaxFloat(float * addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
        __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));

    return old;
}
#endif // !MISCELLANEOUS_H
