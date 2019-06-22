#ifndef SAMPLER_H
#define SAMPLER_H
#include <array>
#include <glm/glm.hpp>
#include <functional>
#include <vector>
#include <unordered_map>
#include <memory>
#include "surface.h"
#include "miscellaneous.h"

using namespace glm;
class Grid {
public:
    using Cube = std::array<vec3, 8>;
    using LevelSet = Surface;
    using Judger = std::function<bool(vec3)>;
    Grid(int t_lx = 0, int t_ly = 0, int t_lz = 0, int t_rx = 0, int t_ry = 0, int t_rz = 0, float t_len = 0.0f) : 
        lx(t_lx), ly(t_ly), lz(t_lz), rx(t_rx), ry(t_ry), rz(t_rz), len(t_len) {}
private:
    std::unordered_multimap<ivec3, vec3> samples;
    float len;
    int lx, ly, lz;
    int rx, ry, rz;
    bool checkPoissonCriterion(const vec3 &sample, const ivec3 &v, const float &r) const;
    float getNearest(const vec3 &point, const ivec3& hash, const float &r) const;
    void insert2Samples(const vec3 &point, const float &r);
public:
    std::vector<vec3> sampleSurface(const LevelSet &f, float r, int t, float e) ;
    std::vector<vec3> sampleVolume(Judger g, const LevelSet &f,  std::vector<vec3> &seeds, const float &r, const int &k) ;
    void relaxSamples(Judger g, const LevelSet &f, float r, int s, int t, bool is_surface = true) ;
};
#endif // !SAMPLER_H
