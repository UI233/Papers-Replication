#ifndef PARTICLE_H
#define PARTICLE_H
#include <glm/glm.hpp>
#include <string>
#include <memory>
#include <map>
#include "sampler.h"
using namespace glm;

enum ParticleType {
    FLUID,
    SOLID,
    AIR
};

template<ParticleType PT>
class Particle {
public:
    Particle(const vec3 &t_v, const vec3& t_p, const float &t_m = 1.0f, const float &t_rho = 1000.0f) :
        v(t_v), p(t_p), m(t_m), pre(0.0f), gradpre(0.0f), rho(t_rho) {}
    Particle() : v(0.0f), p(0.0f), rho(), pre(0.0f), gradpre(0.0f), m(1.0f) {}
    // represnet the velocity, position and density respectively
    vec3 v, p;
    float pre;
    vec3 gradpre;
    float rho;
    float m;
};

template<>
class Particle<SOLID> {
public:
    Particle(const vec3 &t_v, const vec3& t_p, const float &t_m = 1.0f, const vec3 &t_n = vec3(0.0f)) :
        v(t_v), p(t_p), m(t_m), pre(0.0f), gradpre(0.0f), normal(t_n), rho(1000.0f) {}
    Particle() : v(0.0f), p(0.0f), rho(1000.0f) ,pre(0.0f), gradpre(0.0f), m(1.0f) {}
    // represnet the velocity, position and density respectively
    vec3 v, p;
    float pre;
    vec3 gradpre;
    vec3 normal;
    float m;
    float rho;
};

class ParticleManager {
    using surface_ptr = std::shared_ptr<Surface>;
public:
    std::unordered_multimap<ivec3, Particle<FLUID>> fluid;
    std::unordered_multimap<ivec3, Particle<AIR>> air;
    std::unordered_multimap<ivec3, Particle<SOLID>> solid;
    ParticleManager(float t_len = 0.0f) : r(1.73205f * t_len), h(1.8826f * t_len),  len(t_len), m(1.0f) {}
    void initSamples(const std::vector<surface_ptr> &init_fluid , const std::vector<surface_ptr>& init_solid);
    void reSample(bool emission);
    inline float getLen() const { return len; }
    inline float getH() const { return h; }
    std::pair<vec3, vec3> getAABB() const;
    
    friend bool checkAir(const vec3 &sample, const ParticleManager &a);
    friend bool checkSolid(const vec3 &sample, const ParticleManager &a, const Surface &f);

    std::vector<Particle<FLUID>> p_fluid;
    std::vector<Particle<AIR>> p_air;
    std::vector<Particle<SOLID>> p_solid;

    std::vector<surface_ptr> boundary;
    std::vector<surface_ptr> water;
    float len, r, h;
private:
    unsigned int lx, ly, lz;
    Grid grid;
    float m;
};

bool checkAir(const vec3 &sample, const ParticleManager &a);
bool checkSolid(const vec3 &sample, const ParticleManager &a, const Surface &f);
#endif // !PATICLE_H
