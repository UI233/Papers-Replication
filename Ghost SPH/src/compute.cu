#include "compute.h"
#include <iostream>

static const float time_step = 0.001f;
static const float PI = 3.1415926;

std::ostream& operator << (std::ostream &out, const vec3 &vec) {
    return out << "(" << vec.x << " " << vec.y << " " << vec.z << ")" << std::endl;
}

// Kernel Function
float W(vec3 x, float h) {
    h *= 2.0f;
    float r = length(x);
    float h_9 = h * h * h * h * h * h * h * h * h;
    if (r >= h)
        return 0.0f;
    return (365.0f / (64.0f * PI * h_9)) * (h * h - r * r)  * (h * h - r * r) * (h * h - r * r);
}

vec3 gradW(vec3 x, float h) {
    if (x == vec3(0.0f))
        return vec3(0.0f);
    float r = length(x);
    h *= 2.0f;
    float h_9 = h * h * h * h * h * h * h * h * h;
    float factor = 945.0f / (32.0f * PI * h_9);
    if (r >= h)
        return vec3(0.0f);
    return -factor * (h * h - r * r) * normalize(x);
}

float getH(ParticleManager &particles) {
    double sum = 0.0;
    std::vector<Particle<FLUID>> temp;
    for (auto &flow : particles.fluid)
        temp.push_back(flow.second);
    for (decltype(temp)::size_type i = 0; i < temp.size(); ++i) {
        for (decltype(temp)::size_type j = i + 1; j < temp.size(); ++j)
            sum += length(temp[i].p - temp[j].p);
        ivec3 coord = temp[i].p / particles.r;
    }
    sum /= temp.size() * (temp.size() - 1) / 2;
    return sum;
}

void computeDensity(ParticleManager &particles, const float &h) {
    const int depth(2.0f * particles.getH() / particles.r + 1);
    // Update the density for fluid
    for (auto &flow : particles.fluid) {
        float sum = 0.0f;
        auto &pi = flow.second;
        for (int i = -depth; i <= depth; ++i)
            for (int j = -depth; j <= depth; ++j)
                for (int k = -depth; k <= depth; ++k) {
                    ivec3 inc(i, j, k);
                    ivec3 coord = ivec3(pi.p / particles.r) + inc;
                    auto p0 = particles.fluid.equal_range(coord);
                    for (auto itr = p0.first; itr != p0.second; ++itr) {
                        auto &par = itr->second;
                        sum += W(flow.second.p - par.p, h) * par.m;
                    }
                    auto p1 = particles.air.equal_range(coord);
                    for (auto itr = p1.first; itr != p1.second; ++itr) {
                        auto &par = itr->second;
                        sum += W(flow.second.p - par.p, h) * par.m;
                    }
                    auto p2 = particles.solid.equal_range(coord);
                    for (auto itr = p2.first; itr != p2.second; ++itr) {
                        auto &par = itr->second;
                        sum += W(flow.second.p - par.p, h) * par.m;
                    }
                }
        flow.second.rho = sum;
    }

    // Update the density for solid
    for (auto &soli : particles.solid) {
        auto &pi = soli.second;
        float d = 10000000.0f;
        for (int i = -depth; i <= depth; ++i)
            for (int j = -depth; j <= depth; ++j)
                for (int k = -depth; k <= depth; ++k) {
                    ivec3 inc(i, j, k);
                    ivec3 coord = ivec3(pi.p / particles.r) + inc;
                    auto p0 = particles.fluid.equal_range(coord);
                    for (auto itr = p0.first; itr != p0.second; ++itr)
                        if (length(soli.second.p - itr->second.p) < d) {
                            soli.second.rho = itr->second.rho;
                            d = length(soli.second.p - itr->second.p);
                        }
                }
    }
}

void computePressure(ParticleManager &particles, const float &h) {
    static const float rho0 = 1000;
    const int depth(2 * particles.getH() / particles.r + 1);
    // Update the pressure for fluid particles
    for (auto &flow : particles.fluid) {
        auto t = flow.second.rho / rho0;
        flow.second.pre = 2000.0f * ((t * t * t * t) * (t * t) * t - 1);
    }

    // Update the pressure for solid
    for (auto &soli : particles.solid) {
        auto t = soli.second.rho / rho0;
        soli.second.pre = 2000.0f * ((t * t * t * t) * (t * t) * t - 1);
    }
    // Update the pressure for air
    for (auto &air : particles.air) {
        auto t = air.second.rho / rho0;
        air.second.pre = 2000.0f * ((t * t * t * t) * (t * t) * t - 1);
    }
    // Update grad(pressure) for liquid and air particles
    int cnt = 0;
    for (auto &flow : particles.fluid) {
        auto& gradp = flow.second.gradpre;
        auto& pi = flow.second;

        gradp = vec3(0.0f);
        for (int i = -depth; i <= depth; ++i) {
            for (int j = -depth; j <= depth; ++j) {
                for (int k = -depth; k <= depth; ++k) {
                    ivec3 inc(i, j, k);
                    ivec3 coord = ivec3(pi.p / particles.getLen()) + inc;
                    auto p0 = particles.fluid.equal_range(coord);
                    for (auto itr = p0.first; itr != p0.second; ++itr) {
                        auto &pj = itr->second;
                        gradp -= pj.m * (pi.pre / (pi.rho * pi.rho) + pj.pre / (pj.rho * pj.rho))  * gradW(pi.p - pj.p, h);
                        if (length(pi.p - pj.p) < 2 * h)
                            ++cnt;
                    }

                    auto p1 = particles.solid.equal_range(coord);
                    for (auto itr = p1.first; itr != p1.second; ++itr) {
                        auto &pj = itr->second;
                        gradp -= pj.m * (pi.pre / (pi.rho * pi.rho) + pj.pre / (pj.rho * pj.rho))  * gradW(pi.p - pj.p, h);
                        if (length(pi.p - pj.p) < 2 * h)
                            ++cnt;
                    }

                    auto p2 = particles.air.equal_range(coord);
                    for (auto itr = p2.first; itr != p2.second; ++itr) {
                        auto &pj = itr->second;
                        gradp -= pj.m * (pi.pre / (pi.rho * pi.rho) + pj.pre / (pj.rho * pj.rho))  * gradW(pi.p - pj.p, h);
                        if (length(pi.p - pj.p) < 2 * h)
                            ++cnt;
                    }
                }
            }
        }
        gradp *= pi.m;
    }
}

void computeVelocity(ParticleManager &particles, const float &h) {
    const int depth(2.0f * particles.getH() / particles.r + 1);
    static const vec3 gravity(0.0f, 0.0f, 0.0f);
    static const float epsilon = 0.5f;
    // Update the velocity for liquid
    for (auto &flow : particles.fluid) {
        auto &pi = flow.second;
        pi.v += time_step * (gravity + pi.gradpre / pi.rho);
    }
    // Update the velocity for solid particles
    float d;
    for (auto &soli : particles.solid) {
        d = 10000000.0f;
        auto& pi = soli.second;
        for (int i = -depth; i <= depth; ++i) {
            for (int j = -depth; j <= depth; ++j) {
                for (int k = -depth; k <= depth; ++k) {
                    ivec3 inc(i, j, k);
                    ivec3 coord = ivec3(pi.p / particles.getLen()) + inc;
                    auto p0 = particles.fluid.equal_range(coord);
                    for (auto itr = p0.first; itr != p0.second; ++itr) {
                        auto &par = itr->second;
                        if (length(par.p - soli.second.p) < d) {
                            soli.second.v = par.v - dot(par.v, soli.second.normal) * soli.second.normal;
                            d = length(par.p - soli.second.p);
                        }
                    }
                }
            }
        }
    }

    // Apply viscosity to liquid particle
    auto liq = particles.fluid;
    for (auto &flow : particles.fluid) {
        auto &pi = flow.second;

        for (int i = -depth; i <= depth; ++i)
            for (int j = -depth; j <= depth; ++j)
                for (int k = -depth; k <= depth; ++k) {
                    ivec3 inc(i, j, k);
                    ivec3 coord = ivec3(pi.p / particles.r) + inc;
                    auto p0 = liq.equal_range(coord);
                    for (auto itr = p0.first; itr != p0.second; ++itr) {
                        auto &pj = itr->second;
                        auto temp = epsilon * pj.m / (pj.rho)  * W(pi.p - pj.p, h) * (pj.v - pi.v);
                        pi.v += temp;
                    }

                    auto p1 = particles.solid.equal_range(coord);
                    for (auto itr = p1.first; itr != p1.second; ++itr) {
                        auto &pj = itr->second;
                        auto temp = epsilon * pj.m / (pj.rho)  * W(pi.p - pj.p, h) * (pj.v - pi.v);
                        pi.v += temp;
                    }
                }
    }
    // Update the velocity for air paticle
    for (auto &air : particles.air) {
        auto &pi = air.second;
        d = 10000000.0f;
        for (int i = -depth; i <= depth; ++i)
            for (int j = -depth; j <= depth; ++j)
                for (int k = -depth; k <= depth; ++k) {
                    ivec3 inc(i, j, k);
                    ivec3 coord = ivec3(pi.p / particles.getLen()) + inc;
                    auto p0 = particles.fluid.equal_range(coord);
                    for (auto itr = p0.first; itr != p0.second; ++itr)
                        if (length(air.second.p - itr->second.p) < d) {
                            air.second.v = itr->second.v;
                            d = length(air.second.p - itr->second.p);
                        }
                }
    }
}

void computePosition(ParticleManager &particles) {
    for (auto &particle : particles.fluid) {
        auto p0 = particle.second.p;
        auto p1 = p0 + time_step * particle.second.v;
        particle.second.p = p1;
        for (auto &bound : particles.boundary) {
            vec3 diff = p1 - p0;
            vec3 temp = p1;
            if (bound->operator()(temp) < 0.0f) {
                temp = bound->project(temp);
            }
            particle.second.p = temp;
        }
    }
    auto temp = std::move(particles.fluid);
    for (auto &par : temp)
        particles.fluid.insert(std::make_pair(par.second.p / particles.getLen(), par.second));

    for (auto &particle : particles.air)
        particle.second.p += time_step * particle.second.v;
}
