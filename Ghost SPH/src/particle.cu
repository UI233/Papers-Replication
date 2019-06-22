#include "particle.h"
#include "compute.h"
#include <functional>
#include <algorithm>
static constexpr float rho0 = 1000;
static constexpr float rhol = 1000;
static constexpr float rhos = 787;
bool checkAir(const vec3 &sample, const ParticleManager &a) {
    ivec3 coord = sample / a.len;
    float da = 2 * a.h + 0.0000000001f;
    float dw = 2 * a.h + 0.0000000001f;
    float ds = 2 * a.h + 0.0000000001f;
    int depth = da / a.len;
    for (int i = -depth; i <= depth; i++)
        for (int j = -depth; j <= depth; j++)
            for (int k = -depth; k <= depth; k++) {
                ivec3 inc = ivec3(i, j, k);
                auto p0 = a.fluid.equal_range(coord + inc);
                for (auto i = p0.first; i != p0.second; ++i)
                    dw = std::min(dw, length(i->second.p - sample));

                auto p1 = a.solid.equal_range(coord + inc);
                for (auto i = p1.first; i != p1.second; ++i)
                    ds = std::min(ds, length(i->second.p - sample));

                auto p2 = a.air.equal_range(coord + inc);
                for (auto i = p2.first; i != p2.second; ++i)
                    da = std::min(da, length(i->second.p - sample));
            }
    bool poisson = da > a.r && ds > a.r && dw > a.r;
    return poisson && dw < 2 * a.h;
}

bool checkSolid(const vec3 &sample, const ParticleManager &a, const Surface &f) {
    ivec3 coord = sample / a.len;
    float d = 2 * a.r;
    int depth = a.r / a.len + 1;
    for (int i = -depth; i <= depth; i++)
        for (int j = -depth; j <= depth; j++)
            for (int k = -depth; k <= depth; k++) {
                ivec3 inc = ivec3(i, j, k);
                auto p0 = a.fluid.equal_range(coord + inc);
                for (auto i = p0.first; i != p0.second; ++i)
                    d = std::min(d, length(i->second.p - sample));

                auto p1 = a.solid.equal_range(coord + inc);
                for (auto i = p1.first; i != p1.second; ++i)
                    d = std::min(d, length(i->second.p - sample));

                auto p2 = a.air.equal_range(coord + inc);
                for (auto i = p2.first; i != p2.second; ++i)
                    d = std::min(d, length(i->second.p - sample));
            }
    return d > a.r && f(sample) <= 0.0f && f(sample) > -2 * a.h;
}

void ParticleManager::initSamples(const std::vector<surface_ptr> &init_fluid, const std::vector<surface_ptr> &init_solid) {
    static auto merge_box = [](const std::pair<vec3, vec3> &a, const std::pair<vec3, vec3> &b) {
        return std::make_pair(
            vec3(std::min(a.first.x, b.first.x), std::min(a.first.y, b.first.y), std::min(a.first.z, b.first.z)),
            vec3(std::max(a.second.x, b.second.x), std::max(a.second.y, b.second.y), std::max(a.second.z, b.second.z)));
    };

    std::pair<vec3, vec3> box;
    if (!init_fluid.empty())
        box = init_fluid[0]->boundingBox();
    else if (!init_solid.empty())
        box = init_solid[0]->boundingBox();

    for (auto &fluid : init_fluid)
        box = merge_box(box, fluid->boundingBox());
    for (auto &solid : init_solid)
        box = merge_box(box, solid->boundingBox());
    grid = Grid(box.first.x / len - 2, box.first.y / len - 2, box.first.z / len - 2, box.second.x / len + 2, box.second.y / len + 2, box.second.z / len + 2, len);
    std::vector<vec3> temp, res, normal;


    std::cout << "Sample fluid:\n";
    // Sample fluid 
    auto ret_true = [](const vec3 &) {return true; };
    for (auto &flow : init_fluid) {
        water.push_back(flow);
        res = grid.sampleSurface(*flow, r, 20, 1.0845);
        for (auto &entry : res)
            fluid.insert(std::make_pair(ivec3(entry / len), Particle<FLUID>(vec3(0.0f), entry)));
        temp = res;
        res = grid.sampleVolume([](vec3) {return true; }, *flow, res, r, 8);
        for (auto &entry : res)
            fluid.insert(std::make_pair(ivec3(entry / len), Particle<FLUID>(vec3(0.0f), entry)));
    }

    std::cout << "Number: " << fluid.size() << std::endl;

    temp.clear();
    solid.clear();
    int cnt = 0;
    int sz = 0;
    for (auto &soli : init_solid) {
        boundary.push_back(soli);
        std::cout << "Sample solid" + std::to_string(cnt++);
        std::cout << std::endl;
        res = grid.sampleSurface(*soli, r, 8, 1.084);
        temp = res;
        normal.clear();
        for (auto& res : temp)
            normal.push_back(soli->getNormal(res));
        for (int i = 0; i < temp.size(); i++) {
            ivec3 coord = temp[i] / len;
            solid.insert(std::make_pair(coord, Particle<SOLID>(vec3(0.0f), temp[i], m, normal[i])));
        }
        normal.clear();
        std::vector<vec3> seeds = res;
        auto judger = [&](vec3 v) {return checkSolid(v, *this, *soli); };
        auto samples = grid.sampleVolume(judger, *soli, seeds, r, 5);
        for (auto &res : samples)
            normal.push_back(soli->getNormal(res));
        for (int i = 0; i < samples.size(); i++) {
            ivec3 coord = samples[i] / len;
            solid.insert(std::make_pair(coord, Particle<SOLID>(vec3(0.0f), samples[i], m, normal[i])));
        }
        std::cout << "Number: " << solid.size() - sz << std::endl;
        sz = solid.size();
    }

    std::cout << "Sample air:\n";
    // Sample the air particles
    reSample(false);
    std::cout << "Number: " << air.size() << std::endl;
    // Normalize the mass of particles
    computeDensity(*this, h);
    float sum = 0.0f;
    for (auto &flow : fluid)
        sum += flow.second.rho;
    sum /= fluid.size();
    
    for (auto &flow : fluid) {
        flow.second.m *= rho0 / sum;
        m = flow.second.m;
    }

    p_solid.reserve(solid.size());
    p_air.reserve(air.size());
    p_fluid.reserve(fluid.size());

    for (auto &air : air) {
        air.second.m = m;
        p_air.push_back(air.second);
    }

    for (auto &soli : solid) {
        soli.second.m = m;
        p_solid.push_back(soli.second);
    }

    // computeDensity(*this, h);
    for (auto &flu : fluid) {
        flu.second.m = m;
        p_fluid.push_back(flu.second);
    }

    //solid.clear();
    air.clear();
    fluid.clear();
    return;
}

void ParticleManager::reSample(bool computing) {
    if (computing) {
        for (auto &point : p_fluid) {
            fluid.insert(std::make_pair(point.p / len, point));
        }
    }

    std::vector<vec3> seeds;
    std::vector<vec3> temp;
    air.clear();
    Sphere sp = Sphere(vec3(0.0), 100000000000000.0f);
    for (auto &flow : fluid) {
        int sum = 0;
        for (int i = -1; i <= 1; ++i)
            for (int j = -1; j <= 1; ++j)
                for (int k = -1; k <= 1; ++k) {
                    ivec3 coord = flow.first + ivec3(i, j, k);
                    sum += fluid.count(coord);
                }
        if (sum >= 4 && sum < 18)
            seeds.push_back(flow.second.p);
    }
    auto judger = std::bind(checkAir, std::placeholders::_1, *this);
    auto res = grid.sampleVolume(judger, sp, seeds, r, 5);
    for (auto &entry : res)
        air.insert(std::make_pair(ivec3(entry / len), Particle<AIR>(vec3(0), entry, m)));
    if (res.size() == 0) {
        Sphere sp = Sphere(vec3(0.0), 100000000000000.0f);
        for (auto &flow : fluid) {
            seeds.push_back(flow.second.p);
        }
        auto judger = std::bind(checkAir, std::placeholders::_1, *this);
        auto res = grid.sampleVolume(judger, sp, seeds, r, 5);
    }

    if (computing) {
        fluid.clear();
        p_air.clear();
        for (auto &point : air) {
            point.second.m = m;
            p_air.push_back(point.second);
        }
        air.clear();
    }

    return;
}

std::pair<vec3, vec3> ParticleManager::getAABB() const {
    vec3 min(INFINITY, INFINITY, INFINITY), max(-INFINITY, -INFINITY, -INFINITY);

    for (auto &point : p_air) {
        min.x = std::min(min.x, point.p.x);
        min.y = std::min(min.y, point.p.y);
        min.z = std::min(min.z, point.p.z);

        max.x = std::max(max.x, point.p.x);
        max.y = std::max(max.y, point.p.y);
        max.z = std::max(max.z, point.p.z);
    }

    for (auto &point : p_solid) {
        min.x = std::min(min.x, point.p.x);
        min.y = std::min(min.y, point.p.y);
        min.z = std::min(min.z, point.p.z);

        max.x = std::max(max.x, point.p.x);
        max.y = std::max(max.y, point.p.y);
        max.z = std::max(max.z, point.p.z);
    }

    for (auto &point : p_fluid) {
        min.x = std::min(min.x, point.p.x);
        min.y = std::min(min.y, point.p.y);
        min.z = std::min(min.z, point.p.z);

        max.x = std::max(max.x, point.p.x);
        max.y = std::max(max.y, point.p.y);
        max.z = std::max(max.z, point.p.z);
    }

    return std::make_pair(min, max);
}
