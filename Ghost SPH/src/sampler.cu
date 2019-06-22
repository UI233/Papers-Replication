#include "sampler.h"
#include <utility>
#include <random>
#include <map>
#include <ctime>
#include <cmath>
#include <queue>
using namespace glm;
static constexpr float PI = 3.1415926535897932384626433832795f;
static constexpr float PI_2 = 1.5707963267948966192313216916398f;

vec3 sampleUnitSphere() {
    static std::default_random_engine eng(time(NULL));
    static std::uniform_real_distribution<float> dis(0, 1);
    
    float u = 2 * PI * dis(eng);
    float v = acos(dis(eng) * 2.0f - 1.0f) ;

    return vec3(sin(v) * cos(u), sin(v) * sin(u), cos(v));
}

void Grid::insert2Samples(const vec3 &point, const float &r) {
    samples.insert(std::make_pair(point / r, point));
}

// hash = x * (lz * ly) + y * ly + z
float Grid::getNearest(const vec3 &point, const ivec3& v, const float &r) const {
    float dmin = 2000000.0f;
    const int depth = 1;
    for(int i = -depth; i <= depth; ++i)
        for(int j = -depth; j <= depth; ++j)
            for(int k = -depth; k <= depth; ++k) {
                    auto inc = ivec3(i, j, k);
                    auto p = samples.equal_range(v + inc);
                    for (auto itr = p.first; itr != p.second; ++itr) {
                        dmin = std::min(dmin, length(itr->second - point));
                        if (itr ->second != point)
                            if (dmin <= r)
                                return dmin;
                    }
                    // if (res != samples.end())
                    //    dmin = std::min(dmin, length(res->second - point));
                }
    return dmin;
}

bool Grid::checkPoissonCriterion(const vec3 &sample, const ivec3 &v, const float &r) const {
    if (samples.count(v))
        return false;
    auto a = getNearest(sample, v, r);
    return  a > r;
}

std::vector<vec3> Grid::sampleSurface(const LevelSet &f, float r, int t, float e) {
    static std::default_random_engine gen(time(NULL));
    static std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    vec3 p;
    int hit = 0;
    std::vector<vec3> res;
    // Iterate over the grid
    for(int x = lx; x <= rx; ++x)
        for(int y = ly; y <= ry; ++y)
            for(int z = lz; z <= rz; ++z) {
                Cube cell;
                int cnt = 0;
                unsigned int i = cell.size();
                cell[0] = vec3(len * x, len * y, len * z);
                float sign = f(cell[0]);
                for(int ix = 0; ix <= 1; ++ix)
                    for(int iy = 0; iy <= 1; ++iy)
                        for (int iz = 0; iz <= 1; ++iz) {
                            auto ele = vec3(len * x, len * y, len * z) + len * vec3(ix, iy, iz);
                            cell[cnt] = ele;
                            if (f(cell[cnt]) * sign <= 0.0f) {
                                i = cnt;
                                break;
                            }
                            ++cnt;
                        }
                if(i < cell.size()) { // level set change sign in this grid
                    bool found = false;
                    for (int j = 0; j < t && !found; ++j) {
                        p = cell[0] + dis(gen) * (cell[1] - cell[0]) + dis(gen) * (cell[3] - cell[0])
                            + dis(gen) * (cell[4] - cell[0]);
                        p = f.project(p);
                        ivec3 coord = p / r;
                        found = checkPoissonCriterion(p, coord, r);
                    }

                    if (found) {
                        ++hit;
                        insert2Samples(p,r);
                        do {
                            found = false;
                            p = f.project(p + e * r * f.getRandomTan(p));
                            ivec3 coord = p / r;
                            if (checkPoissonCriterion(p, coord, r)){
                                insert2Samples(p, r);
                                found = true;
                            }
                        } while (found);
                        found = false;
                    }
                }
            }
    relaxSamples([](vec3) {return true;}, f, r, 5, t);
    for (auto &sample : samples)
        res.push_back(sample.second);
    samples.clear();
    std::cout << "hit: " << hit << std::endl;
    return std::move(res);
}

std::vector<vec3> Grid::sampleVolume(Judger g, const LevelSet &f, std::vector<vec3> &temp, const float &r, const int &k) {
    static std::default_random_engine eng(time(NULL));
    std::uniform_real_distribution<float> dis(r, 2.0f * r);
    static std::default_random_engine engi(time(NULL));
    static std::uniform_int_distribution<unsigned int> disi;

    std::vector<vec3> res;
    if (temp.empty())
        return res;
    for (auto &p : temp) {
        insert2Samples(p, r);
    }

    for (int i = disi(engi) % temp.size(); !temp.empty(); i = disi(engi) % temp.size()) {
        bool found = false;
        vec3 st = temp[i];
        for (int j = 0; j < k; ++j) {
            auto sample = st + dis(eng) * sampleUnitSphere();
            ivec3 v = sample / r;
            if (f(sample) <= 0.0 && g(sample) && checkPoissonCriterion(sample, v, r)) {
                found = true;
                st = sample;
                insert2Samples(sample, r);
                temp.push_back(sample);
                res.push_back(sample);
            }
        }

        if (!found) {
            std::swap(temp[i], temp[temp.size() - 1]);
            temp.pop_back();
        }

        if (temp.empty())
            break;
    }
    samples.clear();
    for (auto &entry : res) {
        insert2Samples(entry, r);
    }
    res.clear();
    relaxSamples(g, f, r, k, 5, false);
    for(auto &sample : samples)
        res.push_back(sample.second);
    samples.clear();
    return std::move(res);
}

void Grid::relaxSamples(Judger g, const LevelSet &f, float r, int k, int t, bool is_surface) {
    std::vector<vec3> ball;
    ball.reserve(8);

    for (int i = 0; i < k; ++i) {
        for (auto &sample : samples) {
            ball.clear();
            // need to handle
            ivec3 coord = sample.second / r;
            float d = 2 * r;
            int depth = 2;
            for (int i = -depth; i <= depth; ++i)
                for (int j = -depth; j <= depth; ++j)
                    for (int k = -depth; k <= depth; ++k)
                        if  (i || j || k) {
                            auto res = samples.find(coord + ivec3(i, j, k));
                            if (res != samples.end()) {
                                float len = length(res->second - sample.second);
                                if (len < 2 * r) {
                                    d = std::min(d, len);
                                    ball.push_back(res->second);
                                }
                            }
                        }
            vec3 p_new = sample.second;
            for (int i = 0; i < t; ++i) {
                float tau = static_cast<float>(t - i) / t;
                vec3 cand = sample.second + tau * r * sampleUnitSphere();
                if (f(cand) > 0.0f || is_surface) 
                    cand = f.project(cand);

                float d_ = 2.0 * r;
                for (auto &other : ball)
                    d_ = std::min(d_, length(other - cand));
                if (d_ > d && g(cand)) {
                    p_new =  cand;
                    d = d_;
                }
            }
            sample.second = p_new;
        }
    }
    return;
}