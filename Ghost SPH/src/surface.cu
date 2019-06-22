#include "Surface.h"
#include <ctime>
#include <random>
#include <cmath>
using namespace glm;
constexpr float PI = 3.1415926;

vec3 sampleTan(const vec3 &normal) {
    static std::default_random_engine eng(time(NULL));
    static std::uniform_real_distribution<float> dis(0, 1);
    vec3 u = normalize(cross(normal, vec3(0.2f, 0.1f, 0.5f)));
    vec3 v = normalize(cross(u, normal));
    float angle = 2.0f * PI * dis(eng);

    return sin(angle) * u + cos(angle) * v;
}

float Sphere::operator()(const vec3 &point) const  {
    return length(point - origin)  - r;
}

vec3 Sphere::project(const vec3 &point) const {
    auto np = normalize(point - origin);
    return origin + np * r;
}

vec3 Sphere::getRandomTan(const vec3 &p) const {
    return sampleTan(normalize(p - origin));
}

vec3 Sphere::getNormal(const vec3 &p) const {
    return normalize(p - origin);
}

std::pair<vec3, vec3> Sphere::boundingBox() const {
    return std::make_pair(
        origin - vec3(r),
        origin + vec3(r)
    );
}

float Triangle::operator()(const vec3 &point) const {
    return dot(normal, point - a);
}

vec3 Triangle::project(const vec3 &point) const {
    auto res = point - normal * dot(normal, point - a);
    vec3 s0 = cross(a - res, b - res), s1 = cross(b - res, c - res) ,s2 = cross(c - res, a - res);
    float d0 = dot(s0, normal), d1 = dot(s1, normal), d2 = dot(s2, normal);
    // To check whether this point is within the triangle
    if (d0 >= 0 && d1 >= 0 && d2 >= 0)
        return res;
    else {
        if (length(res - a) < length(res - b))
        {
            if (length(res - c) < length(res - a))
                return c;
            else return a;
        }
        else {
            if (length(res - c) < length(res - b))
                return c;
            else return b;
        }
    }
}

vec3 Triangle::getRandomTan(const vec3 &p) const {
    return sampleTan(normal);
}

vec3 Triangle::getNormal(const vec3 &p) const {
    return normal;
}

std::pair<vec3, vec3> Triangle::boundingBox() const {
    return std::make_pair(
        vec3(std::min(a.x, std::min(b.x, c.x)), std::min(a.y, std::min(b.y, c.y)), std::min(a.z, std::min(b.z, c.z))),
        vec3(std::max(a.x, std::max(b.x, c.x)), std::max(a.y, std::max(b.y, c.y)), std::max(a.z, std::max(b.z, c.z)))
    );
}

std::array<float, 6> Cube::getDistances(const vec3 &point) const {
    static const auto computePlaneDis = [](const vec3 &point,const vec3 &origin, const vec3 &normal) {
        return dot(normal, point - origin);
    };

    float l[3]{ length(y), length(z), length(x) };
    vec3 normal[3]{ normalize(cross(x, z)), normalize(cross(y, x)), normalize(cross(z, y))};
    std::array<float, 6> d;
    for (int i = 0; i < 3; ++i)
        d[i] = computePlaneDis(point, origin, normal[i]);
    for(int i = 3; i < 6; ++i)
        d[i] =  -l[i - 3] - d[i - 3]; 
    return std::move(d);
}

std::pair<int, float> Cube::getNearestFace(const vec3 &point) const {
    auto d = getDistances(point);
    int idx = 0;
    for (int i = 0; i < 6; ++i)
        if (fabs(d[i]) < fabs(d[idx])) 
            idx = i;
    return std::make_pair(idx, d[idx]);
}

float Cube::operator()(const vec3 &point) const {
    /*auto d = getDistances(point);
    bool neg = true;
    for (int i = 0; i < 6 && neg; ++i)
        neg = (d[i] <= 0.0);
    if (neg)
        return getNearestFace(point).second;
    else {
        float res = 0.0f;
        for (auto &dis : d)
            if (dis >= 0.0f)
                res += dis * dis;
        return sqrt(res);
    }*/

    auto within = [](float v) -> bool {
        return v >= 0.0f && v <= 1.0f;
    };

    auto nearest = [](float v) -> float {
        return v < 0.5f ? v : 1.0f - v;
    };
    vec3 coord = transform * (point - origin);

    float dis = 0.0f;
    vec3 now = coord.x * x + coord.y * y + coord.z * z + origin;
    vec3 center = origin + 0.5f * x + 0.5f * y + 0.5f * z;

    if (within(coord.x) && within(coord.y) && within(coord.z)) {
        float dx = nearest(coord.x) * x.length();
        float dy = nearest(coord.y) * y.length();
        float dz = nearest(coord.z) * z.length();
        return - std::min(dx, std::min(dy, dz));
    }

    float temp;
    if (coord.x < 0.0f) {
        temp = -coord.x * x.length();
        dis += temp * temp;
    }
    else if (coord.x > 1.0f) {
        temp = (coord.x - 1.0f) * x.length();
        dis += temp * temp;
    }

    if (coord.y < 0.0f) {
        temp = -coord.y * y.length();
        dis += temp * temp;
    }
    else if (coord.y > 1.0f) {
        temp = (coord.y - 1.0f) * y.length();
        dis += temp * temp;
    }

    if (coord.z < 0.0f) {
        temp = -coord.z * z.length();
        dis += temp * temp;
    }
    else if (coord.z > 1.0f) {
        temp = (coord.z - 1.0f) * z.length();
        dis += temp * temp;
    }

    return sqrt(dis);
}

vec3 Cube::project(const vec3 &point) const {
    auto clamp = [](float v, float min, float max) -> float{
        return v < min ? min : (v > max ? max : v);
    };

    vec3 coord = transform * (point - origin);
    coord.x = clamp(coord.x, 0.0f, 1.0f);
    coord.y = clamp(coord.y, 0.0f, 1.0f);
    coord.z = clamp(coord.z, 0.0f, 1.0f);
    if(coord.x == 0.0f || coord.x == 1.0f || coord.y == 0.0f || coord.y == 1.0f || coord.z == 0.0f || coord.z == 1.0f)
        return origin + coord.x * x + coord.y * y + coord.z * z;

    vec3 mdis(0.0f);
    mdis.x = x.length() * std::min(coord.x, 1.0f - coord.x);
    mdis.y = y.length() * std::min(coord.y, 1.0f - coord.y);
    mdis.z = z.length() * std::min(coord.z, 1.0f - coord.z);
    
    if (mdis.x <= mdis.y && mdis.x <= mdis.z) {
        if (coord.x < 0.5f)
            coord.x = 0.0f;
        else coord.x = 1.0f;
    }
    else if (mdis.y <= mdis.x && mdis.y <= mdis.z) {
        if (coord.y < 0.5f)
            coord.y = 0.0f;
        else coord.y = 1.0f;
    }
    else {
        if (coord.z < 0.5f)
            coord.z = 0.0f;
        else coord.z = 1.0f;
    }

    return origin + coord.x * x + coord.y * y + coord.z * z;
}

vec3 Cube::getRandomTan(const vec3 &point) const {
    return sampleTan(getNormal(point));
}

vec3 Cube::getNormal(const vec3 &point) const {
    auto p = getNearestFace(point);
    vec3 normal[3]{ normalize(cross(x, z)), normalize(cross(y, x)), normalize(cross(z, y))};
    if (p.first < 3)
        return normal[p.first];
    else return -normal[p.first - 3];
}

std::pair<vec3, vec3> Cube::boundingBox() const {
    vec3 min = origin, max = origin;

    for (int i = 0; i < 8; ++i) {
        auto point = origin;
        if (i & 1)
            point += x;
        if (i & 2)
            point += y;
        if (i & 4)
            point += z;
        min.x = std::min(min.x, point.x);
        min.y = std::min(min.y, point.y);
        min.z = std::min(min.z, point.z);

        max.x = std::max(max.x, point.x);
        max.y = std::max(max.y, point.y);
        max.z = std::max(max.z, point.z);
    }
    return std::make_pair(min, max);
}
