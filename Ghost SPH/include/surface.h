#ifndef SURFACE_H
#define SURFACE_H
#include <glm/glm.hpp>
#include <array>
#include <utility>
using namespace glm;
class Surface {
public:
    virtual ~Surface() {};
    virtual float operator()(const vec3 &point) const = 0;
    virtual vec3 project(const vec3 &point) const = 0;
    virtual vec3 getRandomTan(const vec3 &p) const = 0;
    virtual vec3 getNormal(const vec3 &p) const = 0;
    virtual std::pair<vec3, vec3> boundingBox() const = 0;
};

class Sphere : public Surface {
public:
    Sphere() : origin(), r() {}
    Sphere(const vec3 &t_origin, const float &t_r) : origin(t_origin), r(t_r) {}
    float operator()(const vec3 &point) const final;
    vec3 project(const vec3 &point) const final;
    vec3 getRandomTan(const vec3 &p) const final;
    vec3 getNormal(const vec3 &p) const ;
    std::pair<vec3, vec3> boundingBox() const;
private:
    vec3 origin;
    float r;
};

class Triangle : public Surface {
public:
    Triangle() : a(), b(), c() {}
    Triangle(const vec3 &t_a, const vec3 &t_b, const vec3 &t_c): a(t_a), b(t_b), c(t_c){
        normal = normalize(cross(b - a, c - a));
    }
    float operator()(const vec3 &point) const final;
    vec3 project(const vec3 &point) const final;
    vec3 getRandomTan(const vec3 &p) const final;
    vec3 getNormal(const vec3 &p) const;
    std::pair<vec3, vec3> boundingBox() const;
private:
    // Presume that vertices are stored in CCW
    vec3 a, b, c;
    vec3 normal;
};

class Cube : public Surface {
public:
    Cube() {}
    Cube(vec3 o, vec3 t_x, vec3 t_y, vec3 t_z) :
        origin(o), x(t_x), y(t_y), z(t_z), transform(t_x, t_y, t_z) {
        transform = inverse(transpose(transform));
    }
    float operator()(const vec3 &point) const final;
    vec3 project(const vec3 &point) const final;
    vec3 getRandomTan(const vec3 &p) const final;
    vec3 getNormal(const vec3 &p) const;
    std::pair<vec3, vec3> boundingBox() const;
private:
    std::pair<int, float> getNearestFace(const vec3 &point) const;
    std::array<float, 6> getDistances(const vec3 &point) const;
    vec3 origin;
    vec3 x, y, z;
    mat3 transform;
};
#endif // !SURFACE_H
