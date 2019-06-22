#ifndef COMPUTE_H
#define COMPUTE_H
#include <glm/glm.hpp>
#include "particle.h"
using namespace glm;
void computeDensity(ParticleManager &particles, const float &h);
void computePressure(ParticleManager &particles, const float &h);
void computeVelocity(ParticleManager &particles, const float &h);
void computePosition(ParticleManager &particles);
float getH(ParticleManager &particles);

#endif // !COMPUTE_H
