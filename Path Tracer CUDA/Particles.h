#pragma once
#include <glm/glm.hpp>

#define WALL_FRICTION 0.5f
#define FLOOR_FRICTION 0.8f
#define CEIL_FRICTION 0.3f
#define GRAVITY -12.0f
#define D_TIME 0.003f
#define FORCE_MULTIPLIER 0.2f

class Particles {
public:
    glm::vec3 position;
    glm::vec3 velocity;
    glm::vec3 displacement;
    float radius;

    float clock_correction = 1;

    Particles(glm::vec3 position, glm::vec3 velocity, int radius = 1.0f) : position(position),
        velocity(velocity), radius(radius) {
        displacement = glm::vec3(0);
    }

    void processMotion(float cc) {
        clock_correction = cc;

        displacement.x = velocity.x * D_TIME;
        position.x += displacement.x * clock_correction;

        displacement.y = velocity.y * D_TIME - 0.5 * (GRAVITY)*powf(D_TIME, 2);
        velocity.y = velocity.y - (GRAVITY) * (D_TIME);

        position.y -= displacement.y * clock_correction;
    }

    void impartForce(int Force, int Force_angle_degree) {
        Force *= FORCE_MULTIPLIER;
        velocity.x = Force * cosf(Force_angle_degree * 3.14159265f / 180.0f);
        velocity.y = Force * sinf(Force_angle_degree * 3.14159265f / 180.0f);
    }
};