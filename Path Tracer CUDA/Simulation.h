#pragma once
#include "Particles.h"
#include <vector>

#define CLOCK_CORRECTION 0.5f

class Simulation {
public:
    Simulation(int width,int height) : width(width), height(height) {
    }

    void update() {
        simulate_particles();
    }

    void addParticle(const Particles& p) {
        particles.push_back(p);
    }

    std::vector<glm::vec3> getPositions() {
        std::vector<glm::vec3> temp;
        for (Particles& p : particles) {
            temp.push_back(p.position);
        }
        return temp;
    }

private:
    std::vector<Particles> particles;

    void simulate_particles() {
        for (int i = 0; i < (particles.size()); i++) {
            particles[i].processMotion(1 + particles.size() * CLOCK_CORRECTION);
            check_limits(particles[i]);
            detect_collisions(particles[i], i);
        }
    }

    void check_limits(Particles& t) // checks for collisions with the walls
    {
        if (t.position.x >= (width - t.radius)) // check for right Boundary
        {
            t.position.x = width - t.radius - 1;
            t.velocity.x *= -1; // update direction
            t.velocity.x *= FLOOR_FRICTION; // account for inelastic collision
        } else if (t.position.x <= t.radius) // check for left Boundary
        {
            t.position.x = t.radius + 1;
            t.velocity.x *= -1; // update direction
            t.velocity.x *= FLOOR_FRICTION; // account for inelastic collision
        }

        if (t.position.y <= t.radius) {
            t.position.y = t.radius + 1;
            t.velocity.y *= -1;
            t.velocity.y *= FLOOR_FRICTION;
        } else if (t.position.y >= (height - t.radius)) {
            t.position.y = height - t.radius - 1;
            t.velocity.y *= -1;
            t.velocity.y *= FLOOR_FRICTION;
        }
    }

    void detect_collisions(Particles& t, int self_index) {
        for (int i = 0; i < particles.size(); i++)
        {
            if (i != self_index) {
                float d_bw_centres = glm::length(particles[i].position - t.position);
                if (d_bw_centres <= 2 * t.radius) {
                    glm::vec3 tempVelocity = t.velocity;
                    t.velocity = particles[i].velocity;
                    particles[i].velocity = tempVelocity;
                }
            }
        }
    }

    int width;
    int height;
};