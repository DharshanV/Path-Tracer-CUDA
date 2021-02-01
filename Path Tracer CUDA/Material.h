#pragma once
#include "vec3.h"

class Material {
public:
    __host__ __device__
    Material() : albedoColor(1, 0, 0), diffuseColor(0), specularExpo(2) {}
    __host__ __device__
    Material(const Material& m)
        : albedoColor(m.albedoColor),
        diffuseColor(m.diffuseColor),
        specularExpo(m.specularExpo) {
    }
    __host__ __device__
    Material(const vec3& a, const vec3& color, float specular)
        : albedoColor(a), diffuseColor(color), specularExpo(specular) {
    }
    __host__ __device__
    void setDiffuse(const vec3& diffuseColor) { this->diffuseColor = diffuseColor; }
    __host__ __device__
    void setAlbedo(const vec3& albedoColor) { this->albedoColor = albedoColor; }
    __host__ __device__
    void setSpecular(const float& specular) { this->specularExpo = specular; }

    vec3 diffuseColor;
    vec3 albedoColor;
    float specularExpo;
};