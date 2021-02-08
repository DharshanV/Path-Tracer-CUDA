#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <cuda_runtime.h>
#include <iostream>

#include "ObjLoader.h"

typedef struct Sphere {
    __host__ __device__
    Sphere(glm::vec3 p, float r = 1.0f) :
            pos(p), radius(r), matIndex(0) { }

    __device__
    inline glm::vec3 normal(const glm::vec3& hitPoint) {
        return glm::normalize(hitPoint - pos);
    }

    glm::vec3 pos;
    float radius;
    uint16_t matIndex;
} Sphere;

typedef struct Plane {
    __host__ __device__
    Plane(glm::vec3 pos = glm::vec3(0), glm::vec3 N = glm::vec3(0)) :
        pos(pos), N(N), matIndex(0) {
        N = glm::normalize(N);
    }

    glm::vec3 pos, N;
    uint16_t matIndex;
} Plane;

typedef struct Vertex {
    glm::vec3 position;
    glm::vec3 normals;
    glm::vec3 color;
    glm::vec2 texture;
} Vertex;

typedef struct Triangle {
    Vertex vertices[3];
    glm::vec3 N;
    uint16_t matIndex;
} Triangle;

typedef struct Model {
    Model() {}

    Model(const char* fileName) {
        tinyobj::ObjReaderConfig reader_config;
        tinyobj::ObjReader reader;

        if (!reader.ParseFromFile(fileName, reader_config)) {
            if (!reader.Error().empty()) {
                std::cerr << "TinyObjReader: " << reader.Error();
            }
            exit(1);
        }

        if (!reader.Warning().empty()) {
            std::cerr << "TinyObjReader: " << reader.Warning();
        }

        auto& attrib = reader.GetAttrib();
        auto& shapes = reader.GetShapes();
        auto& materials = reader.GetMaterials();

        for (size_t s = 0; s < shapes.size(); s++) {
            size_t index_offset = 0;
            for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
                int fv = shapes[s].mesh.num_face_vertices[f];
                if (fv != 3) {
                    std::cerr << "More than 3 face vertices. (Unsupported)" << std::endl;
                    exit(1);
                }

                Triangle triangle;
                for (size_t v = 0; v < fv; v++) {
                    tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

                    glm::vec3 pos(0.0f);
                    glm::vec3 norm(0.0f);
                    glm::vec2 tex(0.0f);

                    if (3 * idx.vertex_index < attrib.vertices.size()) {
                        pos[0] = attrib.vertices[3 * idx.vertex_index + 0];
                        pos[1] = attrib.vertices[3 * idx.vertex_index + 1];
                        pos[2] = attrib.vertices[3 * idx.vertex_index + 2];
                    }

                    if (3 * idx.normal_index < attrib.normals.size()) {
                        norm[0] = attrib.normals[3 * idx.normal_index + 0];
                        norm[1] = attrib.normals[3 * idx.normal_index + 1];
                        norm[2] = attrib.normals[3 * idx.normal_index + 2];
                    }

                    if (2 * idx.texcoord_index < attrib.texcoords.size()) {
                        tex[0] = attrib.texcoords[2 * idx.texcoord_index + 0];
                        tex[1] = attrib.texcoords[2 * idx.texcoord_index + 1];
                    }

                    triangle.vertices[v].position = pos;
                    triangle.vertices[v].normals = norm;
                    triangle.vertices[v].texture = tex;
                }
                index_offset += fv;
                shapes[s].mesh.material_ids[f];

                glm::vec3 v0v1 = triangle.vertices[1].position - triangle.vertices[0].position;
                glm::vec3 v0v2 = triangle.vertices[2].position - triangle.vertices[0].position;
                triangle.N = glm::normalize(glm::cross(v0v1, v0v2));
                triangles.push_back(triangle);
            }
        }
    }

    void transform(const glm::mat4& matrix) {
        for (Triangle& t : triangles) {
            for (int i = 0; i < 3; i++) {
                t.vertices[i].position = matrix * glm::vec4(t.vertices[i].position,1.0f);
            }
        }
    }
    std::vector<Triangle> triangles;
} Model;