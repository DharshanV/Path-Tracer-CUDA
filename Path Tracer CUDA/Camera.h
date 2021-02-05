#pragma once
#include <glad/glad.h>
#include <glm/glm.hpp>
#include "Settings.h"

enum Camera_Movement {
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT,
    UP,
    DOWN,
};

class Camera {
public:
    glm::vec3 Position;
    glm::vec3 Front;
    glm::vec3 Up;
    glm::vec3 Right;
    glm::vec3 WorldUp;
    float yaw;
    float pitch;

    __host__ __device__
    Camera(glm::vec3 position = glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f)) : Front(glm::vec3(0.0f, 0.0f, -1.0f)) {
        Position = position;
        WorldUp = up;
        yaw = YAW;
        pitch = PITCH;
        updateCameraVectors();
    }

    __host__ __device__
    inline Ray getRay(float x, float y) {
        float deltaX = (-2.0f * x + 1.0f) * tan(90.0f / 2 * PI / 180);
        float deltaY = (-2.0f * y + 1.0f) * tan(90.0f / 2 * PI / 180);
        glm::vec3 rd = glm::normalize(Right * deltaX + Up * deltaY + Front);
        return Ray(Position, rd);
    }

    void ProcessKeyboard(Camera_Movement direction, float deltaTime) {
        float velocity = MOVEMENT_SPEED * deltaTime;
        if (direction == FORWARD)
            Position += Front * velocity;
        if (direction == BACKWARD)
            Position -= Front * velocity;
        if (direction == LEFT)
            Position -= Right * velocity;
        if (direction == RIGHT)
            Position += Right * velocity;
        if (direction == UP)
            Position += Up * velocity;
        if (direction == DOWN)
            Position -= Up * velocity;
    }

    void ProcessMouseMovement(float xoffset, float yoffset, GLboolean constrainPitch = true) {
        xoffset *= MOUSE_SENSITIVITY;
        yoffset *= MOUSE_SENSITIVITY;

        yaw += xoffset;
        pitch += yoffset;

        if (constrainPitch) {
            if (pitch > 89.0f)
                pitch = 89.0f;
            if (pitch < -89.0f)
                pitch = -89.0f;
        }
        updateCameraVectors();
    }

private:
    __host__ __device__
    void updateCameraVectors() {
        glm::vec3 front;
        front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        front.y = sin(glm::radians(pitch));
        front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
        Front = glm::normalize(front);
        Right = glm::normalize(glm::cross(WorldUp, Front));
        Up = glm::normalize(glm::cross(Front, Right));
    }
};