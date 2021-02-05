#pragma once
#include <iostream>
#include <glad/glad.h>
using namespace std;

class Texture {
public:
	Texture(const float* data, uint32_t width, uint32_t height);
	void bind(unsigned int slot = 0);
	void load(const float* data);
	void unbind();
	GLuint id();
	~Texture();
private:
	GLuint textureID;
	uint32_t width;
	uint32_t height;
};
