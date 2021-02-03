#pragma once
#include <iostream>
#include <glad/glad.h>

using namespace std;

class Texture {
public:
	Texture(const uint8_t* data, uint32_t width, uint32_t height);
	void bind(unsigned int slot = 0);
	void load(const uint8_t* data);
	void unbind();
	~Texture();
private:
	uint32_t textureID;
	uint32_t width;
	uint32_t height;
};
