#pragma once
#include <glad/glad.h>
#include <iostream>
using namespace std;
class VertexBuffer {
public:
	VertexBuffer(const float* data, unsigned int size, GLenum type = GL_STATIC_DRAW);
	void bind() const;
	void unbind() const;

	~VertexBuffer();
private:
	unsigned int bufferID;
};