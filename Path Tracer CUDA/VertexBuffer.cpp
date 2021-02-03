#include "VertexBuffer.h"

VertexBuffer::VertexBuffer(const float* data, unsigned int size, GLenum type) {
	glGenBuffers(1, &bufferID);
	glBindBuffer(GL_ARRAY_BUFFER, bufferID);
	glBufferData(GL_ARRAY_BUFFER, size, data, type);
}

void VertexBuffer::bind() const {
	glBindBuffer(GL_ARRAY_BUFFER, bufferID);
}

void VertexBuffer::unbind() const {
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

VertexBuffer::~VertexBuffer() {
	glDeleteBuffers(1, &bufferID);
}