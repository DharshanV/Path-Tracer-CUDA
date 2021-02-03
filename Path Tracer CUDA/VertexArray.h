#pragma once
#include <glad/glad.h>
#include "VertexBuffer.h"
#include "VertexBufferLayout.h"
class VertexArray {
public:
	VertexArray();

	void bind();
	void unbind();

	void addBuffer(const VertexBuffer& vb, const VertexBufferLayout& vbl);

	~VertexArray();
private:
	unsigned int arrayID;
};