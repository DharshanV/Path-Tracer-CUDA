#include "VertexArray.h"

VertexArray::VertexArray() {
	glGenVertexArrays(1, &arrayID);
	glBindVertexArray(arrayID);
}

void VertexArray::bind() {
	glBindVertexArray(arrayID);
}

void VertexArray::unbind() {
	glBindVertexArray(0);
}

void VertexArray::addBuffer(const VertexBuffer& vb, const VertexBufferLayout& vbl) {
	this->bind();
	vb.bind();
	const vector<Elements>& elements = vbl.getElements();
	unsigned int offset = 0;
	for (unsigned int i = 0; i < elements.size(); i++) {
		const Elements& e = elements[i];
		glEnableVertexAttribArray(i);
		glVertexAttribPointer(i, e.size, e.type, e.normalized, vbl.getStride(), (void*)offset);
		offset += (e.size * sizeof(e.type));
	}
}

VertexArray::~VertexArray() {
	glDeleteVertexArrays(1, &arrayID);
}