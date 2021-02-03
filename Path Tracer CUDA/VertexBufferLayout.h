#pragma once
#include <iostream>
#include <vector>
#include <glad/glad.h>
using namespace std;

struct Elements {
	Elements(int size, unsigned int type,
		unsigned char normalized) {
		this->size = size;
		this->type = type;
		this->normalized = normalized;
	}
	int size;
	unsigned int type;
	unsigned char normalized;
};

class VertexBufferLayout {
public:
	VertexBufferLayout() {
		stride = 0;
	}
	~VertexBufferLayout() {}

	template<typename T>
	void push(int count) {
		cout << "Error pushing" << endl;
		static_assert(!true);
	}

	const vector<Elements>& getElements() const { return elements; };

	int getStride() const {
		return stride;
	}

	template<>
	void push<float>(int count) {
		elements.push_back(Elements(count, GL_FLOAT, false));
		stride += (count * sizeof(float));
	}

	template<>
	void push<unsigned int>(int count) {
		elements.push_back(Elements(count, GL_UNSIGNED_INT, false));
		stride += (count * sizeof(unsigned int));
	}

	template<>
	void push<unsigned char>(int count) {
		elements.push_back(Elements(count, GL_UNSIGNED_BYTE, true));
		stride += (count * sizeof(unsigned char));
	}
private:
	vector<Elements> elements;
	int stride;
};

