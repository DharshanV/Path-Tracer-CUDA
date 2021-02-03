#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <map>

using namespace std;
class Shader {
public:
	Shader(const char* vertexPath, const char* fragmentPath);
	void use();
	void setFloat(const char* name, float value);
	void setInt(const char* name, int value);
	void setBool(const char* name, bool value);
	void setMat4f(const char* name, float* value);
	void setVec3f(const char* name, float* value);
	int getID() const;
	~Shader();
private:
	void getCode(const char* path, string& code);
	void checkCompileErrors(unsigned int shader, string type);
	int getUniformLocation(const char* name);
private:
	int programID;
	map<const char*, int> uniformLocations;
};