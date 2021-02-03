#include "Shader.h"

Shader::Shader(const char* vertexPath, const char* fragmentPath) {
	string vertexCode, fragmentCode;
	const char* cVertexCode;
	const char* cFragmentCode;

	getCode(vertexPath, vertexCode);
	getCode(fragmentPath, fragmentCode);
	cVertexCode = vertexCode.c_str();
	cFragmentCode = fragmentCode.c_str();

	int vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, &cVertexCode, NULL);
	glCompileShader(vertexShader);
	checkCompileErrors(vertexShader, "VERTEX");

	int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, &cFragmentCode, NULL);
	glCompileShader(fragmentShader);
	checkCompileErrors(fragmentShader, "FRAGMENT");

	programID = glCreateProgram();
	glAttachShader(programID, vertexShader);
	glAttachShader(programID, fragmentShader);
	glLinkProgram(programID);
	checkCompileErrors(fragmentShader, "PROGRAM");

	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);
}

void Shader::use() {
	glUseProgram(programID);
}

void Shader::setFloat(const char* name, float value) {
	glUniform1f(getUniformLocation(name), value);
}

void Shader::setInt(const char* name, int value) {
	glUniform1i(getUniformLocation(name), value);
}

void Shader::setBool(const char* name, bool value) {
	glUniform1i(getUniformLocation(name), (int)value);
}

void Shader::setMat4f(const char* name, float* value) {
	glUniformMatrix4fv(getUniformLocation(name), 1, GL_FALSE, value);
}

void Shader::setVec3f(const char* name, float* value) {
	glUniform3fv(getUniformLocation(name), 1, value);
}

int Shader::getID() const {
	return programID;
}

Shader::~Shader() {
	cout << "DELETED SHADER" << endl;
	glDeleteProgram(programID);
}

void Shader::getCode(const char* path, string& code) {
	ifstream in(path);
	stringstream stream;
	stream << in.rdbuf();
	in.close();
	code = stream.str();
}

void Shader::checkCompileErrors(unsigned int shader, std::string type) {
	int success;
	char infoLog[1024];
	if (type != "PROGRAM") {
		glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
		if (!success) {
			glGetShaderInfoLog(shader, 1024, NULL, infoLog);
			std::cout << "ERROR::SHADER_COMPILATION_ERROR of type: " <<
				type << "\n" << infoLog <<
				"\n -------------------------------" << std::endl;
		}
	} else {
		glGetProgramiv(shader, GL_LINK_STATUS, &success);
		if (!success) {
			glGetProgramInfoLog(shader, 1024, NULL, infoLog);
			std::cout << "ERROR::PROGRAM_LINKING_ERROR of type: " <<
				type << "\n" << infoLog <<
				"\n -------------------------------" << std::endl;
		}
	}
}

int Shader::getUniformLocation(const char* name) {
	if (uniformLocations.find(name) != uniformLocations.end()) {
		return uniformLocations[name];
	}
	int location = glGetUniformLocation(programID, name);
	uniformLocations[name] = location;
	return location;
}