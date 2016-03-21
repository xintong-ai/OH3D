//copied from http://r3dux.org/2015/01/a-simple-c-opengl-shader-loader-improved/

/***
author     : r3dux
version    : 0.3 - 15/01/2014
description: Gets GLSL source code either provided as strings or can load from filenames,
             compiles the shaders, creates a shader program which the shaders are linked
             to, then the program is validated and is ready for use via myProgram.use(),
             <draw-stuff-here> then calling myProgram.disable();

             Attributes and uniforms are stored in <string, int> maps and can be added
             via calls to addAttribute(<name-of-attribute>) and then the attribute
             index can be obtained via myProgram.attribute(<name-of-attribute>) - Uniforms
             work in the exact same way.
***/

#ifndef SHADER_PROGRAM_HPP
#define SHADER_PROGRAM_HPP
#include "QApplication"
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <stdexcept>


//
//#ifdef WIN32
//#include <Windows.h>
//#endif
//#include <GL/GL.h>
//#include <GL/glew.h>
//#include <qopenglfunctions>


class ShaderProgram 
{
private:
    // static DEBUG flag - if set to false then, errors aside, we'll run completely silent
    static const bool DEBUG = true;

    // We'll use an enum to differentiate between shaders and shader programs when querying the info log
    enum ObjectType
    {
        SHADER, PROGRAM
    };

    // Shader program and individual shader Ids
    GLuint programId;
    GLuint vertexShaderId;
	GLuint fragmentShaderId;
	GLuint geometryShaderId;

    // How many shaders are attached to the shader program
    GLuint shaderCount;

    // Map of attributes and their binding locations
    std::map<std::string, int> attributeMap;

    // Map of uniforms and their binding locations
    std::map<std::string, int> uniformMap;

    // Has this shader program been initialised?
    bool initialised;

    // ---------- PRIVATE METHODS ----------
    // Private method to return the current shader program info log as a string
    std::string getInfoLog(ObjectType type, int id)
    {
        GLint infoLogLength;
        if (type == SHADER)
        {
            qgl->glGetShaderiv(id, GL_INFO_LOG_LENGTH, &infoLogLength);
        }
        else // type must be ObjectType::PROGRAM
        {
			qgl->glGetProgramiv(id, GL_INFO_LOG_LENGTH, &infoLogLength);
        }

        GLchar *infoLog = new GLchar[infoLogLength + 1];
        if (type == SHADER)
        {
			qgl->glGetShaderInfoLog(id, infoLogLength, NULL, infoLog);
        }
        else // type must be ObjectType::PROGRAM
        {
			qgl->glGetProgramInfoLog(id, infoLogLength, NULL, infoLog);
        }

        // Convert the info log to a string
        std::string infoLogString(infoLog);

        // Delete the char array version of the log
        delete[] infoLog;

        // Finally, return the string version of the info log
        return infoLogString;
    }

    // Private method to compile a shader of a given type
    GLuint compileShader(std::string shaderSource, GLenum shaderType)
    {
        std::string shaderTypeString;
        switch (shaderType)
        {
            case GL_VERTEX_SHADER:
                shaderTypeString = "GL_VERTEX_SHADER";
                break;
            case GL_FRAGMENT_SHADER:
                shaderTypeString = "GL_FRAGMENT_SHADER";
                break;
            case GL_GEOMETRY_SHADER:
				shaderTypeString = "GL_GEOMETRY_SHADER";
                //throw std:: runtime_error("Geometry shaders are unsupported at this time.");
                break;
            default:
                throw std::runtime_error("Bad shader type enum in compileShader.");
                break;
        }

        // Generate a shader id
        // Note: Shader id will be non-zero if successfully created.
		GLuint shaderId = qgl->glCreateShader(shaderType);
        if (shaderId == 0)
        {
            // Display the shader log via a runtime_error
            /*throw std::runtime_error*/
			printf("Could not create shader of type %s : %s \n", shaderTypeString.c_str(), getInfoLog(SHADER, shaderId).c_str());

			printf("problematic shader: \n%s \n", shaderSource.c_str());
			
			exit(1);
        }

        // Get the source string as a pointer to an array of characters
        const char *shaderSourceChars = shaderSource.c_str();

        // Attach the GLSL source code to the shader
        // Params: GLuint shader, GLsizei count, const GLchar **string, const GLint *length
        // Note: The pointer to an array of source chars will be null terminated, so we don't need to specify the length and can instead use NULL.
		qgl->glShaderSource(shaderId, 1, &shaderSourceChars, NULL);

        // Compile the shader
		qgl->glCompileShader(shaderId);

        // Check the compilation status and throw a runtime_error if shader compilation failed
        GLint shaderStatus;
		qgl->glGetShaderiv(shaderId, GL_COMPILE_STATUS, &shaderStatus);
        if (shaderStatus == GL_FALSE)
        {
			printf("Could not create shader of type %s : %s", shaderTypeString.c_str(), getInfoLog(SHADER, shaderId).c_str());
			exit(1);
			//throw std::runtime_error(shaderTypeString + " compilation failed: " + getInfoLog(SHADER, shaderId));
        }
        else
        {
            if (DEBUG)
            {
                std::cout << shaderTypeString << " shader compilation successful." << std::endl;
            }
        }

        // If everything went well, return the shader id
        return shaderId;
    }

    // Private method to compile/attach/link/verify the shaders.
    // Note: Rather than returning a boolean as a success/fail status we'll just consider
    // a failure here to be an unrecoverable error and throw a runtime_error.
    void initialise(std::string vertexShaderSource, std::string fragmentShaderSource, std::string geometryShaderSource = "")
    {
        // Compile the shaders and return their id values
        vertexShaderId   = compileShader(vertexShaderSource,   GL_VERTEX_SHADER);
		fragmentShaderId = compileShader(fragmentShaderSource, GL_FRAGMENT_SHADER);
		if ("" != geometryShaderSource)
			geometryShaderId = compileShader(geometryShaderSource, GL_GEOMETRY_SHADER);

        // Attach the compiled shaders to the shader program
		qgl->glAttachShader(programId, vertexShaderId);
		qgl->glAttachShader(programId, fragmentShaderId);
		if ("" != geometryShaderSource)
			qgl->glAttachShader(programId, geometryShaderId);

        // Link the shader program - details are placed in the program info log
		qgl->glLinkProgram(programId);

        // Once the shader program has the shaders attached and linked, the shaders are no longer required.
        // If the linking failed, then we're going to abort anyway so we still detach the shaders.
		qgl->glDetachShader(programId, vertexShaderId);
		qgl->glDetachShader(programId, fragmentShaderId);
		if ("" != geometryShaderSource)
			qgl->glDetachShader(programId, geometryShaderId);

        // Check the program link status and throw a runtime_error if program linkage failed.
        GLint programLinkSuccess = GL_FALSE;
		qgl->glGetProgramiv(programId, GL_LINK_STATUS, &programLinkSuccess);
        if (programLinkSuccess == GL_TRUE)
        {
            if (DEBUG)
            {
                std::cout << "Shader program link successful." << std::endl;
            }
        }
        else
        {
            throw std::runtime_error("Shader program link failed: " + getInfoLog(PROGRAM, programId) );
        }

        // Validate the shader program
		qgl->glValidateProgram(programId);

        // Check the validation status and throw a runtime_error if program validation failed
        GLint programValidatationStatus;
		qgl->glGetProgramiv(programId, GL_VALIDATE_STATUS, &programValidatationStatus);
        if (programValidatationStatus == GL_TRUE)
        {
            if (DEBUG)
            {
                std::cout << "Shader program validation successful." << std::endl;
            }
        }
        else
        {
            throw std::runtime_error("Shader program validation failed: " + getInfoLog(PROGRAM, programId) );
        }

        // Finally, the shader program is initialised
        initialised = true;
    }

    // Private method to load the shader source code from a file
    std::string loadShaderFromFile(const std::string filename)
    {
        // Create an input filestream and attempt to open the specified file
        std::ifstream file( filename.c_str() );

        // If we couldn't open the file we'll bail out
        if ( !file.good() )
        {
            throw std::runtime_error("Failed to open file: " + filename);
        }

        // Otherwise, create a string stream...
        std::stringstream stream;

        // ...and dump the contents of the file into it.
        stream << file.rdbuf();

        // Now that we've read the file we can close it
        file.close();

        // Finally, convert the stringstream into a string and return it
        return stream.str();
    }


public:
    // Constructor
    ShaderProgram()
    {
        // We start in a non-initialised state - calling initFromFiles() or initFromStrings() will
        // initialise us.
        initialised = false;
//#ifdef WIN32
//		wglMakeCurrent(0,0);
//#endif
        // Generate a unique Id / handle for the shader program
        // Note: We MUST have a valid rendering context before generating the programId or we'll segfault!
		programId = qgl->glCreateProgram();
		qgl->glUseProgram(programId);

        // Initially, we have zero shaders attached to the program
        shaderCount = 0;
    }

    // Destructor
    ~ShaderProgram()
    {
        // Delete the shader program from the graphics card memory to
        // free all the resources it's been using
		qgl->glDeleteProgram(programId);
    }

    // Method to initialise a shader program from shaders provided as files
    void initFromFiles(std::string vertexShaderFilename, std::string fragmentShaderFilename)
    {
        // Get the shader file contents as strings
        std::string vertexShaderSource   = loadShaderFromFile(vertexShaderFilename);
        std::string fragmentShaderSource = loadShaderFromFile(fragmentShaderFilename);

        initialise(vertexShaderSource, fragmentShaderSource);
    }

    // Method to initialise a shader program from shaders provided as strings
	void initFromStrings(std::string vertexShaderSource, std::string fragmentShaderSource, std::string geometryShaderSource = "")
    {
		initialise(vertexShaderSource, fragmentShaderSource, geometryShaderSource);
    }

    // Method to enable the shader program - we'll suggest this for inlining
    inline void use()
    {
        // Santity check that we're initialised and ready to go...
        if (initialised)
        {
			qgl->glUseProgram(programId);
        }
        else
        {
            std::string msg = "Shader program " + programId;
            msg += " not initialised - aborting.";
            throw std::runtime_error(msg);
        }
    }

    // Method to disable the shader - we'll also suggest this for inlining
    inline void disable()
    {
		qgl->glUseProgram(0);
    }

    // Method to return the bound location of a named attribute, or -1 if the attribute was not found
    GLuint attribute(const std::string attributeName)
    {
        // You could do this method with the single line:
        //
        //		return attributeMap[attribute];
        //
        // BUT, if you did, and you asked it for a named attribute which didn't exist
        // like: attributeMap["FakeAttrib"] then the method would return an invalid
        // value which will likely cause the program to segfault. So we're making sure
        // the attribute asked for exists, and if it doesn't then we alert the user & bail.

        // Create an iterator to look through our attribute map (only create iterator on first run -
        // reuse it for all further calls).
        static std::map<std::string, int>::const_iterator attributeIter;

        // Try to find the named attribute
        attributeIter = attributeMap.find(attributeName);

        // Not found? Bail.
        if ( attributeIter == attributeMap.end() )
        {
            throw std::runtime_error("Could not find attribute in shader program: " + attributeName);
        }

        // Otherwise return the attribute location from the attribute map
        return attributeMap[attributeName];
    }

    // Method to returns the bound location of a named uniform
    GLuint uniform(const std::string uniformName)
    {
        // Note: You could do this method with the single line:
        //
        // 		return uniformLocList[uniform];
        //
        // But we're not doing that. Explanation in the attribute() method above.

        // Create an iterator to look through our uniform map (only create iterator on first run -
        // reuse it for all further calls).
        static std::map<std::string, int>::const_iterator uniformIter;

        // Try to find the named uniform
        uniformIter = uniformMap.find(uniformName);

        // Found it? Great - pass it back! Didn't find it? Alert user and halt.
        if ( uniformIter == uniformMap.end() )
        {
            throw std::runtime_error("Could not find uniform in shader program: " + uniformName);
        }

        // Otherwise return the attribute location from the uniform map
        return uniformMap[uniformName];
    }

    // Method to add an attribute to the shader and return the bound location
    int addAttribute(const std::string attributeName)
    {
        // Add the attribute location value for the attributeName key
		attributeMap[attributeName] = qgl->glGetAttribLocation(programId, attributeName.c_str());

        // Check to ensure that the shader contains an attribute with this name
        if (attributeMap[attributeName] == -1)
        {
            throw std::runtime_error("Could not add attribute: " + attributeName + " - location returned -1.");
        }
        else // Valid attribute location? Inform user if we're in debug mode.
        {
            if (DEBUG)
            {
                std::cout << "Attribute " << attributeName << " bound to location: " << attributeMap[attributeName] << std::endl;
            }
        }

        // Return the attribute location
        return attributeMap[attributeName];
    }

    // Method to add a uniform to the shader and return the bound location
    int addUniform(const std::string uniformName)
    {
        // Add the uniform location value for the uniformName key
		uniformMap[uniformName] = qgl->glGetUniformLocation(programId, uniformName.c_str());

        // Check to ensure that the shader contains a uniform with this name
        if (uniformMap[uniformName] == -1)
        {
            throw std::runtime_error("Could not add uniform: " + uniformName + " - location returned -1.");
        }
        else // Valid uniform location? Inform user if we're in debug mode.
        {
            if (DEBUG)
            {
                std::cout << "Uniform " << uniformName << " bound to location: " << uniformMap[uniformName] << std::endl;
            }
        }

        // Return the uniform location
        return uniformMap[uniformName];
    }

}; // End of class

#endif // SHADER_PROGRAM_HPP
