//
//  Created by Izudin Dzafic on 28/07/2020.
//  Copyright © 2020 IDz. All rights reserved.
//
#pragma once
#include <gui/gl/View.h>
#include <gui/Key.h>
#include <gui/Texture.h>
#include <gui/FileDialog.h>
#include "Program.h"
#include <gui/gl/CommandList.h>
#include <gui/gl/Buffer.h>
#include <gui/gl/Textures.h>
#include <sc/IModel.h>

class ViewGLLighting : public gui::gl::View
{
    using Base = gui::gl::View;

    glm::mat4 _perspectiveMatrix;
    glm::mat4 _viewMatrix;
    glm::mat4 _mvpMatrix;

    glm::mat4 _modelViewMat;
    glm::mat4 _normalMat;
    float aspectRatio = 16.0f / 9.0f;

    glm::vec3 _lightPos;
    gui::gl::Buffer _gpuBuffer;

    double dT;
    double t = 0;
    double epsT = 1e-6;
    ssize_t paramIndex = -1;
    sc::IRealStaticModel::NameVector paramNames;
    sc::IRealStaticModel::IndexVector paramIndices;
    sc::IRealStaticModel::ValueVector paramValues;
    sc::IRealDynamicModel* pModel;
    sc::IDynamic* pDynSolver;
    cnt::SafeFullVector<uint32_t> outIndices;
    cnt::SafeFullVector<td::String> outNames;
    sc::IRealStaticModel::ValueVector outValues;
    std::ofstream fOut;

    //gui::gl::Command::AdditionalUniform _uniformMV, _uniformN;
    
    Program _program;
    
    float _angleX = 0;
    float _angleY = 0;

    bool rotation = false;
    float _dAngle = 0.01f;
    float _dAngleAct = 0.f;
    
private:
    bool setupShaders()
    {
        if (!_program.link(":shVert", ":shFrag"))
            return false;
        return true;
    }
    
    //setup data and send it to GPU, create command list
    void setup()
    {
        //_uniformMV.type = gui::gl::Command::Uniform::Mat4Ptr;
        //_uniformMV.pos = 1;
        //_uniformMV.ptrVal = &_modelViewMat;

        //_uniformN.type = gui::gl::Command::Uniform::Mat4Ptr;
        //_uniformN.pos = 2;
        //_uniformN.ptrVal = &_normalMat;

        //specify buffer layouts before creatin the context
        _gpuBuffer.init(64, 100, 100, { gui::gl::DataType::vec3, gui::gl::DataType::vec3 });

        //encode command to set transformation matrix (uniform)
        auto pMtxSetterCmd = _gpuBuffer.createCommand();
        pMtxSetterCmd->createMVPSetter(&_mvpMatrix);
        _gpuBuffer.encode(pMtxSetterCmd);
        
         //Define the cube's vertices and texture coordinates

#define NORMAL_xPLUS 1.0f, 0.0f, 0.0f
#define NORMAL_xMINUS -1.0f, 0.0f, 0.0f
#define NORMAL_yPLUS 0.0f, 1.0f, 0.0f
#define NORMAL_yMINUS 0.0f, -1.0f, 0.0f
#define NORMAL_zPLUS 0.0f, 0.0f, 1.0f
#define NORMAL_zMINUS 0.0f, 0.0f, -1.0f

        float a=1.f;
        float vertices[] = {
            // Front face (number 1)
            a,   a,  a,   NORMAL_zPLUS,    // Vertex 0 (top-right)
           -a,   a,  a,   NORMAL_zPLUS,   // Vertex 1 (top-left)
           -a,  -a,  a,   NORMAL_zPLUS,   // Vertex 2 (bottom-left)
            a,  -a,  a,   NORMAL_zPLUS,   // Vertex 3 (bottom-right)

            // Right face (number 2)
            a, -a,  a, NORMAL_xPLUS,// Vertex 8 (bottom-left)
            a, -a, -a, NORMAL_xPLUS,// Vertex 9 (bottom-right)
            a,  a, -a, NORMAL_xPLUS,// Vertex 10 (top-left)
            a,  a,  a,  NORMAL_xPLUS,// Vertex 11 (top-right)
            
            // Back face (number 6)
            a, -a, -a,  NORMAL_zMINUS, // Vertex 0 (top-right)
           -a, -a, -a,  NORMAL_zMINUS,  // Vertex 1 (top-left)
           -a,  a, -a,  NORMAL_zMINUS, // Vertex 2 (bottom-left)
            a,  a, -a,  NORMAL_zMINUS, // Vertex 3 (bottom-right)
            
            // Left face (number 5)
           -a, -a, -a, NORMAL_xMINUS,// Vertex 12 (bottom-left)
           -a,  -a, a, NORMAL_xMINUS,// Vertex 13 (bottom-right)
           -a,  a,  a, NORMAL_xMINUS,// Vertex 14 (top-left)
           -a,  a, -a, NORMAL_xMINUS,// Vertex 15 (top-right)

            // Top face (number 3)
           -a,  a,  a, NORMAL_yPLUS,// Vertex 16 (bottom-left)
            a,  a,  a, NORMAL_yPLUS,// Vertex 17 (bottom-right)
            a,  a, -a, NORMAL_yPLUS,// Vertex 18 (top-left)
           -a,  a, -a, NORMAL_yPLUS,// Vertex 19 (top-right)

            // Bottom face (number 4)
           -a, -a, -a, NORMAL_yMINUS,   // Vertex 20 (bottom-left)
            a, -a, -a, NORMAL_yMINUS,   // Vertex 21 (bottom-right)
            a, -a,  a, NORMAL_yMINUS,   // Vertex 22 (top-left)
           -a, -a,  a, NORMAL_yMINUS   // Vertex 23 (top-right)
        };
        
        td::UINT4 indices[] =
        {
            0, 1, 2, // Front face
            2, 3, 0, // Front face

            4, 5, 6, // Rigt face
            4, 6, 7, // Rigt face

            8, 9, 11, // Back face
            9, 10, 11, // Back face

            12, 13, 14, // Left face
            12, 14, 15, // Left face

            16, 17, 18, // Top face
            16, 18, 19, // Top face

            20, 21, 22, // Bottom face
            20, 22, 23  // Bottom face
        };

        td::UINT4 nVertices = 4*6;
        
        _gpuBuffer.appendVertices(vertices, nVertices);
        
        td::UINT4 nIndices = 3*2*6;
        _gpuBuffer.appendIndices(indices, nIndices);
        
        //encode command to draw textured cube
        auto pCubeTextureCmd = _gpuBuffer.createCommand();
        pCubeTextureCmd->createDrawElements(gui::gl::Primitive::Triangles, 0, nIndices);
#ifdef USE_TEXTURE_NORMALS
        pCubeTextureCmd->setTexture(gui::gl::Textures::Unit::T1, 1);
#endif
        _gpuBuffer.encode(pCubeTextureCmd);

        if (!_gpuBuffer.commit())
        {
            mu::dbgLog("ERROR! Cannot commit buffer to GPU");
            return;
        }

        // Load textures
        _program.setBuffer(&_gpuBuffer);
        
        //dbgCheckGLError();
    }
protected:
    void onResize(const gui::Size& newSize) override {
        aspectRatio = newSize.width / newSize.height;
        float fov = 90.0f; // Field of view in degrees
        float nearClip = 0.1f; // Near clipping plane
        float farClip = 100.0f; // Far clipping plane
        _perspectiveMatrix = glm::perspective(glm::radians(fov), aspectRatio, nearClip, farClip);
        _mvpMatrix = _perspectiveMatrix * _viewMatrix; //* I for model
    }

    void onInit() override
    {
        auto [major, minor] = getOpenGLVersion();
        mu::dbgLog("Used opengl %d.%d", major, minor);
        
        if (!setupShaders())
        {
            mu::dbgLog("ERROR! OpenGL cannnot setup shaders!");
            assert(false);
            return;
        }
        setup();
        
        
        // Set up the perspective parameters
        float fov = 90.0f; // Field of view in degrees
        float nearClip = 0.1f; // Near clipping plane
        float farClip = 100.0f; // Far clipping plane

        // Create a perspective matrix
        _perspectiveMatrix = glm::perspective(glm::radians(fov), aspectRatio, nearClip, farClip);
        
        // Camera parameters
        glm::vec3 cameraPosition = glm::vec3(2.0, 0.0, 0.0f);  // New camera position
        glm::vec3 cameraTarget = glm::vec3(0.0f, 0.0f, 0.0f);    // Camera target (where the camera is looking)
        glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);         // Up vector

        // Create a new view matrix
        _viewMatrix = glm::lookAt(cameraPosition, cameraTarget, cameraUp);
        
        _mvpMatrix = _perspectiveMatrix * _viewMatrix; //* I for model

        _lightPos = glm::vec3(-2., 0., 2.);
        
        gui::gl::Context::enable(gui::gl::Context::Flag::DepthTest);
        gui::gl::Context::enable(gui::gl::Context::Flag::CullFace);
        
        dbgCheckGLError();
    }

    bool prepareNextFrame() override
    {
        _angleX += _dAngleAct;
        //_angleY += _dAngleAct;

        // Create a model matrix with rotations around X and Y axes
        glm::mat4 modelMatrix = glm::mat4(1.0f);
        modelMatrix = glm::rotate(modelMatrix, _angleX, glm::vec3(1.0f, 0.0f, 0.0f)); // Rotate around X axis
        modelMatrix = glm::rotate(modelMatrix, _angleY, glm::vec3(0.0f, 1.0f, 0.0f)); // Rotate around Y axis

        // Combine the perspective, view, and model matrices to get the final MVP matrix
        _modelViewMat = _viewMatrix * modelMatrix;
        _normalMat = glm::transpose(glm::inverse(_modelViewMat));
        _mvpMatrix = _perspectiveMatrix * _viewMatrix * modelMatrix;
        return true;
    }
    
    void onDraw(const gui::Rect& rect) override
    {
        // Clear
        gui::gl::Context::clear(td::ColorID::Black);
        gui::gl::Context::clear({gui::gl::Context::Clear::Color, gui::gl::Context::Clear::Depth});

        _program.activate();    
        _program.setMV(_modelViewMat);
        _program.setN(_normalMat);
        _program.setLightPos(_lightPos);
        _program.setV(_viewMatrix);
        _program.execute();
        _program.deActivate();
    }

public:
    ViewGLLighting()
    : paramNames(1)
    , paramIndices(1)
    , paramValues(1)
    { 
        gui::gl::Context reqContext(gui::gl::Context::Animation::Yes, gui::gl::DepthBuffer::Size::B2);
        createContext(reqContext, {gui::InputDevice::Event::Keyboard, gui::InputDevice::Event::PrimaryClicks, gui::InputDevice::Event::SecondaryClicks });
        s_sdkPath = mu::getHomePath() / "natID.SDK";
    }
    
    ~ViewGLLighting()
    {
        makeCurrentContext();
    }

    void start() {

#if defined(MU_WINDOWS)
        const char* InFile = "C:/Users/DiV/natID.Examples/Turbina/modeli/turbina_vdc_w.dmodl";
        const char* OutFile = "C:/Users/DiV/natID.Examples/Turbina/temp/rez.txt";
#elif defined(MU_MACOS)
        const char* Folder = "/Volumes/RAMDisk/Res"; // NOTE: adjust output folder!!
#else
        // Linux
        const char* Folder = "/media/RAMDisk/Res"; // NOTE: adjust output folder!!
#endif

        testRealDynamic(sc::IDynamic::Problem::DAE, InFile, OutFile, 20.0, td::String("β_ref"));
    }

    void stop() {
        _dAngle=0.0;
        _dAngleAct = _dAngle;
    }
    
    void updateSpeed(float val)
    {
        _dAngle = val;
        _dAngleAct = _dAngle;// rotation ? _dAngle : 0.f;
    }
    

    void switchRotation()
    {
        rotation = !rotation;
        _dAngleAct = _dAngle;// rotation ? _dAngle : 0.f;
    }



    fo::fs::path s_sdkPath;

    enum class Location { Real = 0, Complex, Selected };

    template <class TNAMES, class TVALS>

    inline void showResults(std::ofstream& fOut, const char* lbl, const TNAMES& outNames, const TVALS& vals)
    {
        fOut << td::endl;
        fOut << lbl << td::endl;
        fOut << "--------------------" << td::endl;
        fOut << "Name      value" << td::endl;
        fOut << "--------------------" << td::endl;
        auto nVals = outNames.size();
        for (decltype(nVals) i = 0; i < nVals; ++i)
            fOut << outNames[i] << ": " << vals[i] << td::endl;
        fOut << "--------------------" << td::endl;
    }


    inline void showResHeader(std::ofstream& fOut, sc::IRealDynamicModel::NameVector& outNames, const char* lbl = nullptr)
    {
        if (lbl)
            fOut << lbl << td::endl;

        fOut << "t";
        for (const auto& name : outNames)
            fOut << " " << name;
        fOut << td::endl;
    }

    inline void showResRow(std::ofstream& fOut, double t, sc::IRealDynamicModel::ValueVector& values)
    {
        fOut << t;
        for (const auto& val : values)
            fOut << " " << val;
        fOut << td::endl;
    }

    
    
    inline void testRealDynamic(sc::IDynamic::Problem problem, const char* inFile, const char* outFile, double tFinal, const td::String& paramName = td::String())
    {
        sc::ILog* pLog = sc::getConsoleLogger();
        pModel = sc::createRealDynamicModel(problem, pLog);
        assert(pModel);
        //mem::PointerReleaser releaser(pModel);
        td::String inFileName = inFile;
        td::String outFileName = outFile;
        if (inFileName.length() == 0 || outFileName.length() == 0)
        {
            return;
        }

        if (!pModel->initFromFile(inFileName))
        {
            return;
        }

        pDynSolver = pModel->getSolverInterface();
        if (!pDynSolver)
        {
            return;
        }

        //res file
        if (!fo::createTextFile(fOut, outFileName))
        {
            return;
        }

        //check params
        if (paramName.length() != 0)
        {
            std::string pom = paramName.c_str();
            paramIndex = pModel->getParameterIndex(paramName);
            if (paramIndex < 0)
            {
                return;
            }

            paramNames[0] = paramName;
            paramIndices[0] = (unsigned int)paramIndex;

            //get parameter values
            pModel->getParameterValues(paramIndices, paramValues);
        }

        //check deltaT
        dT = 0;
        {
            dT = pDynSolver->getStepSize();
            if (dT <= 0)
            {
                dT = 0.001;
                pDynSolver->setStepSize(dT);
            }
        }
        //initial reset
        if (!pDynSolver->reset(0))
        {
            return;
        }

        outIndices = pModel->getOutputSymbolIndices();
        if (outIndices.size() == 0)
        {
            return;
        }

        outNames = pModel->getOutputSymbolNames(outIndices);
        if (outNames.size() == 0)
        {
            return;
        }

        if (!pModel->getOutputSymbolValues(outIndices, outValues))
        {
            return;
        }

        showResHeader(fOut, outNames);

        showResRow(fOut, 0, outValues);

        //    if (paramIndex < 0)
        //    {
        //        std::cout << "INFO: testRealDynamic completed successfully (without param manipulaitons)" << td::endl;
        //        return;
        //    }
        t = 0;
        epsT = 1e-6;

    }

    void runStep() {
        t += dT;
        if (paramIndex >= 0)
        {
            pModel->getNumberOfParameters();
            paramValues[0] = 0;
            pModel->setParameterValues(paramIndices, paramValues);
        }

        auto sol = pDynSolver->step();
        if (sol != sc::Solution::OK)
        {
            std::cout << "ERROR! Cannot solve the problem!" << td::endl;
            return;
        }
        pModel->getOutputSymbolValues(outIndices, outValues);
        updateSpeed(outValues[0]/100);
        showResRow(fOut, t, outValues);
    }

};
