/*
 * @Date: 2019-08-29 09:48:01
 * @LastEditors: zerollzeng
 * @LastEditTime: 2020-03-02 14:58:37
 */
#pragma once
#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvInferRuntime.h"
#include "NvOnnxParser.h"

// using Severity = nvinfer1::ILogger::Severity;
class Logger : public nvinfer1::ILogger
{
public:
    explicit Logger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kWARNING)
        : mReportableSeverity(severity)
    {
    }

    //!
    //! \brief Forward-compatible method for retrieving the nvinfer::ILogger associated with this Logger
    //! \return The nvinfer1::ILogger associated with this Logger
    //!
    //! TODO Once all samples are updated to use this method to register the logger with TensorRT,
    //! we can eliminate the inheritance of Logger from ILogger
    //!
    nvinfer1::ILogger& getIxRTLogger() noexcept { return *this; }

    //!
    //! \brief Implementation of the nvinfer1::ILogger::log() virtual method
    //!
    //! Note samples should not be calling this function directly; it will eventually go away once we eliminate the
    //! inheritance from nvinfer1::ILogger
    //!
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override
    {
        if (severity <= mReportableSeverity)
        {
            std::cout << severityPrefix(mReportableSeverity) << "[IXRT] " << msg << std::endl;
        }
    }

    //!
    //! \brief Method for controlling the verbosity of logging output
    //!
    //! \param severity The logger will only emit messages that have severity of this level or higher.
    //!
    void setReportableSeverity(nvinfer1::ILogger::Severity severity) noexcept { mReportableSeverity = severity; }

    //!
    //! \brief Opaque handle that holds logging information for a particular test
    //!
    //! This object is an opaque handle to information used by the Logger to print test results.
    //! The sample must call Logger::defineTest() in order to obtain a TestAtom that can be used
    //! with Logger::reportTest{Start,End}().
    //!

    nvinfer1::ILogger::Severity getReportableSeverity() const { return mReportableSeverity; }

private:
    //!
    //! \brief returns an appropriate string for prefixing a log message with the given severity
    //!
    static const char* severityPrefix(nvinfer1::ILogger::Severity severity)
    {
        switch (severity)
        {
            case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return "[F] ";
            case nvinfer1::ILogger::Severity::kERROR: return "[E] ";
            case nvinfer1::ILogger::Severity::kWARNING: return "[W] ";
            case nvinfer1::ILogger::Severity::kINFO: return "[I] ";
            case nvinfer1::ILogger::Severity::kVERBOSE: return "[V] ";
            default: assert(0); return "";
        }
    }

    nvinfer1::ILogger::Severity mReportableSeverity;
};  // class Logger

class Trt
{
public:
    /**
     * @description: default constructor, will initialize plugin factory with default parameters.
     */
    Trt(std::string plugin_lib_path = "");

    ~Trt();

    /**
     * @description: create engine from onnx model
     * @onnxModel: path to onnx model
     * @engineFile: path to saved engien file will be load or save, if it's empty them will not
     *              save engine file
     * @return:
     */
    void CreateEngine(const std::string&              onnxModel,
                      const std::string&              engineFile,
                      const std::vector<std::string>& customOutput,
                      int                             mode);

    /**
     * @description: do inference on engine context, make sure you already copy your data to device memory,
     *               see DataTransfer and CopyFromHostToDevice etc.
     */
    void Forward(void** buffer, int batchSize, cudaStream_t stream = nullptr);

    void SetBindingDimensions(int batchSize);
    /**
     * @description: data transfer between host and device, for example befor Forward, you need
     *               copy input data from host to device, and after Forward, you need to transfer
     *               output result from device to host.
     * @bindIndex binding data index, you can see this in CreateEngine log output.
     */

    void SetDevice(int device);

    int GetDevice() const;

    /**
     * @description: get binding data pointer in device. for example if you want to do some post processing
     *               on inference output but want to process them in gpu directly for efficiency, you can
     *               use this function to avoid extra data io
     * @return: pointer point to device memory.
     */
    void* GetBindingPtr(int bindIndex) const;

    /**
     * @description: get binding data size in byte, so maybe you need to divide it by sizeof(T) where T is data type
     *               like float.
     * @return: size in byte.
     */
    size_t GetBindingSize(int bindIndex) const;

    /**
     * @description: get binding dimemsions
     * @return: binding dimemsions, see
     * https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/classnvinfer1_1_1_dims.html
     */
    nvinfer1::Dims   GetBindingDims(int bindIndex) const;
    std::vector<int> GetBindingDimsVec(int bindIndex) const;

    /**
     * @description: get binding data type
     * @return: binding data type, see
     * https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/namespacenvinfer1.html#afec8200293dc7ed40aca48a763592217
     */
    nvinfer1::DataType GetBindingDataType(int bindIndex) const;

    /**
     * @description: get binding name
     */
    std::string GetBindingName(int bindIndex) const;

    /**
     * Add dynamic shape profile
     */
    void AddDynamicShapeProfile(const std::string&      inputName,
                                const std::vector<int>& minDimVec,
                                const std::vector<int>& optDimVec,
                                const std::vector<int>& maxDimVec);

    int GetNbInputBindings() const;

    int GetNbOutputBindings() const;

protected:
    bool DeserializeEngine(const std::string& engineFile);

    void BuildEngine(const std::string& fileName);

    bool BuildEngineWithOnnx(const std::string&              onnxModel,
                             const std::string&              engineFile,
                             const std::vector<std::string>& customOutput);

    /**
     * description: Init resource such as device memory
     */
    void InitEngine();

protected:
    Logger mLogger;

    // tensorrt run mode 0:fp32 1:fp16 2:int8
    int mRunMode;

    void* plugin_handle;

    nvinfer1::NetworkDefinitionCreationFlags mFlags = 0;

    nvinfer1::IBuilderConfig* mConfig = nullptr;

    nvinfer1::IBuilder* mBuilder = nullptr;

    nvinfer1::INetworkDefinition* mNetwork = nullptr;

    nvinfer1::IRuntime* mRuntime = nullptr;

    std::vector<nvinfer1::DataType> mBindingDataType;

    int mNbInputBindings = 0;

    int mNbOutputBindings = 0;

    bool mIsDynamicShape = false;

public:
    // batch size
    int mBatchSize;

    int nbBindings;

    std::vector<size_t> mBindingSize;

    std::vector<void*> mBindingPtr;

    std::vector<nvinfer1::Dims> mBindingDims;

    nvinfer1::IExecutionContext* mContext = nullptr;

    nvinfer1::ICudaEngine* mEngine = nullptr;

    nvinfer1::IOptimizationProfile* mProfile = nullptr;

    std::vector<std::string> mBindingName;

    nvinfer1::Dims4 m_minDim;
    nvinfer1::Dims4 m_optDim;
    nvinfer1::Dims4 m_maxDim;
};
