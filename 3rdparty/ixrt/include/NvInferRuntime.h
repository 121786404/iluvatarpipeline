
#pragma once
#include "NvInferImpl.h"
#include "NvInferRuntimeCommon.h"
namespace nvinfer1 {
class IExecutionContext;
class ICudaEngine;
class IEngineInspector;

enum class TensorLocation : int32_t {
    kDEVICE = 0,
    kHOST = 1,
};

enum class OptProfileSelector : int32_t { kMIN = 0, kOPT = 1, kMAX = 2 };

class INoCopy {
   protected:
    INoCopy() = default;
    virtual ~INoCopy() = default;
    INoCopy(INoCopy const& other) = delete;
    INoCopy& operator=(INoCopy const& other) = delete;
    INoCopy(INoCopy&& other) = delete;
    INoCopy& operator=(INoCopy&& other) = delete;
};

class Weights {
   public:
    DataType type;
    void const* values;
    int64_t count;
};

class IHostMemory : public INoCopy {
   public:
    virtual ~IHostMemory() noexcept = default;

    void* data() const noexcept { return mImpl->data(); }

    std::size_t size() const noexcept { return mImpl->size(); }

    DataType type() const noexcept { return mImpl->type(); }

   protected:
    apiv::VHostMemory* mImpl;
};

enum class DimensionOperation : int32_t {
    kSUM = 0,
    kPROD = 1,
    kMAX = 2,
    kMIN = 3,
    kSUB = 4,
    kEQUAL = 5,
    kLESS = 6,
    kFLOOR_DIV = 7,
    kCEIL_DIV = 8
};

class IDimensionExpr : public INoCopy {
   public:
    bool isConstant() const noexcept { return mImpl->isConstant(); }

    int32_t getConstantValue() const noexcept { return mImpl->getConstantValue(); }

   protected:
    apiv::VDimensionExpr* mImpl;
    virtual ~IDimensionExpr() noexcept = default;
};

class IExprBuilder : public INoCopy {
   public:
    IDimensionExpr const* constant(int32_t value) noexcept { return mImpl->constant(value); }

    IDimensionExpr const* operation(DimensionOperation op, IDimensionExpr const& first,
                                    IDimensionExpr const& second) noexcept {
        return mImpl->operation(op, first, second);
    }

   protected:
    apiv::VExprBuilder* mImpl;
    virtual ~IExprBuilder() noexcept = default;
};

class DimsExprs {
   public:
    int32_t nbDims;
    IDimensionExpr const* d[Dims::MAX_DIMS];
};

struct DynamicPluginTensorDesc {
    PluginTensorDesc desc;

    Dims min;

    Dims max;
};

enum class ProfilingVerbosity : int32_t {
    kLAYER_NAMES_ONLY = 0,
    kNONE = 1,
    kDETAILED = 2,
};

template <>
constexpr inline int32_t EnumMax<ProfilingVerbosity>() noexcept {
    return 3;
}

//!
//! \class ICudaEngine
//!
//! \brief An engine for executing inference on a built network.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class ICudaEngine : public INoCopy {
   public:
    virtual ~ICudaEngine() noexcept = default;

    //!
    //! \brief Get the number of binding indices.
    //!
    //! The relation between binding indices and number of inputs can be formulated to:
    //!     nbInputs * K = getNbBindings()
    //! Where K is number of profilers. Thus first getNbBindings()/K for 0th profiler,
    //! then 1st profiler etc.
    //!
    //! \see getBindingIndex()
    //! \return The number of binding indices.
    //!
    int32_t getNbBindings() const noexcept { return mImpl->getNbBindings(); }
    //!
    //! \brief Retrieve the binding index for a named tensor.
    //!
    //! IExecutionContext::enqueueV2() and IExecutionContext::executeV2() require an array of buffers.
    //!
    //! Model input names are mapped to [0, n-1], where n means number of inputs and outputs
    //!
    //! \see getBindingName()
    //!
    //! \param name The name of input or output of the network
    //! \return The binding index of the given name or -1 if the name is not any one of inputs or outputs
    //!
    int32_t getBindingIndex(char const* name) const noexcept { return mImpl->getBindingIndex(name); }
    //!
    //! \brief Retrieve the binding name from its binding index
    //!
    //! IExecutionContext::enqueueV2() and IExecutionContext::executeV2() require an array of buffers.
    //!
    //! Model input names are mapped to [0, n-1], where n means number of inputs and outputs
    //!
    //! \see getBindingName()
    //!
    //! \param bindingIndex The binding index.
    //! \return The binding name of the given index or nullptr if the index if output of range.
    //!
    char const* getBindingName(int32_t bindingIndex) const noexcept { return mImpl->getBindingName(bindingIndex); }

    //!
    //! \brief Test if a binding is the input of model
    //!
    //! \param bindingIndex The binding index.
    //! \return True if it is input or false.
    //!
    bool bindingIsInput(int32_t bindingIndex) const noexcept { return mImpl->bindingIsInput(bindingIndex); }

    //!
    //! \brief Get dimensions of input or output of a network
    //!
    //! \param bindingIndex The binding index.
    //! \return The binding dimensions for the given index, the nbDims of result will be -1 if index is ilegal
    //!
    Dims getBindingDimensions(int32_t bindingIndex) const noexcept { return mImpl->getBindingDimensions(bindingIndex); }

    //!
    //! \brief Get data type of input or output of a network
    //!
    //! \param bindingIndex The binding index.
    //! \return The binding data type for the given index, or DataType::kUNKNOWN for ilegal index
    //!
    DataType getBindingDataType(int32_t bindingIndex) const noexcept { return mImpl->getBindingDataType(bindingIndex); }

    //!
    //! \brief Get number of layers of a network
    //! \return The number of layers of a network
    //!
    int32_t getNbLayers() const noexcept { return mImpl->getNbLayers(); }

    //!
    //! \brief Serialize the network into a stream
    //!
    IHostMemory* serialize() const noexcept { return mImpl->serialize(); }

    //!
    //! \brief Create an execution context.
    //!
    //! \see IExecutionContext.
    //! \see IExecutionContext::setOptimizationProfile()
    //!
    IExecutionContext* createExecutionContext() noexcept { return mImpl->createExecutionContext(); }

    //! \brief create an execution context without any device memory allocated
    //!
    //! The memory for execution of this device context must be supplied by the application.
    //!
    IExecutionContext* createExecutionContextWithoutDeviceMemory() noexcept {
        return mImpl->createExecutionContextWithoutDeviceMemory();
    }

    //!
    //! \brief Get device memory used of the engine
    //!
    //! \return The size of device memory used of the engine
    //!
    size_t getDeviceMemorySize() const noexcept { return mImpl->getDeviceMemorySize(); }

    //!
    //! \brief Get data format of input or output of a network
    //!
    //! \param bindingIndex The binding index.
    //! \return The binding format for the given index or TensorFormat::kUNKNOWN for ilegal index.
    //!
    TensorFormat getBindingFormat(int32_t bindingIndex) const noexcept { return mImpl->getBindingFormat(bindingIndex); }

    //!
    //! \brief Get binding format descriptor of input or output of a network
    //!
    //! \param bindingIndex The binding index.
    //! \return Return the human readable description of the tensor format, or nullptr if the provided name does not
    //!         map to an input or output tensor.
    //!
    char const* getBindingFormatDesc(int32_t bindingIndex) const noexcept {
        return mImpl->getBindingFormatDesc(bindingIndex);
    }
    //!
    //! \brief Returns the name of the network associated with the engine.
    //!
    //! The name is set during network creation and is retrieved after
    //! building or deserialization.
    //!
    //! \see INetworkDefinition::setName(), INetworkDefinition::getName()
    //!
    //! \return A null-terminated C-style string representing the name of the network.
    //!
    char const* getName() const noexcept { return mImpl->getName(); }
    //!
    //! \brief Get number of optimization profiles
    //!
    //! The number of optimization profiles is at least 1.
    //!
    //! \see IExecutionContext::setOptimizationProfile()
    //!
    int32_t getNbOptimizationProfiles() const noexcept { return mImpl->getNbOptimizationProfiles(); }

    //!
    //! \brief Get the minimum / optimum / maximum dimensions for a given input binding under an optimization
    //! profile.
    //!
    //! \param bindingIndex The input binding index, which must belong to the given profile,
    //!        or be in [0 and bindingsPerProfile-1].
    //!
    //! \param profileIndex The profile index, the range of which must be [0, getNbOptimizationProfiles()-1].
    //!
    //! \param select Options to query the minimum, optimum, or maximum dimensions for the binding.
    //!
    //! \return The minimum / optimum / maximum dimensions for this binding in this profile.
    //!         If the profileIndex or bindingIndex are invalid, return Dims with nbDims=-1.
    Dims getProfileDimensions(int32_t bindingIndex, int32_t profileIndex, OptProfileSelector select) const noexcept {
        return mImpl->getProfileDimensions(bindingIndex, profileIndex, select);
    }

    //!
    //! \brief Get the minimum / optimum / maximum dimensions for an input tensor given its name under an optimization
    //! profile.
    //!
    //! \param tensorName The name of an input tensor.
    //!
    //! \param profileIndex The profile index, which must be between 0 and getNbOptimizationProfiles()-1.
    //!
    //! \param select Whether to query the minimum, optimum, or maximum dimensions for this input tensor.
    //!
    //! \return The minimum / optimum / maximum dimensions for an input tensor in this profile.
    //!         If the profileIndex is invalid or provided name does not map to an input tensor, return Dims{-1, {}}
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including the terminator.
    //!
    Dims getProfileShape(char const* tensorName, int32_t profileIndex, OptProfileSelector select) const noexcept {
        return mImpl->getProfileShape(tensorName, profileIndex, select);
    }

    //!
    //! \brief Create a new engine observer which prints the layer information in an engine or an execution context.
    //!
    //! \see IEngineObserver.
    //!
    IEngineObserver* createEngineObserver() noexcept { return mImpl->createEngineObserver(); }

    ProfilingVerbosity getProfilingVerbosity() const noexcept { return mImpl->getProfilingVerbosity(); }

    IEngineInspector* createEngineInspector() const noexcept { return mImpl->createEngineInspector(); }

    //!
    //! \brief Return number of IO tensors.
    //!
    //! It is the number of input and output tensors for the network from which the engine was built.
    //! The names of the IO tensors can be discovered by calling getIOTensorName(i) for i in 0 to getNbIOTensors()-1.
    //!
    //! \see getIOTensorName()
    //!
    int32_t getNbIOTensors() const noexcept { return mImpl->getNbIOTensors(); }

    //!
    //! \brief Return name of an IO tensor.
    //!
    //! \param index value between 0 and getNbIOTensors()-1
    //!
    //! \see getNbIOTensors()
    //!
    char const* getIOTensorName(int32_t index) const noexcept { return mImpl->getIOTensorName(index); }

    //!
    //! \brief Determine whether a tensor is an input or output tensor.
    //!
    //! \param tensorName The name of an input or output tensor.
    //!
    //! \return kINPUT if tensorName is an input, kOUTPUT if tensorName is an output, or kNONE if neither.
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including the terminator.
    //!
    TensorIOMode getTensorIOMode(char const* tensorName) const noexcept { return mImpl->getTensorIOMode(tensorName); }

    //!
    //! \brief Determine the required data type for a buffer from its tensor name.
    //!
    //! \param tensorName The name of an input or output tensor.
    //!
    //! \return The type of the data in the buffer, or DataType::kFLOAT if the provided name does not map to an input or
    //! output tensor.
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including the terminator.
    //!
    DataType getTensorDataType(char const* tensorName) const noexcept { return mImpl->getTensorDataType(tensorName); }
    //!
    //! \brief Get shape of an input or output tensor.
    //!
    //! \param tensorName The name of an input or output tensor.
    //!
    //! \return shape of the tensor, with -1 in place of each dynamic runtime dimension,
    //!         or Dims{-1, {}} if the provided name does not map to an input or output tensor.
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including the terminator.
    //!
    Dims getTensorShape(char const* tensorName) const noexcept { return mImpl->getTensorShape(tensorName); }

    //!
    //! \brief Set the ErrorRecorder for this interface
    //!
    //! Assigns the ErrorRecorder to object. This will call incRefCount() at least once. When an error occurs
    //! in the object, will transfer to the the error recorder. If set an nullptr, the previous error recorder
    //! will be called decRefCount(). If no ErrorRecorder set, will call global logger.
    //!
    //! \param recorder The error recorder to be registered.
    //
    //! \see getErrorRecorder()
    //!
    void setErrorRecorder(IErrorRecorder* recorder) noexcept { mImpl->setErrorRecorder(recorder); }

    //!
    //! \brief get the ErrorRecorder assigned to this interface.
    //!
    //! Retrieves the error recorder assigned to the object, returns to
    //! nullptr if no recorder assigned
    //!
    //! \return A pointer to the IErrorRecorder object that has been registered or nullptr.
    //!
    //! \see setErrorRecorder()
    //!
    IErrorRecorder* getErrorRecorder() const noexcept { return mImpl->getErrorRecorder(); }

   protected:
    apiv::VCudaEngine* mImpl;
};

struct ExecutionContextTensorDesc {
    //! device ptr of the tensor
    void const* data;
    //! internal paddings, for better performance if used
    Dims paddings;
    //! dimension after internal paddings
    Dims dims;
    //! data type of the tensor
    DataType type;
    //! data format of the tensor
    TensorFormat format;
    int32_t itemsize;
    //! for int8, the scale factor
    float* scale;
    //! number of scales, should be a non-negative number
    int32_t nb_scales;
    //! If the tensor is an initializer
    bool is_initializer;
};

struct ExecutionContextInfo {
    //! The name of the op being executed
    AsciiChar const* opName;
    //! The type of the op being executed
    LayerType type;
    //! The op type defined in onnx fashion, optional
    AsciiChar const* op_type;
    //! The number of inputs of the layer
    int32_t nbInputs;
    //! The number of outputs of the layer
    int32_t nbOutputs;
    //! The input names of the executed op
    AsciiChar const* const* inputNames;
    //! The output names of the executed op
    AsciiChar const* const* outputNames;
    //! The input tensors of the executed op
    ExecutionContextTensorDesc const* inputTensors;
    //! The output tensors of the executed op
    ExecutionContextTensorDesc const* outputTensors;
    //! The hook being executed
    AsciiChar const* hookName;
};

typedef void (*ExecutionHook)(ExecutionContextInfo const*);

enum class ExecutionHookFlag : int32_t {
    //! Batch dimension of network should be explicit.
    kPRERUN = 0x1,
    kPOSTRUN = 0x1 << 1
};

//!
//! \class IExecutionContext
//!
//! \brief Inference context using an engine
//!
//! One ICudaEngine could create multi execution context instance, those context will share the weight of engine
//! but with independent inference buffer for the data that flow between operators
//! If ICudaEngine object accept dynamic shape input, each execution context will use independent optimization profile
//!
//! \warning Inherit from this class, will break forward-compatibility of the API and ABI.
class IExecutionContext : public INoCopy {
   public:
    virtual ~IExecutionContext() noexcept = default;

    //!
    //! \brief Get the related engine.
    //!
    //! \see ICudaEngine
    //!
    ICudaEngine const& getEngine() const noexcept { return mImpl->getEngine(); }

    //!
    //! \brief Set the execution context name.
    //!
    //! This method will copy the name string.
    //!
    void setName(char const* name) noexcept { mImpl->setName(name); }

    //!
    //! \brief Get the name of the execution context.
    //!
    char const* getName() const noexcept { return mImpl->getName(); }

    //!
    //! \brief Set the device memory for use by this execution context.
    //!
    //! \see ICudaEngine::getDeviceMemorySize() ICudaEngine::createExecutionContextWithoutDeviceMemory()
    //!
    void setDeviceMemory(void* memory) noexcept { mImpl->setDeviceMemory(memory); }

    //!
    //! \brief Get the index of the currently used optimization profile.
    //!
    //! If the profile index does not correctly set, this call will return an invalid value of -1,
    //! Inference call like enqueueV2() or executeV2() will fail
    //!
    int32_t getOptimizationProfile() const noexcept { return mImpl->getOptimizationProfile(); }

    //!
    //! \brief Set the input dimensions for specific binding buffer
    //!
    //! \param bindingIndex input buffer binding index that  must be same with the ICudaEngine definition .
    //!
    //! \param dimensions input tensor dimensions . It must be in the
    //!        range of currently used optimization profile that contain mix and max boundary of dynamic shape
    //!
    //! For all input dynamic binding, this method needs to be called before either enqueueV2() or executeV2()
    //! All input setting status can be checked using the method allInputDimensionsSpecified().
    //!
    //! \warning If input dimensions are different than the previous setting, This function will updates context
    //! resource Lead to side effect about performance
    //!
    //! \return false if an error happened, like
    //!          1. bindingIndex is out of range for the currently used optimization profile
    //!          2. binding dimension is out range of min-max range for currently used optimization profile
    //!          3. set dimension for fixed input shape engine
    //!         true if set dimension success
    //!
    bool setBindingDimensions(int32_t bindingIndex, Dims dimensions) noexcept {
        return mImpl->setBindingDimensions(bindingIndex, dimensions);
    }

    //!
    //! \brief Get the dimensions of a binding
    //!
    //! \param bindingIndex index of an binding tensor
    //!
    //! \return Dimension for specific bindingINdex
    //!
    //!  An invalid Dims with nbDims == -1 will be returned, if the bindingIndex is out of range
    //!
    //! If binding index associate to output tensor, then both
    //! allInputDimensionsSpecified() and allInputShapesSpecified() must be true
    //! before calling this method.
    Dims getBindingDimensions(int32_t bindingIndex) const noexcept { return mImpl->getBindingDimensions(bindingIndex); }

    //!
    //! \brief Set shape of named input.
    //!
    //! \param tensorName The name of an input tensor.
    //! \param dims The shape of an input tensor.
    //!
    //! \return True on success, false if the provided name of input does not exist. or some other error occurred
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including the terminator.
    //!
    bool setInputShape(char const* tensorName, Dims const& dims) noexcept {
        return mImpl->setInputShape(tensorName, dims);
    }

    //!
    //! \brief Return the shape of the given input or output.
    //!
    //! \param tensorName The name of an input or output tensor.
    //!
    //! Return Dims{-1, {}} if the provided name does not map to an input or output tensor.
    //! Otherwise return the shape of the input or output tensor.
    //!
    //! A dimension in an input tensor will have a -1 wildcard value if all the following are true:
    //!  * setInputShape() has not yet been called for this tensor
    //!  * The dimension is a runtime dimension that is not implicitly constrained to be a single value.
    //!
    //! A dimension in an output tensor will have a -1 wildcard value if the dimension depends
    //! on values of execution tensors OR if all the following are true:
    //!  * It is a runtime dimension.
    //!  * setInputShape() has NOT been called for some input tensor(s) with a runtime shape.
    //!  * setTensorAddress() has NOT been called for some input tensor(s) with isShapeInferenceIO() = true.
    //!
    //! An output tensor may also have -1 wildcard dimensions if its shape depends on values of tensors supplied to
    //! enqueueV3().
    //!
    //! If the request is for the shape of an output tensor with runtime dimensions,
    //! all input tensors with isShapeInferenceIO() = true should have their value already set,
    //! since these values might be needed to compute the output shape.
    //!
    //! Examples of an input dimension that is implicitly constrained to a single value:
    //! * The optimization profile specifies equal min and max values.
    //! * The dimension is named and only one value meets the optimization profile requirements
    //!   for dimensions with that name.
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including the terminator.
    //!
    Dims getTensorShape(char const* tensorName) const noexcept { return mImpl->getTensorShape(tensorName); }

    //!
    //! \brief Set values of shape tensor
    //!
    //! \warning Not support for now
    //!
    bool setInputShapeBinding(int32_t bindingIndex, int32_t const* data) noexcept {
        return mImpl->setInputShapeBinding(bindingIndex, data);
    }

    //!
    //! \brief Get value of shape tensor
    //!
    //! \warning Not support for now
    //!
    bool getShapeBinding(int32_t bindingIndex, int32_t* data) const noexcept {
        return mImpl->getShapeBinding(bindingIndex, data);
    }

    //!
    //! \brief Whether all input dynamic dimensions have been specified
    //!
    //! \return True if all input dynamic dimensions have been specified or network has no dynamically shaped input
    //! tensors.
    //!
    //!
    bool allInputDimensionsSpecified() const noexcept { return mImpl->allInputDimensionsSpecified(); }

    //!
    //! \brief Whether all input shape bindings have been specified
    //!
    //! \return True if all input shape bindings have been specified or network has no input shape bindings.
    //!
    //! \warning Not support for now
    //!
    bool allInputShapesSpecified() const noexcept

    {
        return mImpl->allInputShapesSpecified();
    }

    //!
    //! \brief Execute inference synchronously
    //!
    //! Input and output buffer pointer should assign to an array then pass to this function
    //! Array size is total number of input and output, index mapping could access by ICudaEngine::getBindingIndex().
    //!
    //! \param bindings An array of pointers to input and output buffers
    //!
    //! \return True if execution succeeded.
    //!
    bool executeV2(void* const* bindings) noexcept { return mImpl->executeV2(bindings); }

    //!
    //! \brief Execute inference asynchronously
    //!
    //! Input and output buffer pointer should assign to an array then pass to this function
    //! Array size is total number of input and output, index mapping could access by ICudaEngine::getBindingIndex().
    //!
    //! \param bindings An array of pointers to input and output buffers
    //! \param stream A cuda stream on which the inference kernels will be used to enqueue cuda kernel
    //! \param inputConsumed An optional event which will notify application while the input buffers has been consumed
    //! user could reload input data
    //!
    //! \return True if the cuda kernels enqueue successfully.
    //!
    //! \warning Calling enqueueV2() use different CUDA streams will lead undefined behavior.
    //!
    bool enqueueV2(void* const* bindings, cudaStream_t stream, cudaEvent_t* inputConsumed) noexcept {
        return mImpl->enqueueV2(bindings, stream, inputConsumed);
    }

    //!
    //! \brief Set an optimization profile for the current context with async
    //! semantics.
    //!
    //! \param profileIndex Index of the profile. The value must in the range of 0 and
    //!        getEngine().getNbOptimizationProfiles() - 1
    //!
    //! \param stream A cuda stream may be used by cudaMemcpyAsyncs, user could set null for now
    //!
    //! When an optimization profile is switched via this API, IXRT may be update context resource.
    //! Different optimization profile contain different dynamic range
    //! Every context will occupy unique profileIndex before all inference method
    //! Every context will called implicitly call setOptimizationProfile(0).
    //!
    //! \return true if the call succeeded, else false (e.g. input out of range)
    //!
    bool setOptimizationProfileAsync(int32_t profileIndex, cudaStream_t stream) noexcept {
        return mImpl->setOptimizationProfileAsync(profileIndex, stream);
    }

    //!
    //! \brief Get profiling result, profile result will store on disk as CSV file
    //! Profiling result contain time cost by kernels, and mapped operators
    //!
    bool getRunningProfiler() { return mImpl->getRunningProfiler(); }

    //!
    //! \brief Set memory address for network input or output tensor by name.
    //!
    //! \param tensorName The name of an input or output tensor.
    //! \param data The pointer (void*) to the data owned by the user.
    //!
    //! \return True on success, false if error occurred.
    //!
    //! An address defaults to nullptr.
    //! Pass data=nullptr to reset to the default state.
    //!
    //! Return false if the provided name does not map to an input or output tensor.
    //!
    //! The pointer must have at least 256-byte alignment.
    //!
    bool setTensorAddress(char const* tensorName, void* data) noexcept {
        return mImpl->setTensorAddress(tensorName, data);
    }

    //!
    //! \brief Get memory address binding to given input or output tensor, or nullptr if the provided name does not map
    //! to an input or output tensor.
    //!
    //! \param tensorName The name of an input or output tensor.
    //!
    void const* getTensorAddress(char const* tensorName) const noexcept { return mImpl->getTensorAddress(tensorName); }

    //!
    //! \brief Set memory address for given input.
    //!
    //! \param tensorName The name of an input tensor.
    //! \param data The pointer (void const*) to the const data owned by the user.
    //!
    //! \return True on success, false if the provided name does not map to an input tensor, does not meet alignment
    //! requirements, or some other error occurred.
    //!
    //! Input addresses can also be set using method setTensorAddress, which requires a (void*).
    //!
    //! See description of method setTensorAddress() for alignment and data type constraints.
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including the terminator.
    //!
    //! \see setTensorAddress()
    //!
    bool setInputTensorAddress(char const* tensorName, void const* data) noexcept {
        return mImpl->setInputTensorAddress(tensorName, data);
    }

    //!
    //! \brief Mark input as consumed.
    //!
    //! \param event The cuda event that is triggered after all input tensors have been consumed.
    //!
    //! \return True on success, false if error occurred.
    //!
    //! Passing event==nullptr removes whatever event was set, if any.
    //!
    bool setInputConsumedEvent(cudaEvent_t event) noexcept { return mImpl->setInputConsumedEvent(event); }

    //!
    //! \brief The event associated with consuming the input.
    //!
    //! \return The cuda event. Nullptr will be returned if the event is not set yet.
    //!
    cudaEvent_t getInputConsumedEvent() const noexcept { return mImpl->getInputConsumedEvent(); }

    //!
    //! \brief Enqueue inference on a stream.
    //!
    //! \param stream A cuda stream that kernels will be enqueued.
    //!
    //! \return True if the kernels were enqueued successfully, false otherwise.
    //!
    //! Modifying or releasing memory that has been bind for the tensors before stream
    //! synchronization or the event passed to setInputConsumedEvent has been being triggered results in undefined
    //! behavior.
    //! Input tensor can be released after the setInputConsumedEvent whereas output tensors require stream
    //! synchronization.
    //!
    bool enqueueV3(cudaStream_t stream) noexcept { return mImpl->enqueueV3(stream); }

    bool registerHook(AsciiChar const* name, ExecutionHook hook, int32_t flag) noexcept {
        return mImpl->registerHook(name, hook, flag);
    }

    bool deregisterHook(AsciiChar const* name) noexcept { return mImpl->deregisterHook(name); }
    //!
    //! \brief Set the ErrorRecorder for this interface
    //!
    //! Assigns the ErrorRecorder to object. This will call incRefCount() at least once. When an error occurs
    //! in the object, will transfer to the the error recorder. If set an nullptr, the previous error recorder
    //! will be called decRefCount(). If no ErrorRecorder set, will call global logger.
    //!
    //! \param recorder The error recorder to be registered.
    //
    //! \see getErrorRecorder()
    //!
    void setErrorRecorder(IErrorRecorder* recorder) noexcept { mImpl->setErrorRecorder(recorder); }

    //!
    //! \brief get the ErrorRecorder assigned to this interface.
    //!
    //! Retrieves the error recorder assigned to the object, returns to
    //! nullptr if no recorder assigned
    //!
    //! \return A pointer to the IErrorRecorder object that has been registered or nullptr.
    //!
    //! \see setErrorRecorder()
    //!
    IErrorRecorder* getErrorRecorder() const noexcept { return mImpl->getErrorRecorder(); }

   protected:
    apiv::VExecutionContext* mImpl;
};

//!
//! \class IOptimizationProfile
//! \brief Optimization profile for dynamic input dimensions and shape tensors.
//!
class IOptimizationProfile : public INoCopy {
   public:
    //!
    //! \brief Set the minimum / optimum / maximum dimensions for a dynamic input tensor.
    //!
    bool setDimensions(char const* inputName, OptProfileSelector select, Dims dims) noexcept {
        return mImpl->setDimensions(inputName, select, dims);
    }

    //!
    //! \brief Get the minimum / optimum / maximum dimensions for a dynamic input tensor.
    //!
    Dims getDimensions(char const* inputName, OptProfileSelector select) const noexcept {
        return mImpl->getDimensions(inputName, select);
    }

    //!
    //! \brief Check if the optimization profile is valid to be passed to an IBuilderConfig object.
    //!
    bool isValid() const noexcept { return mImpl->isValid(); }

   protected:
    apiv::VOptimizationProfile* mImpl;
    virtual ~IOptimizationProfile() noexcept = default;
};

//!
//! \class IRuntime
//! \brief Functionality for deserializing an engine and get to run
//!
class IRuntime : public INoCopy {
   public:
    virtual ~IRuntime() noexcept = default;
    //!
    //! \brief Deserialize engine from memory buffer
    //! \param blob The memory buffer
    //! \param size The byte size of the engine
    //! \return The engine, or nullptr if fail to deserialize
    //!
    ICudaEngine* deserializeCudaEngine(void const* blob, std::size_t size) noexcept {
        return mImpl->deserializeCudaEngine(blob, size, nullptr);
    }

    ICudaEngine* deserializeCudaEngine(IStreamReader& streamReader) {
        return mImpl->deserializeCudaEngine(streamReader);
    }

    ILogger* getLogger() const noexcept { return mImpl->getLogger(); }
    //!
    //! \brief Set the ErrorRecorder for this interface
    //!
    //! Assigns the ErrorRecorder to object. This will call incRefCount() at least once. When an error occurs
    //! in the object, will transfer to the the error recorder. If set an nullptr, the previous error recorder
    //! will be called decRefCount(). If no ErrorRecorder set, will call global logger.
    //!
    //! \param recorder The error recorder to be registered.
    //
    //! \see getErrorRecorder()
    //!
    void setErrorRecorder(IErrorRecorder* recorder) noexcept { mImpl->setErrorRecorder(recorder); }

    //!
    //! \brief get the ErrorRecorder assigned to this interface.
    //!
    //! Retrieves the error recorder assigned to the object, returns to
    //! nullptr if no recorder assigned
    //!
    //! \return A pointer to the IErrorRecorder object that has been registered or nullptr.
    //!
    //! \see setErrorRecorder()
    //!
    IErrorRecorder* getErrorRecorder() const noexcept { return mImpl->getErrorRecorder(); }

   protected:
    apiv::VRuntime* mImpl;
};

class IPluginV2DynamicExt : public IPluginV2Ext {
   public:
    IPluginV2DynamicExt* clone() const noexcept override = 0;

    virtual DimsExprs getOutputDimensions(int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs,
                                          IExprBuilder& exprBuilder) noexcept = 0;

    static constexpr int32_t kFORMAT_COMBINATION_LIMIT = 100;

    virtual bool supportsFormatCombination(int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs,
                                           int32_t nbOutputs) noexcept = 0;

    virtual void configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs,
                                 DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept = 0;

    virtual size_t getWorkspaceSize(PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs,
                                    int32_t nbOutputs) const noexcept = 0;

    virtual int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
                            void const* const* inputs, void* const* outputs, void* workspace,
                            cudaStream_t stream) noexcept = 0;

   protected:
    int32_t getTensorRTVersion() const noexcept override {
        return (static_cast<int32_t>(PluginVersion::kV2_DYNAMICEXT) << 24 | (NV_TENSORRT_VERSION & 0xFFFFFF));
    }

    virtual ~IPluginV2DynamicExt() noexcept {}

   private:
    void configurePlugin(Dims const*, int32_t, Dims const*, int32_t, DataType const*, DataType const*, bool const*,
                         bool const*, PluginFormat, int32_t) noexcept override final {}

    Dims getOutputDimensions(int32_t, Dims const*, int32_t) noexcept override final { return Dims{-1, {}}; }

    int32_t enqueue(int32_t batchSize, void const* const* inputs, void* const* outputs, void* workspace,
                    cudaStream_t stream) noexcept override {
        return 0;
    }
    bool supportsFormat(DataType type, PluginFormat format) const noexcept override { return false; }

    size_t getWorkspaceSize(int32_t maxBatchSize) const noexcept override { return 0; }
};

class IEngineObserver : public INoCopy {
   public:
    virtual ~IEngineObserver() noexcept = default;

    void saveEngineGraph(char const* graph_path) noexcept { mImpl->saveEngineGraph(graph_path); }

    char const* getNodeJson() noexcept { return mImpl->getNodeJson(); }

   protected:
    apiv::VEngineObserver* mImpl;
};  // class IEngineObserver

enum class LayerInformationFormat : int32_t {
    kONELINE = 0,  //!< Print layer information in one line per layer.
    kJSON = 1,     //!< Print layer information in JSON format.
};

class IEngineInspector : public INoCopy {
   public:
    virtual ~IEngineInspector() noexcept = default;

    bool setExecutionContext(IExecutionContext const* context) noexcept { return mImpl->setExecutionContext(context); }

    IExecutionContext const* getExecutionContext() const noexcept { return mImpl->getExecutionContext(); }

    char const* getLayerInformation(int32_t layerIndex, LayerInformationFormat format) const noexcept {
        return mImpl->getLayerInformation(layerIndex, format);
    }

    char const* getEngineInformation(LayerInformationFormat format) const noexcept {
        return mImpl->getEngineInformation(format);
    }

    void setErrorRecorder(IErrorRecorder* recorder) noexcept { mImpl->setErrorRecorder(recorder); }

    IErrorRecorder* getErrorRecorder() const noexcept { return mImpl->getErrorRecorder(); }

   protected:
    apiv::VEngineInspector* mImpl;
};  // class IEngineInspector

}  // namespace nvinfer1

extern "C" void* createInferRuntime_INTERNAL(void* logger, int32_t version) noexcept;

extern "C" nvinfer1::IPluginRegistry* getPluginRegistry() noexcept;

namespace nvinfer1 {

inline IRuntime* createInferRuntime(ILogger& logger) noexcept {
    return static_cast<IRuntime*>(createInferRuntime_INTERNAL(&logger, NV_TENSORRT_VERSION));
}

template <typename T>
class PluginRegistrar {
   public:
    PluginRegistrar() { getPluginRegistry()->registerCreator(instance, ""); }

   private:
    T instance{};
};

}  // namespace nvinfer1

#define REGISTER_TENSORRT_PLUGIN(name) \
    static nvinfer1::PluginRegistrar<name> pluginRegistrar##name {}
