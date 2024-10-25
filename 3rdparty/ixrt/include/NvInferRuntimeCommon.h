#pragma once
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "NvInferRuntimeBase.h"

#if __cplusplus >= 201402L
#define TRT_DEPRECATED [[deprecated]]
#endif

#define NV_TENSORRT_VERSION nvinfer1::kNV_TENSORRT_VERSION_IMPL

// Defines which symbols are exported
#ifdef TENSORRT_BUILD_LIB
#ifdef _MSC_VER
#define TENSORRTAPI __declspec(dllexport)
#else
#define TENSORRTAPI __attribute__((visibility("default")))
#endif
#else
#define TENSORRTAPI
#endif
#define TRTNOEXCEPT

namespace nvinfer1 {

//!
//! \enum DataType
//! \brief Data type enum definition
//!
enum class DataType : int32_t {
    //! Unknown
    kUNKNOWN = 0,
    //! 8-bit integer format.
    kINT8 = 1,
    //! 16-bit floating point format.
    kHALF = 2,
    //! 32-bit floating point format.
    kFLOAT = 3,
    //! 32-bit integer format.
    kINT32 = 4,
    //! 64-bit integer format.
    kINT64 = 5,
    //! 64-bit floating point format.
    kFLOAT64 = 6,
    //! 8-bit bool format.
    kBOOL = 7,
    //! 8-bit unsigned integer format.
    kUINT8 = 8,
    //! Brain float -- has an 8 bit exponent and 8 bit significand.
    kBF16 = 9,
    //! Nvidia FP8, 1 sign bit, 4 exponent bits, 3 mantissa bits, and exponent-bias 7.
    kFP8 = 10,
};

static constexpr int32_t kNV_TENSORRT_VERSION_IMPL =
    (NV_TENSORRT_MAJOR * 1000) + (NV_TENSORRT_MINOR * 100) + NV_TENSORRT_PATCH;

using char_t = char;

using AsciiChar = char_t;

namespace impl {

template <typename T>
struct EnumMaxImpl;

template <>
struct EnumMaxImpl<DataType> {
    static constexpr int32_t kVALUE = 5;
};
}  // namespace impl

template <typename T>
constexpr int32_t EnumMax() noexcept {
    return impl::EnumMaxImpl<T>::kVALUE;
}

//!
//! \class Dims
//! \brief Structure to define the dimensions descriptor for tensor.
//!
//! IxRT could return an invalid dims object that represented by nbDims == -1 and d[i] == 0 for all d.
//!
class Dims32 {
   public:
    //! The maximum number of dimensions supported by tensor.
    static constexpr int32_t MAX_DIMS{8};
    //! Number of dimensions).
    int32_t nbDims;
    //! The content of each dimension.
    int32_t d[MAX_DIMS];
};

//! TensorRT 10.1.0
//! \class Dims
//! \brief Structure to define the dimensions of a tensor.
//!
//! TensorRT can also return an "invalid dims" structure. This structure is
//! represented by nbDims == -1 and d[i] == 0 for all i.
//!
//! TensorRT can also return an "unknown rank" dims structure. This structure is
//! represented by nbDims == -1 and d[i] == -1 for all i.
//!
class Dims64
{
public:
    //! The maximum rank (number of dimensions) supported for a tensor.
    static constexpr int32_t MAX_DIMS{8};

    //! The rank (number of dimensions).
    int32_t nbDims;

    //! The extent of each dimension.
    int64_t d[MAX_DIMS];
};

//!
//! Alias for Dims32.
//!
using Dims = Dims32;

//!
//! \enum TensorFormat
//! \brief Memory format of  tensors.
//!
enum class TensorFormat : int32_t {
    kUNKNOWN = -1,
    //! linear memory format.
    kLINEAR = 0,
    //! channel-last memory format.
    kHWC = 8,

    //! This format is bound to FP16
    kHWC32 = 20,
    //! This format is bound to int8
    kHWC64 = 21,
};

using PluginFormat = TensorFormat;

enum class PluginFieldType : int32_t {

    kFLOAT16 = 0,

    kFLOAT32 = 1,

    kFLOAT64 = 2,

    kINT8 = 3,

    kINT16 = 4,

    kINT32 = 5,

    kCHAR = 6,

    kDIMS = 7,

    kBF16 = 9,

    kFP8 = 10,

    kUNKNOWN = 11
};

class PluginField {
   public:
    AsciiChar const* name;

    void const* data;

    PluginFieldType type;

    int32_t length;

    PluginField(AsciiChar const* const name_ = nullptr, void const* const data_ = nullptr,
                PluginFieldType const type_ = PluginFieldType::kUNKNOWN, int32_t const length_ = 0) noexcept
        : name(name_), data(data_), type(type_), length(length_) {}
};

struct PluginFieldCollection {
    int32_t nbFields;

    PluginField const* fields;
};

struct PluginTensorDesc {
    Dims dims;

    DataType type;

    TensorFormat format;

    float scale;
};

enum class PluginVersion : uint8_t {

    kV2 = 0,

    kV2_EXT = 1,

    kV2_IOEXT = 2,

    kV2_DYNAMICEXT = 3,
};

//!
//! \class ILogger
//!
//! \brief User-implemented logging interface for the builder, refitter and runtime.
//!
//! The logger will be referenced by IBuilder, IRuntime and the objects created by that interface , such like
//! IBuilderConfig, INetworkDefinition, ICudaEngine，IHostMemory, IExecutionContext. The logger need to keep alive until
//! all related object are released.
//!
//! The User-implemented logger object should be thread safe.
//! IXRT will not hold any synchronization primitives while using the interface functions.
//!
class ILogger {
   public:
    //!
    //! \enum Severity
    //!
    //! The severity for every log message.
    //!
    enum class Severity : int32_t {
        //! Fatal error, execution is broken
        kINTERNAL_ERROR = 0,
        //! Application error, execution will not accomplish task as expect
        kERROR = 1,
        //! Application error, but IXRT will use default setting to execute task
        kWARNING = 2,
        //!  Notation messages with routine procedure information.
        kINFO = 3,
        //!  Detail messages from debugging information.
        kVERBOSE = 4,
    };
    //!
    //! User implemented callback function to handle logging output;
    //!
    //! \param severity The severity of the message.
    //! \param msg A null-terminated log content.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!  This method may be call from multi-thread while multi execution context used in different thread
    //!  Or this method will be call by different type of object, seems like runtime, builder
    //!
    virtual void log(Severity severity, AsciiChar const* msg) noexcept = 0;

    ILogger() = default;
    virtual ~ILogger() = default;

   protected:
    ILogger(ILogger const&) = default;
    ILogger(ILogger&&) = default;
    ILogger& operator=(ILogger const&) & = default;
    ILogger& operator=(ILogger&&) & = default;
};

using AllocatorFlags = uint32_t;

class IGpuAllocator {
   public:
    virtual void* allocate(uint64_t const size, uint64_t const alignment, AllocatorFlags const flags) noexcept = 0;

    virtual ~IGpuAllocator() = default;
    IGpuAllocator() = default;

    virtual void* reallocate(void*, uint64_t, uint64_t) noexcept { return nullptr; }

    virtual bool deallocate(void* const memory) noexcept = 0;

   protected:
    IGpuAllocator(IGpuAllocator const&) = default;
    IGpuAllocator(IGpuAllocator&&) = default;
    IGpuAllocator& operator=(IGpuAllocator const&) & = default;
    IGpuAllocator& operator=(IGpuAllocator&&) & = default;
};

class IPluginV2 {
   public:
    virtual int32_t getTensorRTVersion() const noexcept { return NV_TENSORRT_VERSION; }

    virtual AsciiChar const* getPluginType() const noexcept = 0;

    virtual AsciiChar const* getPluginVersion() const noexcept = 0;

    virtual int32_t getNbOutputs() const noexcept = 0;

    virtual Dims getOutputDimensions(int32_t index, Dims const* inputs, int32_t nbInputDims) noexcept = 0;

    virtual bool supportsFormat(DataType type, PluginFormat format) const noexcept = 0;

    virtual void configureWithFormat(Dims const* inputDims, int32_t nbInputs, Dims const* outputDims, int32_t nbOutputs,
                                     DataType type, PluginFormat format, int32_t maxBatchSize) noexcept = 0;

    virtual int32_t initialize() noexcept = 0;

    virtual void terminate() noexcept = 0;

    virtual size_t getWorkspaceSize(int32_t maxBatchSize) const noexcept = 0;

    virtual int32_t enqueue(int32_t batchSize, void const* const* inputs, void* const* outputs, void* workspace,
                            cudaStream_t stream) noexcept = 0;

    virtual size_t getSerializationSize() const noexcept = 0;

    virtual void serialize(void* buffer) const noexcept = 0;

    virtual void destroy() noexcept = 0;

    virtual IPluginV2* clone() const noexcept = 0;

    virtual void setPluginNamespace(AsciiChar const* pluginNamespace) noexcept = 0;

    virtual AsciiChar const* getPluginNamespace() const noexcept = 0;

    IPluginV2() = default;
    virtual ~IPluginV2() noexcept = default;

   protected:
    IPluginV2(IPluginV2 const&) = default;
    IPluginV2(IPluginV2&&) = default;
    IPluginV2& operator=(IPluginV2 const&) & = default;
    IPluginV2& operator=(IPluginV2&&) & = default;
};

class IPluginV2Ext : public IPluginV2 {
   public:
    virtual DataType getOutputDataType(int32_t index, DataType const* inputTypes, int32_t nbInputs) const noexcept = 0;

    virtual void configurePlugin(Dims const* inputDims, int32_t nbInputs, Dims const* outputDims, int32_t nbOutputs,
                                 DataType const* inputTypes, DataType const* outputTypes, bool const* inputIsBroadcast,
                                 bool const* outputIsBroadcast, PluginFormat floatFormat,
                                 int32_t maxBatchSize) noexcept = 0;

    IPluginV2Ext() = default;
    ~IPluginV2Ext() override = default;

    IPluginV2Ext* clone() const noexcept override = 0;

   protected:
    IPluginV2Ext(IPluginV2Ext const&) = default;
    IPluginV2Ext(IPluginV2Ext&&) = default;
    IPluginV2Ext& operator=(IPluginV2Ext const&) & = default;
    IPluginV2Ext& operator=(IPluginV2Ext&&) & = default;

    int32_t getTensorRTVersion() const noexcept override {
        return static_cast<int32_t>((static_cast<uint32_t>(PluginVersion::kV2_EXT) << 24U) |
                                    (static_cast<uint32_t>(NV_TENSORRT_VERSION) & 0xFFFFFFU));
    }

    void configureWithFormat(Dims const*, int32_t, Dims const*, int32_t, DataType, PluginFormat,
                             int32_t) noexcept override {}
};

class IPluginV2IOExt : public IPluginV2Ext {
   public:
    virtual void configurePlugin(PluginTensorDesc const* in, int32_t nbInput, PluginTensorDesc const* out,
                                 int32_t nbOutput) noexcept = 0;

    virtual bool supportsFormatCombination(int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs,
                                           int32_t nbOutputs) const noexcept = 0;

    IPluginV2IOExt() = default;
    ~IPluginV2IOExt() override = default;

   protected:
    IPluginV2IOExt(IPluginV2IOExt const&) = default;
    IPluginV2IOExt(IPluginV2IOExt&&) = default;
    IPluginV2IOExt& operator=(IPluginV2IOExt const&) & = default;
    IPluginV2IOExt& operator=(IPluginV2IOExt&&) & = default;

    int32_t getTensorRTVersion() const noexcept override {
        return static_cast<int32_t>((static_cast<uint32_t>(PluginVersion::kV2_IOEXT) << 24U) |
                                    (static_cast<uint32_t>(NV_TENSORRT_VERSION) & 0xFFFFFFU));
    }
};

class IPluginCreator {
   public:
    virtual int32_t getTensorRTVersion() const noexcept { return NV_TENSORRT_VERSION; }

    virtual AsciiChar const* getPluginName() const noexcept = 0;

    virtual AsciiChar const* getPluginVersion() const noexcept = 0;

    virtual PluginFieldCollection const* getFieldNames() noexcept = 0;

    virtual IPluginV2* createPlugin(AsciiChar const* name, PluginFieldCollection const* fc) noexcept = 0;

    virtual IPluginV2* deserializePlugin(AsciiChar const* name, void const* serialData,
                                         size_t serialLength) noexcept = 0;

    virtual void setPluginNamespace(AsciiChar const* pluginNamespace) noexcept = 0;

    virtual AsciiChar const* getPluginNamespace() const noexcept = 0;

    IPluginCreator() = default;
    virtual ~IPluginCreator() = default;

   protected:
    IPluginCreator(IPluginCreator const&) = default;
    IPluginCreator(IPluginCreator&&) = default;
    IPluginCreator& operator=(IPluginCreator const&) & = default;
    IPluginCreator& operator=(IPluginCreator&&) & = default;
};

class IPluginRegistry {
   public:
    virtual bool registerCreator(IPluginCreator& creator, AsciiChar const* const pluginNamespace) noexcept = 0;

    virtual IPluginCreator* const* getPluginCreatorList(int32_t* const numCreators) const noexcept = 0;

    virtual IPluginCreator* getPluginCreator(AsciiChar const* const pluginName, AsciiChar const* const pluginVersion,
                                             AsciiChar const* const pluginNamespace = "") noexcept = 0;

    virtual bool deregisterCreator(IPluginCreator const& creator) noexcept = 0;

    IPluginRegistry() = default;
    IPluginRegistry(IPluginRegistry const&) = delete;
    IPluginRegistry(IPluginRegistry&&) = delete;
    IPluginRegistry& operator=(IPluginRegistry const&) & = delete;
    IPluginRegistry& operator=(IPluginRegistry&&) & = delete;

   protected:
    virtual ~IPluginRegistry() noexcept = default;
};

//!
//! \class ILoggerFinder
//!
//! \brief A virtual base class to find a logger.
//! Allows a plugin to find an instance of a logger if it needs to emit a log message.
//! A pointer to an instance of this class is passed to a plugin shared library on initialization when that plugin
//! is serialized as part of a version-compatible plan. See the plugin chapter in the developer guide for details.
//!
class ILoggerFinder {
   public:
    //!
    //! \brief Get the logger used by the engine or execution context which called the plugin method.
    //!
    //! \warning Must be called from the thread in which the plugin method was called.
    //!
    //! \return A pointer to the logger.
    //!
    virtual ILogger* findLogger() = 0;

   protected:
    virtual ~ILoggerFinder() = default;
};

}  // namespace nvinfer1
