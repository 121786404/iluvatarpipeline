
#pragma once
#include "NvInferLegacyDims.h"
#include "NvInferRuntimeCommon.h"
namespace nvinfer1 {

class IActivationLayer;
class ILayer;
class ITensor;
class INetworkDefinition;
class IBuilderConfig;
class ICudaEngine;
class IOptimizationProfile;
class IHostMemory;
class IExecutionContext;
class IDimensionExpr;
class IEngineObserver;
class IConvolutionLayer;
class IElementWiseLayer;
class IGatherLayer;
class IUnaryLayer;
class IPoolingLayer;
class IIdentityLayer;
class IReduceLayer;
class IParametricReLULayer;
class ISoftMaxLayer;
class IResizeLayer;
class IGridSampleLayer;
class ISliceLayer;
class IConstantLayer;
class ICastLayer;
class ITopKLayer;
class Weights;
class IPluginV2Layer;
class IConcatenationLayer;
class ISelectLayer;
class IInt8Calibrator;
class IMatrixMultiplyLayer;
class IQuantizeLayer;
class IDequantizeLayer;
class IFillLayer;
class INormalizationLayer;
class IEinsumLayer;
struct Permutation;
class IShuffleLayer;
class IShapeLayer;
class IDeconvolutionLayer;
class IFullyConnectedLayer;
class IAssertionLayer;
class ITimingCache;
class ExecutionContextInfo;
class IEngineInspector;

enum class ActivationType : int32_t;
using BuilderFlags = uint32_t;
enum class BuilderFlag : int32_t;
enum class PreviewFeature : int32_t;
enum class LayerType : int32_t;
enum class TensorLocation : int32_t;
enum class OptProfileSelector : int32_t;
enum class DimensionOperation : int32_t;
enum class PaddingMode : int32_t;
enum class ReduceOperation : int32_t;
enum class UnaryOperation : int32_t;
enum class ElementWiseOperation : int32_t;
enum class PoolingType : int32_t;
enum class GatherMode : int32_t;
enum class ResizeCoordinateTransformation : int32_t;
enum class ResizeMode : int32_t;
enum class ResizeRoundMode : int32_t;
enum class ResizeSelector : int32_t;
enum class MatrixOperation : int32_t;
enum class FillOperation : int32_t;
enum class SliceMode : int32_t;
enum class TopKOperation : int32_t;
enum class SampleMode : int32_t;
enum class InterpolationMode : int32_t;
using NetworkDefinitionCreationFlags = uint32_t;
enum class ExecutionHookFlag : int32_t;
enum class LayerInformationFormat : int32_t;
enum class ProfilingVerbosity : int32_t;
typedef void (*ExecutionHook)(ExecutionContextInfo const*);

namespace apiv {
class VRoot {
   public:
    virtual ~VRoot() noexcept = default;
};

class VEngineInspector : public VRoot {
   public:
    virtual IEngineInspector* getPImpl() noexcept = 0;
    virtual bool setExecutionContext(IExecutionContext const* context) noexcept = 0;
    virtual IExecutionContext const* getExecutionContext() const noexcept = 0;
    virtual char const* getLayerInformation(int32_t layerIndex, LayerInformationFormat format) const noexcept = 0;
    virtual char const* getEngineInformation(LayerInformationFormat format) const noexcept = 0;
    virtual void setErrorRecorder(IErrorRecorder* recorder) noexcept = 0;
    virtual IErrorRecorder* getErrorRecorder() const noexcept = 0;
};

class VHostMemory : public VRoot {
   public:
    virtual void* data() const noexcept = 0;
    virtual std::size_t size() const noexcept = 0;
    virtual DataType type() const noexcept = 0;
};

class VDimensionExpr : public VRoot {
   public:
    virtual bool isConstant() const = 0;
    virtual int32_t getConstantValue() const = 0;
};

class VExprBuilder : public VRoot {
   public:
    virtual IDimensionExpr const* constant(int32_t value) = 0;
    virtual IDimensionExpr const* operation(DimensionOperation op, IDimensionExpr const& first,
                                            IDimensionExpr const& second) = 0;
};

class VBuilder : public VRoot {
   public:
    virtual bool platformHasFastFp16() const noexcept = 0;
    virtual bool platformHasFastInt8() const noexcept = 0;
    virtual void setGpuAllocator(IGpuAllocator* allocator) noexcept = 0;
    virtual IBuilderConfig* createBuilderConfig() noexcept = 0;
    virtual INetworkDefinition* createNetworkV2(NetworkDefinitionCreationFlags flags) noexcept = 0;
    virtual IOptimizationProfile* createOptimizationProfile() noexcept = 0;
    virtual void reset() noexcept = 0;
    virtual IHostMemory* buildSerializedNetwork(INetworkDefinition& network, IBuilderConfig& config) noexcept = 0;
    virtual bool isNetworkSupported(INetworkDefinition const& network, IBuilderConfig const& config) const noexcept = 0;
    virtual ILogger* getLogger() const noexcept = 0;
    virtual bool setMaxThreads(int32_t maxThreads) noexcept = 0;
    virtual int32_t getMaxThreads() const noexcept = 0;
    virtual void setErrorRecorder(IErrorRecorder* recorder) noexcept = 0;
    virtual IErrorRecorder* getErrorRecorder() const noexcept = 0;
};

class VNetworkDefinition : public VRoot {
   public:
    virtual int32_t getNbLayers() const noexcept = 0;
    virtual ILayer* getLayer(int32_t index) const noexcept = 0;
    virtual int32_t getNbInputs() const noexcept = 0;
    virtual ITensor* getInput(int32_t index) const noexcept = 0;
    virtual int32_t getNbOutputs() const noexcept = 0;
    virtual ITensor* getOutput(int32_t index) const noexcept = 0;
    virtual void setName(char const* name) noexcept = 0;
    virtual char const* getName() const noexcept = 0;
    virtual ITensor* addInput(char const* name, DataType type, Dims dimensions) noexcept = 0;
    virtual void markOutput(ITensor& tensor) noexcept = 0;
    virtual IConvolutionLayer* addConvolution(ITensor& input, int32_t nbOutputMaps, DimsHW kernelSize,
                                              Weights kernelWeights, Weights biasWeights) noexcept = 0;
    virtual IConvolutionLayer* addConvolutionNd(ITensor& input, int32_t nbOutputMaps, Dims kernelSize,
                                                Weights kernelWeights, Weights biasWeights) noexcept = 0;
    virtual ISoftMaxLayer* addSoftMax(ITensor& input) noexcept = 0;
    virtual IReduceLayer* addReduce(ITensor& input, ReduceOperation operation, uint32_t reduceAxes,
                                    bool keepDimensions) noexcept = 0;
    virtual IUnaryLayer* addUnary(ITensor& input, UnaryOperation operation) noexcept = 0;
    virtual IElementWiseLayer* addElementWise(ITensor& input1, ITensor& input2, ElementWiseOperation op) noexcept = 0;
    virtual IPluginV2Layer* addPluginV2(ITensor* const* inputs, int32_t nbInputs, IPluginV2& plugin) noexcept = 0;
    virtual IPoolingLayer* addPoolingNd(ITensor& input, PoolingType type, Dims windowSize) noexcept = 0;
    virtual IActivationLayer* addActivation(ITensor& input, ActivationType type) noexcept = 0;
    virtual IResizeLayer* addResize(ITensor& input) noexcept = 0;
    virtual IParametricReLULayer* addParametricReLU(ITensor& input, ITensor& slope) noexcept = 0;
    virtual IConcatenationLayer* addConcatenation(ITensor* const* inputs, int32_t nbInputs) noexcept = 0;
    virtual IGatherLayer* addGather(ITensor& data, ITensor& indices, int32_t axis) noexcept = 0;
    virtual IGatherLayer* addGatherV2(ITensor& data, ITensor& indices, GatherMode mode) noexcept = 0;
    virtual IIdentityLayer* addIdentity(ITensor& input) noexcept = 0;
    virtual IMatrixMultiplyLayer* addMatrixMultiply(ITensor& input0, MatrixOperation op0, ITensor& input1,
                                                    MatrixOperation op1) noexcept = 0;
    virtual IDequantizeLayer* addDequantize(ITensor& input, ITensor& scale) noexcept = 0;
    virtual IQuantizeLayer* addQuantize(ITensor& input, ITensor& scale) noexcept = 0;
    virtual IFillLayer* addFill(Dims dimensions, FillOperation op) noexcept = 0;
    virtual ISelectLayer* addSelect(ITensor& condition, ITensor& thenInput, ITensor& elseInput) noexcept = 0;
    virtual ISliceLayer* addSlice(ITensor& input, Dims start, Dims size, Dims stride) noexcept = 0;
    virtual IConstantLayer* addConstant(Dims dimensions, Weights weights) noexcept = 0;
    virtual INormalizationLayer* addNormalization(ITensor& input, ITensor& scale, ITensor& bias,
                                                  uint32_t axesMask) noexcept = 0;
    virtual IShuffleLayer* addShuffle(ITensor& input) noexcept = 0;
    virtual ICastLayer* addCast(ITensor& input, DataType toType) noexcept = 0;
    virtual IEinsumLayer* addEinsum(ITensor* const* inputs, int32_t nbInputs, char const* equation) noexcept = 0;
    virtual ITopKLayer* addTopK(ITensor& input, TopKOperation op, int32_t k, uint32_t reduceAxes) noexcept = 0;
    virtual IGridSampleLayer* addGridSample(ITensor& input, ITensor& grid) noexcept = 0;
    virtual IShapeLayer* addShape(ITensor& input) noexcept = 0;
    virtual IDeconvolutionLayer* addDeconvolutionNd(ITensor& input, int32_t nbOutputMaps, Dims kernelSize,
                                                    Weights kernelWeights, Weights biasWeights) noexcept = 0;
    virtual IFullyConnectedLayer* addFullyConnected(ITensor& input, int32_t nbOutputs, Weights kernelWeights,
                                                    Weights biasWeights) noexcept = 0;
    virtual IAssertionLayer* addAssertion(ITensor& condition, char const* message) noexcept = 0;
    virtual void setErrorRecorder(IErrorRecorder* recorder) noexcept = 0;
    virtual IErrorRecorder* getErrorRecorder() const noexcept = 0;
};

class VTensor : public VRoot {
   public:
    virtual char const* getName() const noexcept = 0;
    virtual void setName(char const* name) noexcept = 0;
    virtual Dims getDimensions() const noexcept = 0;
    virtual void setDimensions(Dims dimensions) const noexcept = 0;
    virtual void setType(DataType type) noexcept = 0;
    virtual DataType getType() const noexcept = 0;
    virtual bool isNetworkInput() const noexcept = 0;
    virtual bool isNetworkOutput() const noexcept = 0;
    virtual TensorLocation getLocation() const noexcept = 0;
    virtual void setLocation(TensorLocation location) noexcept = 0;
    virtual void setDimensionName(int32_t index, char const* name) noexcept = 0;
    virtual char const* getDimensionName(int32_t index) const noexcept = 0;
    virtual bool isShapeTensor() const noexcept = 0;
    virtual bool isExecutionTensor() const noexcept = 0;
    virtual bool setDynamicRange(float min, float max) noexcept = 0;
    virtual bool dynamicRangeIsSet() const noexcept = 0;
    virtual void resetDynamicRange() noexcept = 0;
    virtual float getDynamicRangeMin() const noexcept = 0;
    virtual float getDynamicRangeMax() const noexcept = 0;
};

class VLayer : public VRoot {
   public:
    virtual LayerType getType() const noexcept = 0;
    virtual void setName(char const* name) noexcept = 0;
    virtual char const* getName() const noexcept = 0;
    virtual int32_t getNbInputs() const noexcept = 0;
    virtual ITensor* getInput(int32_t index) const noexcept = 0;
    virtual int32_t getNbOutputs() const noexcept = 0;
    virtual ITensor* getOutput(int32_t index) const noexcept = 0;
    virtual void setInput(int32_t index, ITensor& tensor) noexcept = 0;
    virtual void setPrecision(DataType dataType) noexcept = 0;
    virtual void setOutputType(int32_t index, DataType dataType) noexcept = 0;
    virtual DataType getPrecision() const noexcept = 0;
    virtual DataType getOutputType(int32_t index) const noexcept = 0;
};

class VTimingCache : public VRoot {
   public:
    virtual nvinfer1::IHostMemory* serialize() const noexcept = 0;
    virtual bool combine(ITimingCache const& inputCache, bool ignoreMismatch) noexcept = 0;
    virtual bool reset() noexcept = 0;
};

class VBuilderConfig : public VRoot {
   public:
    virtual void setFlags(BuilderFlags builderFlags) noexcept = 0;
    virtual BuilderFlags getFlags() const noexcept = 0;
    virtual void clearFlag(BuilderFlag builderFlag) noexcept = 0;
    virtual void setFlag(BuilderFlag builderFlag) noexcept = 0;
    virtual bool getFlag(BuilderFlag builderFlag) const noexcept = 0;
    virtual void reset() noexcept = 0;
    virtual int32_t addOptimizationProfile(IOptimizationProfile const* profile) noexcept = 0;
    virtual int32_t getNbOptimizationProfiles() const noexcept = 0;
    virtual void setProfilingVerbosity(ProfilingVerbosity verbosity) noexcept = 0;
    virtual IInt8Calibrator* getInt8Calibrator() const noexcept = 0;
    virtual void setInt8Calibrator(IInt8Calibrator* calibrator) noexcept = 0;
    virtual nvinfer1::ITimingCache* createTimingCache(void const* blob, std::size_t size) const noexcept = 0;
    virtual bool setTimingCache(ITimingCache const& cache, bool ignoreMismatch) noexcept = 0;
    virtual nvinfer1::ITimingCache const* getTimingCache() const noexcept = 0;
    virtual void setPreviewFeature(PreviewFeature feature, bool enable) noexcept = 0;
    virtual bool getPreviewFeature(PreviewFeature feature) const noexcept = 0;
    virtual ProfilingVerbosity getProfilingVerbosity() const noexcept = 0;
};

class VOptimizationProfile : public VRoot {
   public:
    virtual bool setDimensions(char const* inputName, OptProfileSelector select, Dims dims) noexcept = 0;
    virtual Dims getDimensions(char const* inputName, OptProfileSelector select) const noexcept = 0;
    virtual bool isValid() const noexcept = 0;
};

class VCudaEngine : public VRoot {
   public:
    virtual int32_t getNbBindings() const noexcept = 0;
    virtual int32_t getBindingIndex(char const* name) const noexcept = 0;
    virtual char const* getBindingName(int32_t bindingIndex) const noexcept = 0;
    virtual bool bindingIsInput(int32_t bindingIndex) const noexcept = 0;
    virtual Dims getBindingDimensions(int32_t bindingIndex) const noexcept = 0;
    virtual DataType getBindingDataType(int32_t bindingIndex) const noexcept = 0;
    virtual int32_t getNbLayers() const noexcept = 0;
    virtual IHostMemory* serialize() const noexcept = 0;
    virtual IExecutionContext* createExecutionContext() noexcept = 0;
    virtual IExecutionContext* createExecutionContextWithoutDeviceMemory() noexcept = 0;
    virtual size_t getDeviceMemorySize() const noexcept = 0;
    virtual TensorFormat getBindingFormat(int32_t bindingIndex) const noexcept = 0;
    virtual char const* getBindingFormatDesc(int32_t bindingIndex) const noexcept = 0;
    virtual char const* getName() const noexcept = 0;
    virtual int32_t getNbOptimizationProfiles() const noexcept = 0;
    virtual Dims getProfileDimensions(int32_t bindingIndex, int32_t profileIndex,
                                      OptProfileSelector select) const noexcept = 0;
    virtual Dims getProfileShape(char const* tensorName, int32_t profileIndex,
                                 OptProfileSelector select) const noexcept = 0;
    virtual IEngineObserver* createEngineObserver() noexcept = 0;
    virtual IEngineInspector* createEngineInspector() const noexcept = 0;
    virtual int32_t getNbIOTensors() const noexcept = 0;
    virtual char const* getIOTensorName(int32_t index) const noexcept = 0;
    virtual TensorIOMode getTensorIOMode(char const* tensorName) const noexcept = 0;
    virtual DataType getTensorDataType(char const* tensorName) const noexcept = 0;
    virtual Dims getTensorShape(char const* tensorName) const noexcept = 0;

    virtual void setErrorRecorder(IErrorRecorder* recorder) noexcept = 0;
    virtual IErrorRecorder* getErrorRecorder() const noexcept = 0;
    virtual ProfilingVerbosity getProfilingVerbosity() const noexcept = 0;
};

class VRuntime : public VRoot {
   public:
    virtual ICudaEngine* deserializeCudaEngine(void const* blob, std::size_t size, void* pluginFactory) noexcept = 0;
    virtual ICudaEngine* deserializeCudaEngine(IStreamReader& streamReader) noexcept = 0;
    virtual ILogger* getLogger() const noexcept = 0;
    virtual void setErrorRecorder(IErrorRecorder* recorder) noexcept = 0;
    virtual IErrorRecorder* getErrorRecorder() const noexcept = 0;
};

class VExecutionContext : public VRoot {
   public:
    virtual ICudaEngine const& getEngine() const noexcept = 0;
    virtual void setName(char const* name) noexcept = 0;
    virtual char const* getName() const noexcept = 0;
    virtual void setDeviceMemory(void* memory) noexcept = 0;
    virtual int32_t getOptimizationProfile() const noexcept = 0;
    virtual bool setBindingDimensions(int32_t bindingIndex, Dims dimensions) noexcept = 0;
    virtual Dims getBindingDimensions(int32_t bindingIndex) const noexcept = 0;
    virtual bool setInputShapeBinding(int32_t bindingIndex, int32_t const* data) noexcept = 0;
    virtual bool getShapeBinding(int32_t bindingIndex, int32_t* data) const noexcept = 0;
    virtual bool setInputShape(char const* tensorName, Dims const& dims) noexcept = 0;
    virtual Dims getTensorShape(char const* tensorName) const noexcept = 0;
    virtual bool allInputDimensionsSpecified() const noexcept = 0;
    virtual bool allInputShapesSpecified() const noexcept = 0;
    virtual bool executeV2(void* const* bindings) noexcept = 0;
    virtual bool enqueueV2(void* const* bindings, cudaStream_t stream, cudaEvent_t* inputConsumed) noexcept = 0;
    virtual bool setOptimizationProfileAsync(int32_t profileIndex, cudaStream_t stream) noexcept = 0;
    virtual bool getRunningProfiler() const noexcept = 0;
    virtual bool setTensorAddress(char const* tensorName, void* data) noexcept = 0;
    virtual void const* getTensorAddress(char const* tensorName) const noexcept = 0;
    virtual bool setInputTensorAddress(char const* tensorName, void const* data) noexcept = 0;
    virtual bool setInputConsumedEvent(cudaEvent_t event) noexcept = 0;
    virtual cudaEvent_t getInputConsumedEvent() const noexcept = 0;
    virtual bool enqueueV3(cudaStream_t stream) noexcept = 0;
    virtual bool registerHook(AsciiChar const* name, ExecutionHook hook, int32_t flag) noexcept = 0;
    virtual bool deregisterHook(AsciiChar const* name) noexcept = 0;
    virtual void setErrorRecorder(IErrorRecorder* recorder) noexcept = 0;
    virtual IErrorRecorder* getErrorRecorder() const noexcept = 0;
};

class VEngineObserver : public VRoot {
   public:
    virtual void saveEngineGraph(char const* graph_path) noexcept = 0;

    virtual char const* getNodeJson() noexcept = 0;
};

class VConvolutionLayer : public VRoot {
   public:
    virtual void setNbOutputMaps(int32_t nbOutputMaps) noexcept = 0;
    virtual int32_t getNbOutputMaps() const noexcept = 0;
    virtual void setNbGroups(int32_t nbGroups) noexcept = 0;
    virtual int32_t getNbGroups() const noexcept = 0;
    //  virtual void setKernelWeights(Weights weights) noexcept = 0;
    //  virtual Weights getKernelWeights() const noexcept = 0;
    //  virtual void setBiasWeights(Weights weights) noexcept = 0;
    //  virtual Weights getBiasWeights() const noexcept = 0;
    virtual void setPrePadding(Dims padding) noexcept = 0;
    virtual Dims getPrePadding() const noexcept = 0;
    virtual void setPostPadding(Dims padding) noexcept = 0;
    virtual Dims getPostPadding() const noexcept = 0;
    virtual void setPaddingMode(PaddingMode paddingMode) noexcept = 0;
    virtual PaddingMode getPaddingMode() const noexcept = 0;
    virtual void setKernelSizeNd(Dims kernelSize) noexcept = 0;
    virtual Dims getKernelSizeNd() const noexcept = 0;
    virtual void setStrideNd(Dims stride) noexcept = 0;
    virtual Dims getStrideNd() const noexcept = 0;
    virtual void setPaddingNd(Dims padding) noexcept = 0;
    virtual Dims getPaddingNd() const noexcept = 0;
    virtual void setDilationNd(Dims dilation) noexcept = 0;
    virtual Dims getDilationNd() const noexcept = 0;
};

class VDeconvolutionLayer : public VRoot {
   public:
    virtual void setNbOutputMaps(int32_t nbOutputMaps) noexcept = 0;
    virtual int32_t getNbOutputMaps() const noexcept = 0;
    virtual void setNbGroups(int32_t nbGroups) noexcept = 0;
    virtual int32_t getNbGroups() const noexcept = 0;
    //  virtual void setKernelWeights(Weights weights) noexcept = 0;
    //  virtual Weights getKernelWeights() const noexcept = 0;
    //  virtual void setBiasWeights(Weights weights) noexcept = 0;
    //  virtual Weights getBiasWeights() const noexcept = 0;
    virtual void setPrePadding(Dims padding) noexcept = 0;
    virtual Dims getPrePadding() const noexcept = 0;
    virtual void setPostPadding(Dims padding) noexcept = 0;
    virtual Dims getPostPadding() const noexcept = 0;
    virtual void setPaddingMode(PaddingMode paddingMode) noexcept = 0;
    virtual PaddingMode getPaddingMode() const noexcept = 0;
    virtual void setKernelSizeNd(Dims kernelSize) noexcept = 0;
    virtual Dims getKernelSizeNd() const noexcept = 0;
    virtual void setStrideNd(Dims stride) noexcept = 0;
    virtual Dims getStrideNd() const noexcept = 0;
    virtual void setPaddingNd(Dims padding) noexcept = 0;
    virtual Dims getPaddingNd() const noexcept = 0;
    virtual void setDilationNd(Dims dilation) noexcept = 0;
    virtual Dims getDilationNd() const noexcept = 0;
};

class VActivationLayer : public VRoot {
   public:
    virtual void setActivationType(ActivationType type) noexcept = 0;
    virtual ActivationType getActivationType() const noexcept = 0;
    virtual void setAlpha(float alpha) noexcept = 0;
    virtual void setBeta(float beta) noexcept = 0;
    virtual float getAlpha() const noexcept = 0;
    virtual float getBeta() const noexcept = 0;
};

class VUnaryLayer : public VRoot {
   public:
    virtual void setOperation(UnaryOperation op) noexcept = 0;
    virtual UnaryOperation getOperation() const noexcept = 0;
};
class VParametricReLULayer : public VRoot {
   public:
};

class VSelectLayer : public VRoot {};

class VSoftMaxLayer : public VRoot {
   public:
    virtual void setAxes(uint32_t axes) noexcept = 0;
    virtual uint32_t getAxes() const noexcept = 0;
};

class VReduceLayer : public VRoot {
   public:
    virtual void setOperation(ReduceOperation op) noexcept = 0;
    virtual ReduceOperation getOperation() const noexcept = 0;
    virtual void setReduceAxes(uint32_t reduceAxes) noexcept = 0;
    virtual uint32_t getReduceAxes() const noexcept = 0;
    virtual void setKeepDimensions(bool keepDimensions) noexcept = 0;
    virtual bool getKeepDimensions() const noexcept = 0;
};

class VEinsumLayer : public VRoot {
   public:
    virtual bool setEquation(char const* equation) noexcept = 0;
    virtual char const* getEquation() const noexcept = 0;
};

class VElementWiseLayer : public VRoot {
   public:
    virtual void setOperation(ElementWiseOperation op) noexcept = 0;
    virtual ElementWiseOperation getOperation() const noexcept = 0;
};

class VGatherLayer : public VRoot {
   public:
    virtual void setGatherAxis(int32_t axis) noexcept = 0;
    virtual int32_t getGatherAxis() const noexcept = 0;
    virtual void setNbElementWiseDims(int32_t k) noexcept = 0;
    virtual int32_t getNbElementWiseDims() const noexcept = 0;
    virtual void setMode(GatherMode mode) noexcept = 0;
    virtual GatherMode getMode() const noexcept = 0;
};

class VPluginV2Layer : public VRoot {
   public:
    virtual IPluginV2& getPlugin() noexcept = 0;
};

class VPoolingLayer : public VRoot {
   public:
    virtual void setPoolingType(PoolingType type) noexcept = 0;
    virtual PoolingType getPoolingType() const noexcept = 0;
    virtual void setBlendFactor(float blendFactor) noexcept = 0;
    virtual float getBlendFactor() const noexcept = 0;
    virtual void setAverageCountExcludesPadding(bool exclusive) noexcept = 0;
    virtual bool getAverageCountExcludesPadding() const noexcept = 0;
    virtual void setPrePadding(Dims padding) noexcept = 0;
    virtual Dims getPrePadding() const noexcept = 0;
    virtual void setPostPadding(Dims padding) noexcept = 0;
    virtual Dims getPostPadding() const noexcept = 0;
    virtual void setPaddingMode(PaddingMode paddingMode) noexcept = 0;
    virtual PaddingMode getPaddingMode() const noexcept = 0;
    virtual void setWindowSizeNd(Dims windowSize) noexcept = 0;
    virtual Dims getWindowSizeNd() const noexcept = 0;
    virtual void setStrideNd(Dims stride) noexcept = 0;
    virtual Dims getStrideNd() const noexcept = 0;
    virtual void setPaddingNd(Dims padding) noexcept = 0;
    virtual Dims getPaddingNd() const noexcept = 0;
};
class VSliceLayer : public VRoot {
   public:
    virtual void setStart(Dims start) noexcept = 0;
    virtual Dims getStart() const noexcept = 0;
    virtual void setSize(Dims size) noexcept = 0;
    virtual Dims getSize() const noexcept = 0;
    virtual void setStride(Dims stride) noexcept = 0;
    virtual Dims getStride() const noexcept = 0;
    virtual void setMode(SliceMode mode) noexcept = 0;
    virtual SliceMode getMode() const noexcept = 0;
};

class VResizeLayer : public VRoot {
   public:
    virtual void setOutputDimensions(Dims dimensions) noexcept = 0;
    virtual Dims getOutputDimensions() const noexcept = 0;
    virtual void setScales(float const* scales, int32_t nbScales) noexcept = 0;
    virtual int32_t getScales(int32_t size, float* scales) const noexcept = 0;
    virtual void setResizeMode(ResizeMode resizeMode) noexcept = 0;
    virtual ResizeMode getResizeMode() const noexcept = 0;
    virtual void setCoordinateTransformation(ResizeCoordinateTransformation coordTransform) noexcept = 0;
    virtual ResizeCoordinateTransformation getCoordinateTransformation() const noexcept = 0;
    virtual void setSelectorForSinglePixel(ResizeSelector selector) noexcept = 0;
    virtual ResizeSelector getSelectorForSinglePixel() const noexcept = 0;
    virtual void setNearestRounding(ResizeRoundMode value) noexcept = 0;
    virtual ResizeRoundMode getNearestRounding() const noexcept = 0;
};

class VGridSampleLayer : public VRoot {
   public:
    virtual void setInterpolationMode(InterpolationMode mode) noexcept = 0;
    virtual InterpolationMode getInterpolationMode() const noexcept = 0;
    virtual void setAlignCorners(bool alignCorners) noexcept = 0;
    virtual bool getAlignCorners() const noexcept = 0;
    virtual bool setSampleMode(SampleMode mode) noexcept = 0;
    virtual SampleMode getSampleMode() const noexcept = 0;
};

class VConcatenationLayer : public VRoot {
   public:
    virtual void setAxis(int32_t axis) noexcept = 0;
    virtual int32_t getAxis() const noexcept = 0;
};
class VQuantizeLayer : public VRoot {
   public:
    virtual int32_t getAxis() const noexcept = 0;
    virtual void setAxis(int32_t axis) noexcept = 0;
};

class VDequantizeLayer : public VRoot {
   public:
    virtual int32_t getAxis() const noexcept = 0;
    virtual void setAxis(int32_t axis) noexcept = 0;
};

class VFillLayer : public VRoot {
   public:
    virtual void setDimensions(Dims dimensions) noexcept = 0;
    virtual Dims getDimensions() const noexcept = 0;
    virtual void setOperation(FillOperation op) noexcept = 0;
    virtual FillOperation getOperation() const noexcept = 0;
    virtual void setAlpha(double alpha) noexcept = 0;
    virtual double getAlpha() const noexcept = 0;
    virtual void setBeta(double beta) noexcept = 0;
    virtual double getBeta() const noexcept = 0;
};

class VIdentityLayer : public VRoot {
   public:
};

class VMatrixMultiplyLayer : public VRoot {
   public:
    virtual void setOperation(int32_t index, MatrixOperation op) noexcept = 0;
    virtual MatrixOperation getOperation(int32_t index) const noexcept = 0;
};

class VConstantLayer : public VRoot {
   public:
    virtual void setWeights(Weights weights) noexcept = 0;
    virtual Weights getWeights() const noexcept = 0;
    virtual void setDimensions(Dims dimensions) noexcept = 0;
    virtual Dims getDimensions() const noexcept = 0;
};
class VNormalizationLayer : public VRoot {
   public:
    virtual void setEpsilon(float eps) noexcept = 0;
    virtual float getEpsilon() const noexcept = 0;
    virtual void setAxes(uint32_t axesMask) noexcept = 0;
    virtual uint32_t getAxes() const noexcept = 0;
    virtual void setNbGroups(int32_t nbGroups) noexcept = 0;
    virtual int32_t getNbGroups() const noexcept = 0;
    virtual void setComputePrecision(DataType type) noexcept = 0;
    virtual DataType getComputePrecision() const noexcept = 0;
};

class VShuffleLayer : public VRoot {
   public:
    virtual void setFirstTranspose(Permutation const& permutation) noexcept = 0;
    virtual Permutation getFirstTranspose() const noexcept = 0;
    virtual void setReshapeDimensions(Dims dimensions) noexcept = 0;
    virtual Dims getReshapeDimensions() const noexcept = 0;
    virtual void setSecondTranspose(Permutation const& permutation) noexcept = 0;
    virtual Permutation getSecondTranspose() const noexcept = 0;
    virtual void setZeroIsPlaceholder(bool zeroIsPlaceholder) = 0;
    virtual bool getZeroIsPlaceholder() const = 0;
};

class VCastLayer : public VRoot {
   public:
    virtual void setToType(DataType toType) noexcept = 0;
    virtual DataType getToType() const noexcept = 0;
};

class VTopKLayer : public VRoot {
   public:
    virtual void setOperation(TopKOperation op) noexcept = 0;
    virtual TopKOperation getOperation() const noexcept = 0;
    virtual void setK(int32_t k) noexcept = 0;
    virtual int32_t getK() const noexcept = 0;
    virtual void setReduceAxes(uint32_t reduceAxes) noexcept = 0;
    virtual uint32_t getReduceAxes() const noexcept = 0;
};

class VShapeLayer : public VRoot {
   public:
};

class VFullyConnectedLayer : public VRoot {
   public:
    virtual void setNbOutputChannels(int32_t nbOutputs) noexcept = 0;
    virtual int32_t getNbOutputChannels() const noexcept = 0;
    //  virtual void setKernelWeights(Weights weights) noexcept = 0;
    //  virtual Weights getKernelWeights() const noexcept = 0;
    //  virtual void setBiasWeights(Weights weights) noexcept = 0;
    //  virtual Weights getBiasWeights() const noexcept = 0;
};

class VAssertionLayer : public VRoot {
   public:
    virtual void setMessage(char const* message) noexcept = 0;
    virtual char const* getMessage() const noexcept = 0;
};

}  // namespace apiv
}  // namespace nvinfer1
