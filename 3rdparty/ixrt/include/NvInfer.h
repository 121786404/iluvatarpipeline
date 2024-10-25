#pragma once
#include "NvInferRuntime.h"
namespace nvinfer1 {

//!
//! \enum LayerType
//!
//! \brief The type values of layer classes.
//!
//! \see ILayer::getType()
//!
enum class LayerType : int32_t {
    kUNKNOWN = 0,
    kCONVOLUTION = 1,
    kELEMENTWISE = 2,
    kACTIVATION = 3,
    kRESIZE = 4,
    kRESHAPE = 5,
    kPOOLING = 6,
    kFULLY_CONNECTED = 7,
    kCONCATENATION = 8,
    kSOFTMAX = 9,
    kBATCH_NORMALIZATION = 10,
    kREDUCE = 11,
    kNMS = 16,
    kNONMAXSUPPRESSION = 17,
    kFOCUS = 19,
    kCONFORMER_ENCODER_CTC = 28,
    kCONFORMER_ENCODER_CTC_Fp16 = 29,
    kCONV2DTRANSPOSE = 30,
    kTRANSPOSE = 32,
    kSLICE = 33,
    kPADDING = 34,
    kCAST = 47,
    kLOGSOFTMAX = 51,
    kEMBED_LAYERNORMALIZATION = 53,
    kATTENTION = 54,
    kMATMUL = 55,
    kSKIP_LAYERNORMALIZATION = 56,
    kBIASGELU = 57,
    kGATHER = 58,
    kSPLIT = 60,
    kSQUEEZE = 76,
    kLSTM = 77,
    kUNARY = 78,
    kUNSQUEEZE = 82,
    kINSTANCE_NORMALIZATION = 83,
    kTILE = 84,
    kALIGN_CHANNEL = 86,
    kUNALIGN_CHANNEL = 87,
    kMINMAX = 88,
    kRELATION = 89,
    kLOGICAL = 90,
    kSHAPE = 91,
    kEXPAND = 95,
    kERF = 96,
    kSCATTER = 97,
    kSCATTER_ND = 98,
    kIDENTITY = 99,
    kEINSUM = 100,
    kGROUPNORM = 101,
    kNON_ZERO = 107,
    kPARAMETRIC_RELU = 108,
    kTOPK = 109,
    kRANGE = 110,
    kCONSTANT_OF_SHAPE = 111,
    kQUANTIZE = 114,
    kDEQUANTIZE = 115,
    kPLUGIN_V2 = 116,
    kFILL = 118,
    kMATRIX_MULTIPLY = 119,
    kCONSTANT = 121,
    kNORMALIZATION = 122,
    kSELECT = 123,
    kSHUFFLE = 124,
    kASSERTION = 125,
    kLAYERNORM = 126,
    kTRILU = 127,
    kMEMORY_TRANSFER = 128,
    kCUMSUM = 129,
    kCONDITION = 130,           //!< Condition layer
    kCONDITIONAL_INPUT = 131,   //!< Conditional Input layer
    kCONDITIONAL_OUTPUT = 132,  //!< Conditional Output layer
    kMOD = 133,
    kONNX_SLICE = 134,
    kCLIP = 135,
    kONE_HOT = 136,
    kSCALES_TO_SIZES = 137,
    kGRID_SAMPLE = 138,
};

//!
//! \enum ActivationType
//!
//! \brief Enumerates the types of activation to specify at an activation layer.
//!
enum class ActivationType : int32_t {
    kUNKNOWN = -1,
    kRELU = 0,               //!< Rectified linear activation: x>=0 ? x : 0
    kSIGMOID = 1,            //!< Sigmoid activation: 1 / (1 + exp(-x))
    kTANH = 2,               //!< TanH activation: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    kLEAKY_RELU = 3,         //!< LeakyRelu activation: x>=0 ? x : alpha * x.
    kELU = 4,                //!< Elu activation: x>=0 ? x : alpha * (exp(x) - 1).
    kSELU = 5,               //!< Selu activation: x>0 ? beta * x : beta * (alpha*exp(x) - alpha)
    kSOFTSIGN = 6,           //!< Softsign activation: x / (1+|x|)
    kSOFTPLUS = 7,           //!< Parametric softplus activation: alpha*log(exp(beta*x)+1)
    kCLIP = 8,               //!< Clip activation: max(alpha, min(beta, x))
    kHARD_SIGMOID = 9,       //!< Hard sigmoid activation: max(0, min(1, alpha*x+beta))
    kSCALED_TANH = 10,       //!< Scaled tanh activation: alpha*tanh(beta*x)
    kTHRESHOLDED_RELU = 11,  //!< Thresholded ReLU activation: x>alpha ? x : 0
    kSILU = 12,              //!< SiLU activation: x / (1 + exp(-x)) or x * sigmoid(x)
    kMISH = 13,              //!< Mish activation: x * tanh(ln(1 + exp(x))
    kHARD_SWISH = 14,        //!< Hard swish activation: x < -3 ? 0 : x < 3 ? (x + 3) * x / 6 : x
    kGELU = 15,              //!< Gelu activation: x * 0.5 * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
};

enum class ElementWiseOperation : int32_t {
    kUNKNOWN = 0,
    kSUM = 1,
    kSUB = 2,
    kPROD = 3,
    kDIV = 4,
    kPOW = 5,
    kEQUAL = 6,
    kMAX = 7,
    kMIN = 8,
    kFLOOR_DIV = 9,
};

enum class PoolingType : int32_t { kUNKNOWN = 0, kAVERAGE = 1, kGLOBAL_AVERAGE = 2, kMAX = 3 };
enum class ReshapeType : int32_t { kUNKNOWN = 0, kNORMAL = 1, kFLATTEN = 2 };
enum class MinMaxType : int32_t {
    kUNKNOWN = 0,
    kMIN = 1,
    kMAX = 2,
};

enum class RelationType : int32_t {
    kUNKNOWN = 0,
    kEQUAL = 1,
    kGREATER = 2,
    kGREATER_OR_EQUAL = 3,
    kLESS = 4,
    kLESS_OR_EQUAL = 5
};

enum class LogicalType : int32_t {
    kUNKNOWN = 0,
    kAND = 1,
    kOR = 2,
    kXOR = 3,
};
enum class PaddingMode : int32_t {
    kEXPLICIT_ROUND_DOWN = 0,
    kEXPLICIT_ROUND_UP = 1,
    kSAME_UPPER = 2,
    kSAME_LOWER = 3,
    kCAFFE_ROUND_DOWN = 4,
    kCAFFE_ROUND_UP = 5
};

enum class SliceMode : int32_t {
    kDEFAULT = 0,
    kWRAP = 1,
    kCLAMP = 2,
    kFILL = 3,
    kREFLECT = 4,
};
enum class ScaleModeTrt : int32_t { kUNIFORM = 0, kCHANNEL = 1, kELEMENTWISE = 2 };

enum class GatherModeTrt : int32_t { kDEFAULT = 0, kELEMENT = 1, kND = 2 };

enum class SliceModeTrt : int32_t {
    kSTRICT_BOUNDS = 0,
    kWRAP = 1,
    kCLAMP = 2,
    kFILL = 3,
    kREFLECT = 4,

};

enum class SampleMode : int32_t {
    kSTRICT_BOUNDS = 0,  //!< Fail with error when the coordinates are out of bounds.
    kWRAP = 1,           //!< Coordinates wrap around periodically.
    kCLAMP = 2,          //!< Out of bounds indices are clamped to bounds.
    kFILL = 3,           //!< Use fill input value when coordinates are out of bounds.
    kREFLECT = 4,        //!< Coordinates reflect. The axis of reflection is the middle of the perimeter pixel and the
                   //!< reflections are repeated indefinitely within the padded regions. Repeats values for a single
                   //!< pixel and throws error for zero pixels.
};

enum class InterpolationMode : int32_t {
    kNEAREST = 0,  //!< ND (0 < N <= 8) nearest neighbor resizing.
    kLINEAR = 1,   //!< Supports bilinear (2D) interpolation
    kCUBIC = 2     //!< Supports bicubic (2D) interpolation
};

//!
//! \enum TopKOperation
//!
//! \brief Enumerates the operations that can be performed by a TopK layer.
//!
enum class TopKOperation : int32_t {
    //!< Max of the elements.
    kMAX = 0,
    //!< Min of the elements.
    kMIN = 1,
};

enum class ScatterModeTrt : int32_t {
    kELEMENT = 0,
    kND = 1,
};

//!
//! \class ITensor
//! \brief A tensor in a network definition.
//!
class ITensor : public INoCopy {
   public:
    char const* getName() const noexcept { return mImpl->getName(); }

    //!
    //! \brief Set the tensor name.
    //!
    //! \param name The name.
    //!
    void setName(char const* name) noexcept { mImpl->setName(name); }

    //!
    //! \brief Set the dimensions of a tensor.
    //!     Only available for input tensors of a network.
    //! \param dimensions Dimensions to be set
    //!
    void setDimensions(Dims dimensions) noexcept { mImpl->setDimensions(dimensions); }

    //!
    //! \brief Get the dimensions of a tensor.
    //!
    //! \return The dimensions.
    //!
    Dims getDimensions() const noexcept { return mImpl->getDimensions(); }

    //!
    //! \brief Set the type of a tensor.
    //!
    void setType(DataType type) noexcept { mImpl->setType(type); }

    //!
    //! \brief Get the type of a tensor.
    //!
    //! \return The dimensions.
    //!
    DataType getType() const noexcept { return mImpl->getType(); }

    //!
    //! \brief Whether the tensor is a network input.
    //!
    bool isNetworkInput() const noexcept { return mImpl->isNetworkInput(); }

    //!
    //! \brief Whether the tensor is a network output.
    //!
    bool isNetworkOutput() const noexcept { return mImpl->isNetworkOutput(); }

    //!
    //! \brief Get the location of a tensor.
    //!
    //! \return The location.
    //!
    TensorLocation getLocation() const noexcept { return mImpl->getLocation(); }

    //!
    //! \brief Set the location of a tensor.
    //!
    void setLocation(TensorLocation location) noexcept { mImpl->setLocation(location); }

    //!
    //! \brief  Whether the tensor is a shape tensor.
    //!
    bool isShapeTensor() const noexcept { return mImpl->isShapeTensor(); }

    //!
    //! \brief Whether the tensor is an execution tensor.
    //!
    bool isExecutionTensor() const noexcept { return mImpl->isExecutionTensor(); }

    //!
    //! \brief Name a dimension of an input tensor.
    //!
    //! \param index index of the dimension
    //! \param name of the dimension, as a pointer to a null-terminated character sequence.
    //!
    //! \see getDimensionName()
    //!
    void setDimensionName(int32_t index, char const* name) noexcept { mImpl->setDimensionName(index, name); }

    //!
    //! \brief Get the name of an input dimension.
    //!
    //! \param index index of the dimension
    //!
    //! \return The name of the input dimension, or nullptr if the dimension has no name.
    //!         The name is a pointer to a null-terminated character sequence.
    //!
    //! \see setDimensionName()
    //!
    char const* getDimensionName(int32_t index) const noexcept { return mImpl->getDimensionName(index); }

    //!
    //! \brief Set dynamic range for the tensor
    //!
    //! Currently, the larger of the absolute values of the provided bounds is used.
    //! \return Whether the dynamic range was set successfully.
    //!
    bool setDynamicRange(float min, float max) noexcept { return mImpl->setDynamicRange(min, max); }

    //!
    //! \brief Query whether dynamic range is set.
    //!
    //! \return True if dynamic range is set, false otherwise.
    //!
    bool dynamicRangeIsSet() const noexcept { return mImpl->dynamicRangeIsSet(); }

    //!
    //! \brief Undo effect of setDynamicRange.
    //!
    void resetDynamicRange() noexcept { mImpl->resetDynamicRange(); }

    //!
    //! \brief Get minimum of dynamic range.
    //!
    //! \return Minimum of dynamic range, or quiet NaN if range was not set.
    //!
    float getDynamicRangeMin() const noexcept { return mImpl->getDynamicRangeMin(); }

    //!
    //! \brief Get maximum of dynamic range.
    //!
    //! \return Maximum of dynamic range, or quiet NaN if range was not set.
    //!
    float getDynamicRangeMax() const noexcept { return mImpl->getDynamicRangeMax(); }

   protected:
    apiv::VTensor* mImpl;
    virtual ~ITensor() noexcept = default;
};

class ILayer : public INoCopy {
   public:
    //!
    //! \brief Get the type of the layer.
    //!
    LayerType getType() const noexcept { return mLayer->getType(); }

    //!
    //! \brief Get the name of the layer.
    //!
    char const* getName() const noexcept { return mLayer->getName(); }

    //!
    //! \brief Set the name of a layer.
    //!
    void setName(char const* name) noexcept { mLayer->setName(name); }

    //!
    //! \brief Get the number of inputs in the layer.
    //!
    //! \return The number of inputs in the layer.
    //!
    int32_t getNbInputs() const noexcept { return mLayer->getNbInputs(); }

    //!
    //! \brief Get the layer output tensor specified by the given index.
    //!
    //! \param index The index of the input tensor.
    //!
    //! \return The input tensor, or nullptr if the index is out of range.
    //!
    ITensor* getInput(int32_t index) const noexcept { return mLayer->getInput(index); }

    //!
    //! \brief Get the number of outputs in the layer.
    //!
    //! \return The number of outputs in the layer.
    //!
    int32_t getNbOutputs() const noexcept { return mLayer->getNbOutputs(); }

    //!
    //! \brief Get the layer output tensor specified by the given index.
    //!
    //! \param index The index of the output tensor.
    //!
    //! \return The output tensor, or nullptr if the index is out of range.
    //!
    ITensor* getOutput(int32_t index) const noexcept { return mLayer->getOutput(index); }

    //!
    //! \brief Set the precision.
    //!
    //! \return the layer precision.
    //!
    void setInput(int32_t index, ITensor& tensor) noexcept { return mLayer->setInput(index, tensor); }

    //!
    //! \brief set the precision.
    //!
    //! \param dataType the precision.
    //!
    void setPrecision(DataType dataType) noexcept { mLayer->setPrecision(dataType); }

    //!
    //! \brief get the precision.
    //!
    //! \return the layer precision.
    //!
    DataType getPrecision() const noexcept { return mLayer->getPrecision(); }

    //!
    //! \brief get the output type
    //!
    //! \param index the index of the output
    //!
    //! \return the output type.
    //!
    DataType getOutputType(int32_t index) const noexcept { return mLayer->getOutputType(index); }

    void setOutputType(int32_t index, DataType dataType) noexcept { mLayer->setOutputType(index, dataType); }

   protected:
    virtual ~ILayer() noexcept = default;
    apiv::VLayer* mLayer;
};

//!
//! \class IConvolutionLayer
//! \brief A convolution layer in a network definition.
//!
class IConvolutionLayer : public ILayer {
   public:
    //!
    //! \brief Set the number of output maps.
    //!
    void setNbOutputMaps(int32_t nbOutputMaps) noexcept { mImpl->setNbOutputMaps(nbOutputMaps); }

    //!
    //! \brief Get the number of output maps.
    //!
    int32_t getNbOutputMaps() const noexcept { return mImpl->getNbOutputMaps(); }

    //!
    //! \brief Set the number of groups.
    //!
    //! Default: 1
    //!
    void setNbGroups(int32_t nbGroups) noexcept { mImpl->setNbGroups(nbGroups); }

    //!
    //! \brief Get the number of groups.
    //!
    int32_t getNbGroups() const noexcept { return mImpl->getNbGroups(); }

    //!
    //! \brief Set the kernel weights.
    //!
    // void setKernelWeights(Weights weights) noexcept { mImpl->setKernelWeights(weights); }

    //!
    //! \brief Get the kernel weights.
    //!
    // Weights getKernelWeights() const noexcept { return mImpl->getKernelWeights(); }

    //!
    //! \brief Set the bias weights.
    //!
    // void setBiasWeights(Weights weights) noexcept { mImpl->setBiasWeights(weights); }

    //!
    //! \brief Get the bias weights.
    //!
    // Weights getBiasWeights() const noexcept{ return mImpl->getBiasWeights(); }

    //!
    //! \brief Set the pre-padding.
    //!
    void setPrePadding(Dims padding) noexcept { mImpl->setPrePadding(padding); }

    //!
    //! \brief Get the pre-padding.
    //!
    Dims getPrePadding() const noexcept { return mImpl->getPrePadding(); }

    //!
    //! \brief Set the post-padding.
    //!
    void setPostPadding(Dims padding) noexcept { mImpl->setPostPadding(padding); }

    //!
    //! \brief Get the post-padding.
    //!
    Dims getPostPadding() const noexcept { return mImpl->getPostPadding(); }

    //!
    //! \brief Set the padding mode.
    //!
    //! Default: kEXPLICIT_ROUND_DOWN
    //!
    void setPaddingMode(PaddingMode paddingMode) noexcept { mImpl->setPaddingMode(paddingMode); }

    //!
    //! \brief Get the padding mode.
    //!
    PaddingMode getPaddingMode() const noexcept { return mImpl->getPaddingMode(); }

    //!
    //! \brief Set the kernel size.
    //!
    void setKernelSizeNd(Dims kernelSize) noexcept { mImpl->setKernelSizeNd(kernelSize); }

    //!
    //! \brief Get the kernel size.
    //!
    Dims getKernelSizeNd() const noexcept { return mImpl->getKernelSizeNd(); }

    //!
    //! \brief Set the stride.
    //!
    //! Default: (1, 1, ..., 1)
    //!
    void setStrideNd(Dims stride) noexcept { mImpl->setStrideNd(stride); }

    //!
    //! \brief Get the stride.
    //!
    //! Default: (1, 1, ..., 1)
    //!
    Dims getStrideNd() const noexcept { return mImpl->getStrideNd(); }

    //!
    //! \brief Set the padding.
    //!
    //! Padding is symmetric. Default: (0, 0, ..., 0)
    //!
    void setPaddingNd(Dims padding) noexcept { mImpl->setPaddingNd(padding); }

    //!
    //! \brief Get the padding.
    //!
    //! If the padding is asymmetric, the pre-padding is returned.
    //!
    Dims getPaddingNd() const noexcept { return mImpl->getPaddingNd(); }

    //!
    //! \brief Set the dilation.
    //!
    //! Default: (1, 1, ..., 1)
    //!
    void setDilationNd(Dims dilation) noexcept { mImpl->setDilationNd(dilation); }

    //!
    //! \brief Get the dilation.
    //!
    Dims getDilationNd() const noexcept { return mImpl->getDilationNd(); }

    //!
    //! \brief Append or replace an input of this layer with a specific tensor
    //!
    //! \param index the index of the input to modify.
    //! \param tensor the new input tensor
    //!
    //! Only index 0 (data input) is valid, unless explicit-quantization mode is enabled.
    //! In explicit-quantization mode, input with index 1 is the kernel-weights tensor, if present.
    //! The kernel-weights tensor must be a build-time constant (computable at build-time via constant-folding)
    //! and an output of a dequantize layer.
    //! If input index 1 is used then the kernel-weights parameter must be set to empty Weights.
    //!
    //! \see getKernelWeights(), setKernelWeights()
    //!
    //! The indices are as follows:
    //!
    //! - 0: The input activation tensor.
    //! - 1: The kernel weights tensor (a constant tensor).
    //!
    //! If this function is called with the value 1, then the function getNbInputs() changes
    //! from returning 1 to 2.
    using ILayer::setInput;

   protected:
    virtual ~IConvolutionLayer() noexcept = default;
    apiv::VConvolutionLayer* mImpl;
};

//!
//! \class IDeconvolutionLayer
//!
//! \brief A deconvolution layer in a network definition.
//!
class IDeconvolutionLayer : public ILayer {
   public:
    //!
    //! \brief Set the number of output feature maps for the deconvolution.
    //!
    //! \see getNbOutputMaps()
    //!
    void setNbOutputMaps(int32_t nbOutputMaps) noexcept { mImpl->setNbOutputMaps(nbOutputMaps); }

    //!
    //! \brief Get the number of output feature maps for the deconvolution.
    //!
    //! \see setNbOutputMaps()
    //!
    int32_t getNbOutputMaps() const noexcept { return mImpl->getNbOutputMaps(); }

    //!
    //! \brief Set the number of groups for a deconvolution.
    //!
    //! The input tensor channels are divided into \p nbGroups groups, and a deconvolution is executed for each group,
    //! using a filter per group. The results of the group convolutions are concatenated to form the output.
    //!
    //! Default: 1
    //!
    //! \see getNbGroups()
    //!
    void setNbGroups(int32_t nbGroups) noexcept { mImpl->setNbGroups(nbGroups); }

    //!
    //! \brief Get the number of groups for a deconvolution.
    //!
    //! \see setNbGroups()
    //!
    int32_t getNbGroups() const noexcept { return mImpl->getNbGroups(); }

    //!
    //! \brief Set the kernel weights for the deconvolution.
    //!
    //! The weights are specified as a contiguous array in \p CKRS order, where \p C the number of
    //! input channels, \p K the number of output feature maps, and \p R and \p S are the height and width
    //! of the filter.
    //!
    //! \see getWeights()
    //!
    // void setKernelWeights(Weights weights) noexcept
    // {
    //     mImpl->setKernelWeights(weights);
    // }

    //!
    //! \brief Get the kernel weights for the deconvolution.
    //!
    //! \see setNbGroups()
    //!
    // Weights getKernelWeights() const noexcept
    // {
    //     return mImpl->getKernelWeights();
    // }

    //!
    //! \brief Set the bias weights for the deconvolution.
    //!
    //! Bias is optional. To omit bias, set the count value of the weights structure to zero.
    //!
    //! The bias is applied per-feature-map, so the number of weights (if non-zero) must be equal to the number of
    //! output feature maps.
    //!
    //! \see getBiasWeights()
    //!
    // void setBiasWeights(Weights weights) noexcept
    // {
    //     mImpl->setBiasWeights(weights);
    // }

    //!
    //! \brief Get the bias weights for the deconvolution.
    //!
    //! \see getBiasWeights()
    //!
    // Weights getBiasWeights() const noexcept
    // {
    //     return mImpl->getBiasWeights();
    // }

    //!
    //! \brief Set the multi-dimension pre-padding.
    //!
    //! Default: (0, 0, ..., 0)
    //!
    //! \see getPrePadding()
    //!
    void setPrePadding(Dims padding) noexcept { mImpl->setPrePadding(padding); }

    //!
    //! \brief Get the pre-padding.
    //!
    //! \see setPrePadding()
    //!
    Dims getPrePadding() const noexcept { return mImpl->getPrePadding(); }

    //!
    //! \brief Set the multi-dimension post-padding.
    //!
    //! The output will be trimmed by this number of elements on the end of every dimension.
    //! In other words, it resembles the inverse of a convolution layer with this padding size.
    //! Negative padding is not supported.
    //!
    //! Default: (0, 0, ..., 0)
    //!
    //! \see getPostPadding()
    //!
    void setPostPadding(Dims padding) noexcept { mImpl->setPostPadding(padding); }

    //!
    //! \brief Get the padding.
    //!
    //! \see setPostPadding()
    //!
    Dims getPostPadding() const noexcept { return mImpl->getPostPadding(); }

    //!
    //! \brief Set the padding mode.
    //!
    //! Padding mode takes precedence if both setPaddingMode and setPre/PostPadding are used.
    //!
    //! Default: kEXPLICIT_ROUND_DOWN
    //!
    //! \see getPaddingMode()
    //!
    void setPaddingMode(PaddingMode paddingMode) noexcept { mImpl->setPaddingMode(paddingMode); }

    //!
    //! \brief Get the padding mode.
    //!
    //! Default: kEXPLICIT_ROUND_DOWN
    //!
    //! \see setPaddingMode()
    //!
    PaddingMode getPaddingMode() const noexcept { return mImpl->getPaddingMode(); }

    //!
    //! \brief Set the multi-dimension kernel size.
    //!
    //! \see getKernelSizeNd()
    //!
    void setKernelSizeNd(Dims kernelSize) noexcept { mImpl->setKernelSizeNd(kernelSize); }

    //!
    //! \brief Get the multi-dimension kernel size.
    //!
    //! \see setKernelSizeNd()
    //!
    Dims getKernelSizeNd() const noexcept { return mImpl->getKernelSizeNd(); }

    //!
    //! \brief Set the multi-dimension stride.
    //!
    //! \see getStrideNd()
    //!
    void setStrideNd(Dims stride) noexcept { mImpl->setStrideNd(stride); }

    //!
    //! \brief Get the multi-dimension stride.
    //!
    //! \see setStrideNd()
    //!
    Dims getStrideNd() const noexcept { return mImpl->getStrideNd(); }

    //!
    //! \brief Set the multi-dimension padding.
    //!
    //! The output will be trimmed by this number of elements on both sides of every dimension.
    //! In other words, it resembles the inverse of a convolution layer with this padding size.
    //! Padding is symmetric, and negative padding is not supported.
    //!
    //! Default: (0, 0, ..., 0)
    //!
    //! \see getPaddingNd()
    //!
    void setPaddingNd(Dims padding) noexcept { mImpl->setPaddingNd(padding); }

    //!
    //! \brief Get the multi-dimension padding.
    //!
    //! If the padding is asymmetric, the pre-padding is returned.
    //!
    //! \see setPaddingNd()
    //!
    Dims getPaddingNd() const noexcept { return mImpl->getPaddingNd(); }

    //! \brief Set the multi-dimension dilation.
    //!
    //! Default: (1, 1, ..., 1)
    //!
    //! \see getDilationNd()
    //!
    void setDilationNd(Dims dilation) noexcept { mImpl->setDilationNd(dilation); }

    //!
    //! \brief Get the multi-dimension dilation.
    //!
    //! \see setDilationNd()
    //!
    Dims getDilationNd() const noexcept { return mImpl->getDilationNd(); }

    //!
    //! \brief Append or replace an input of this layer with a specific tensor
    //!
    //! \param index the index of the input to modify.
    //! \param tensor the new input tensor
    //!
    //! Only index 0 (data input) is valid, unless explicit-quantization mode is enabled.
    //! In explicit-quantization mode, input with index 1 is the kernel-weights tensor, if present.
    //! The kernel-weights tensor must be a build-time constant (computable at build-time via constant-folding)
    //! and an output of a dequantize layer.
    //! If input index 1 is used then the kernel-weights parameter must be set to empty Weights.
    //!
    //! \see getKernelWeights(), setKernelWeights()
    //!
    //! The indices are as follows:
    //!
    //! - 0: The input activation tensor.
    //! - 1: The kernel weights tensor (a constant tensor).
    //!
    //! If this function is called with the value 1, then the function getNbInputs() changes
    //! from returning 1 to 2.
    using ILayer::setInput;

   protected:
    virtual ~IDeconvolutionLayer() noexcept = default;
    apiv::VDeconvolutionLayer* mImpl;
};

//!
//! \class IActivationLayer
//!
//! \brief An Activation layer in a network definition.
//!
//! This layer applies activation function to every input element.
//!
//! The output keep the same shape as the input.
//!
class IActivationLayer : public ILayer {
   public:
    //!
    //! \brief Set activation function type
    //!
    void setActivationType(ActivationType type) noexcept { mImpl->setActivationType(type); }

    //!
    //! \brief Get activation function type
    //!
    ActivationType getActivationType() const noexcept { return mImpl->getActivationType(); }

    //!
    //! \brief Set the alpha parameter (must be finite value).
    //!
    //! Alpha value will be used by the following activations:
    //! LeakyRelu, Elu, Selu, Softplus, Clip, HardSigmoid, ScaledTanh,
    //! ThresholdedRelu. Ignored by the other activations.
    //!
    void setAlpha(float alpha) noexcept { mImpl->setAlpha(alpha); }

    //!
    //! \brief Set the beta parameter (must be finite value).
    //!
    //! Beta will be used by the following activations:
    //! Selu, Softplus, Clip, HardSigmoid, ScaledTanh. Ignored by the other activations.
    //!
    void setBeta(float beta) noexcept { mImpl->setBeta(beta); }

    //!
    //! \brief Get the alpha parameter.
    //!
    float getAlpha() const noexcept { return mImpl->getAlpha(); }

    //!
    //! \brief Get the beta parameter.
    //!
    float getBeta() const noexcept { return mImpl->getBeta(); }

   protected:
    virtual ~IActivationLayer() noexcept = default;
    apiv::VActivationLayer* mImpl;
};

//!
//! \class IParametricReLULayer
//!
//! \brief Parametric ReLU operation.
//!
class IParametricReLULayer : public ILayer {
   protected:
    apiv::VParametricReLULayer* mImpl;
    virtual ~IParametricReLULayer() noexcept = default;
};

//!
//! \enum UnaryOperation
//!
//! \brief Enumerates the unary operations that be performed by Unary layer.
//!
enum class UnaryOperation : int32_t {
    kUNKNOWN = 0,
    kEXP = 1,      //!< Exponentiation.
    kLOG = 2,      //!< Log (base e).
    kSQRT = 3,     //!< Square root.
    kSIN = 4,      //!< Sine.
    kCOS = 5,      //!< Cosine.
    kASIN = 6,     //!< Inverse sine.
    kACOS = 7,     //!< Inverse cosine.
    kATANH = 8,    //!< Inverse hyperbolic tangent.
    kABS = 9,      //!< Absolute value.
    kSQUARE = 10,  //!< Square.
    kFLOOR = 11,   //!< Floor.
    kCEIL = 12,    //!< Ceiling.
    kROUND = 13,   //!< Round to nearest even for float datatype.
    kRECIP = 14,   //!< Reciprocal.
    kNEG = 15,     //!< Negation.
    kTAN = 16,     //!< Tangent.
    kSINH = 17,    //!< Hyperbolic sine.
    kCOSH = 18,    //!< Hyperbolic cosine.
    kATAN = 19,    //!< Inverse tangent.
    kASINH = 20,   //!< Inverse hyperbolic sine.
    kACOSH = 21,   //!< Inverse hyperbolic cosine.
    kERF = 22,     //!< Gauss error function.
    kNOT = 23,     //!< Logical NOT.
    kSIGN = 24,    //!< Sign, If input > 0, output 1; if input < 0, output -1; if input == 0, output 0.
};

//!
//! \class IUnaryLayer
//!
//! \brief Layer that represents an unary operation.
//!
class IUnaryLayer : public ILayer {
   public:
    //!
    //! \brief Set the unary operation for the layer.
    //!
    void setOperation(UnaryOperation op) noexcept { mImpl->setOperation(op); }

    //!
    //! \brief Get the unary operation for the layer.
    //!
    UnaryOperation getOperation() const noexcept { return mImpl->getOperation(); }

   protected:
    apiv::VUnaryLayer* mImpl;
    virtual ~IUnaryLayer() noexcept = default;
};

//! \class ISoftMaxLayer
//!
//! \brief A Softmax layer in a userdefine network definition.
//!
class ISoftMaxLayer : public ILayer {
   public:
    //!
    //! \brief Set the axis along which softmax is computed.
    //!
    //! \param axes The axis along which softmax is computed.
    //!        For example, when doing softmax along axis (from 0 to size-1), axes = 1 << axis;
    //!        and axis = log2(axes).
    //!
    void setAxes(uint32_t axes) noexcept { mImpl->setAxes(axes); }

    //!
    //! \brief Get the axis.
    //!
    uint32_t getAxes() const noexcept { return mImpl->getAxes(); }
    using ILayer::setInput;

   protected:
    virtual ~ISoftMaxLayer() noexcept = default;
    apiv::VSoftMaxLayer* mImpl;
};

//! \enum ResizeMode
//!
//! \brief Enumerates various modes of resize in the resize layer.
//!
enum class ResizeMode : int32_t { kNEAREST = 0, kLINEAR = 1 };

//!
//! \enum ResizeCoordinateTransformation
//!
//! \brief The resize coordinate transformation function.
//!
enum class ResizeCoordinateTransformation : int32_t {
    kALIGN_CORNERS = 0,
    kASYMMETRIC = 1,
    kHALF_PIXEL = 2,
};

//!
//! \enum ResizeSelector
//!
//! \brief The coordinate selector when resize to single pixel output.
//!
enum class ResizeSelector : int32_t {
    //! Use formula to map the original index.
    kFORMULA = 0,
    //! Select the upper left pixel.
    kUPPER = 1,
};

//!
//! \enum ResizeRoundMode
//!
//! \brief The rounding mode for nearest neighbor resize.
//!
enum class ResizeRoundMode : int32_t {
    //! Round half up.
    kHALF_UP = 0,
    //! Round half down.
    kHALF_DOWN = 1,
    //! Round to floor.
    kFLOOR = 2,
    //! Round to ceil.
    kCEIL = 3,
};

//! \class IResizeLayer
//!
//! \brief A resize layer in a network definition.
//!
//! Resize layer can be used for resizing a N-D tensor.
//!
class IResizeLayer : public ILayer {
   public:
    //!
    //! \brief Set the output dimensions.
    //!
    //! \param dimensions The output dimensions. Number of output dimensions must be the same as the number of input
    //! dimensions.
    //!
    void setOutputDimensions(Dims dimensions) noexcept { return mImpl->setOutputDimensions(dimensions); }

    //!
    //! \brief Get the output dimensions.
    //!
    //! \return The output dimensions.
    //!
    Dims getOutputDimensions() const noexcept { return mImpl->getOutputDimensions(); }

    //!
    //! \brief Set the resize scales.
    //!
    //! \param scales An array of resize scales.
    //! \param nbScales Number of scales. Number of scales must be equal to the number of input dimensions.
    //!
    void setScales(float const* scales, int32_t nbScales) noexcept { mImpl->setScales(scales, nbScales); }

    //!
    //! \brief Copies resize scales to scales[0, ..., nbScales-1], where nbScales is the number of scales that were set.
    //!
    //! \param size The number of scales to get..
    //! \param scales Pointer to where to copy the scales.
    //!
    //! \return The number of resize scales
    //!
    int32_t getScales(int32_t size, float* scales) const noexcept { return mImpl->getScales(size, scales); }

    //!
    //! \brief Set resize mode for an input tensor.
    //!
    //! Supported resize modes are Nearest Neighbor and Linear.
    //!
    //! \see ResizeMode
    //!
    void setResizeMode(ResizeMode resizeMode) noexcept { mImpl->setResizeMode(resizeMode); }

    //!
    //! \brief Get resize mode for an input tensor.
    //!
    //! \return The resize mode.
    //!
    ResizeMode getResizeMode() const noexcept { return mImpl->getResizeMode(); }

    //!
    //! \brief Append or replace an input of this layer with a specific tensor
    //!
    //! \param index the index of the input to modify.
    //! \param tensor the new input tensor.
    //!
    //! Sets the input tensor for the given index. The index must be 0 for a static resize layer.
    //!
    using ILayer::setInput;

    //!
    //! \brief Set coordinate transformation function.
    //!
    //! The function maps a coordinate in the output tensor to a coordinate in the input tensor.
    //!
    void setCoordinateTransformation(ResizeCoordinateTransformation coordTransform) noexcept {
        mImpl->setCoordinateTransformation(coordTransform);
    }

    //!
    //! \brief Get coordinate transformation function.
    //!
    ResizeCoordinateTransformation getCoordinateTransformation() const noexcept {
        return mImpl->getCoordinateTransformation();
    }

    //!
    //! \brief Set coordinate selector function when resized to single pixel.
    //!
    //! When resize to single pixel image
    //
    void setSelectorForSinglePixel(ResizeSelector selector) noexcept { mImpl->setSelectorForSinglePixel(selector); }

    //!
    //! \brief Get the coordinate selector function when resized to single pixel.
    //!
    //! \return The selector function.
    //!
    ResizeSelector getSelectorForSinglePixel() const noexcept { return mImpl->getSelectorForSinglePixel(); }

    //!
    //! \brief Set rounding mode for nearest neighbor resize.
    //!
    //! This value is used for nearest neighbor interpolation rounding. It is applied after coordinate transformation.
    //!
    void setNearestRounding(ResizeRoundMode value) noexcept { mImpl->setNearestRounding(value); }

    //!
    //! \brief Get rounding mode for nearest neighbor resize.
    //!
    //! \return The rounding mode.
    //!
    ResizeRoundMode getNearestRounding() const noexcept { return mImpl->getNearestRounding(); }

   protected:
    virtual ~IResizeLayer() noexcept = default;
    apiv::VResizeLayer* mImpl;
};

//! \class IGridSampleLayer
//!
//! \brief A GridSample layer in a network definition.
//!
//! This layer uses an input tensor and a grid tensor to produce an interpolated output tensor.
//! The input and grid tensors must be shape tensors of rank 4. The only supported SampleMode
//! values are SampleMode::kCLAMP, SampleMode::kFILL, and SampleMode::kREFLECT.
//!
class IGridSampleLayer : public ILayer {
   public:
    //!
    //! \brief Set the grid sample interpolation mode.
    //!
    void setInterpolationMode(InterpolationMode mode) noexcept { mImpl->setInterpolationMode(mode); }

    //!
    //! \brief Get the grid sample interpolation mode.
    //!
    InterpolationMode getInterpolationMode() const noexcept { return mImpl->getInterpolationMode(); }

    //!
    //! \brief Set the align corners mode.
    //!
    void setAlignCorners(bool alignCorners) noexcept { mImpl->setAlignCorners(alignCorners); }

    //!
    //! \brief Get the align corners mode.
    //!
    bool getAlignCorners() const noexcept { return mImpl->getAlignCorners(); }

    //!
    //! \brief Set the sample mode.
    //!
    bool setSampleMode(SampleMode mode) noexcept { return mImpl->setSampleMode(mode); }

    //!
    //! \brief Get the sample mode.
    //!
    SampleMode getSampleMode() const noexcept { return mImpl->getSampleMode(); }

   protected:
    apiv::VGridSampleLayer* mImpl;
    virtual ~IGridSampleLayer() noexcept = default;
};

//!
//! \enum ReduceOperation
//!
//! \brief Enumerates the reduce operations be performed by Reduce layer.
//!
enum class ReduceOperation : int32_t {
    kUNKNOWN = 0,
    kL1 = 1,
    kL2 = 2,
    kLOG_SUM = 3,
    kLOG_SUM_EXP = 4,
    kMAX = 5,
    kMEAN = 6,
    kMIN = 7,
    kPROD = 8,
    kSUM = 9,
    kSUM_SQUARE = 10,
    kARGMAX = 11,
    kARGMIN = 12,
    kAVG = 13
};

//!
//! \class IReduceLayer
//!
//! \brief Layer that represents a reduction across a non-bool tensor.
//!
class IReduceLayer : public ILayer {
   public:
    //!
    //! \brief Set the reduce operation for the layer.
    //!
    void setOperation(ReduceOperation op) noexcept { mImpl->setOperation(op); }

    //!
    //! \brief Get the reduce operation for the layer.
    //!
    ReduceOperation getOperation() const noexcept { return mImpl->getOperation(); }

    //!
    //! \brief Set the axes over which to reduce.
    //!
    void setReduceAxes(uint32_t reduceAxes) noexcept { mImpl->setReduceAxes(reduceAxes); }

    //!
    //! \brief Get the axes over which to reduce for the layer.
    //!
    uint32_t getReduceAxes() const noexcept { return mImpl->getReduceAxes(); }

    //!
    //! \brief Set the boolean that specifies whether or not to keep the reduced dimensions for the layer.
    //!
    void setKeepDimensions(bool keepDimensions) noexcept { mImpl->setKeepDimensions(keepDimensions); }

    //!
    //! \brief Get the boolean that specifies whether or not to keep the reduced dimensions for the layer.
    //!
    bool getKeepDimensions() const noexcept { return mImpl->getKeepDimensions(); }

   protected:
    apiv::VReduceLayer* mImpl;
    virtual ~IReduceLayer() noexcept = default;
};

class IElementWiseLayer : public ILayer {
   public:
    //!
    //! \brief Set the binary operation.
    //!
    void setOperation(ElementWiseOperation op) noexcept { return mImpl->setOperation(op); }

    //!
    //! \brief Get the binary operation.
    //!
    ElementWiseOperation getOperation() const noexcept { return mImpl->getOperation(); }

   protected:
    apiv::VElementWiseLayer* mImpl;
    virtual ~IElementWiseLayer() noexcept = default;
};

//! \class IIdentityLayer
//!
//! \brief A layer that represents the identity function.
//!
//! If the output type is explicitly specified via setOutputType, IIdentityLayer can be
//! used to convert data type from one to another. Identity between the same
//! type (kFLOAT -> kFLOAT for example) will mapping data from input tensor.
//! The only valid conversions are:
//!
//!     (kFLOAT | kHALF | kINT32 | kBOOL) -> (kFLOAT | kHALF | kINT32)
//!
class IIdentityLayer : public ILayer {
   protected:
    apiv::VIdentityLayer* mImpl;
    virtual ~IIdentityLayer() noexcept = default;
};

//!
//! \class ISelectLayer
//!
//! \brief A Select layer in a network definition. Work like onnx where
//!
class ISelectLayer : public ILayer {
   protected:
    virtual ~ISelectLayer() noexcept = default;
    apiv::VSelectLayer* mImpl;
};

//!
//! \brief IGatherLayer work mode
//!
//! \see IGatherLayer
//!
enum class GatherMode : int32_t {
    kDEFAULT = 0,  //!< Similar to ONNX Gather
    kELEMENT = 1,  //!< Similar to ONNX GatherElements
    kND = 2        //!< Similar to ONNX GatherND
};

//!
//! Maximum number of item in GatherMode enum.
//!
//! \see GatherMode
//!
template <>
constexpr inline int32_t EnumMax<GatherMode>() noexcept {
    return 3;
}

//!
//! \class IGatherLayer
//!
//! \brief A Gather layer in a network definition. Supports many kinds of gather operation.
//!
//! The Gather layer accept two input tensors, Data and Indices, then produce output tensor .
//! Additionally, there are three control parameters: mode, nbElementwiseDims, and axis
//! Please access gather operation detail from：https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gather
//!
//!
class IGatherLayer : public ILayer {
   public:
    //!
    //! \brief Set the axis used by GatherMode::kELEMENTS and GatherMode::kDEFAULT
    //! The axis must be less than the number of dimensions in the data input.
    //! The axis defaults value is 0.
    //!
    //! \warning Undefined behavior when used with GatherMode::kND.
    //!
    //! \see getGatherAxis()
    //!
    void setGatherAxis(int32_t axis) noexcept { mImpl->setGatherAxis(axis); }

    //!
    //! \brief Get the axis to gather on.
    //! \warning Undefined behavior when used with GatherMode::kND.
    //!
    //! \see setGatherAxis()
    //!
    int32_t getGatherAxis() const noexcept { return mImpl->getGatherAxis(); }

    //! \brief Set the number of leading dimensions of indices tensor to be handled elementwise.
    //! The NbElementWiseDims must be less than the Rank of the data input.
    //! \param elementWiseDims number of dims to be handled as elementwise.
    //!
    //! Default: 0
    //!
    //! The value of nbElementWiseDims and GatherMode are checked during network validation:
    //!
    //! GatherMode::kDEFAULT:It can be 0 or 1
    //! GatherMode::kND: nbElementWiseDims can be between 0 and one less than rank(data).
    //! GatherMode::kELEMENT: nbElementWiseDims must be 0
    //!
    void setNbElementWiseDims(int32_t elementWiseDims) noexcept { mImpl->setNbElementWiseDims(elementWiseDims); }

    //!
    //! \brief Get the number of leading dimensions of indices tensor to be handled elementwise.
    //!
    int32_t getNbElementWiseDims() const noexcept { return mImpl->getNbElementWiseDims(); }

    //!
    //! \brief Set the gather mode.
    //!
    //! \see getMode()
    //!
    void setMode(GatherMode mode) noexcept { mImpl->setMode(mode); }

    //!
    //! \brief Get the gather mode.
    //!
    //! \see setMode()
    //!
    GatherMode getMode() const noexcept { return mImpl->getMode(); }

   protected:
    apiv::VGatherLayer* mImpl;
    virtual ~IGatherLayer() noexcept = default;
};

//!
//! \class IPluginV2Layer
//!
//! \brief Layer type for pluginV2
//!
//! \see IPluginV2
//!
//! \warning Do not inherit from this class.
//!
class IPluginV2Layer : public ILayer {
   public:
    //!
    //! \brief Get the plugin of the layer.
    //!
    //! \see IPluginV2
    //!
    IPluginV2& getPlugin() noexcept { return mImpl->getPlugin(); }

   protected:
    apiv::VPluginV2Layer* mImpl;
    virtual ~IPluginV2Layer() noexcept = default;
};

//! \class IPoolingLayer
//!
//! \brief The layer to reduce the dimensions of the feature maps.
class IPoolingLayer : public ILayer {
   public:
    //!
    //! \brief Set type of pooling.
    //!
    //! \see getPoolingType(), PoolingType
    //!
    void setPoolingType(PoolingType type) noexcept { mImpl->setPoolingType(type); }

    //!
    //! \brief Get the type of pooling.
    //!
    //! \see setPoolingType(), PoolingType
    //!
    PoolingType getPoolingType() const noexcept { return mImpl->getPoolingType(); }

    //!
    //! \brief Not implemented yet
    void setBlendFactor(float blendFactor) noexcept { mImpl->setBlendFactor(blendFactor); }

    //!
    //! \brief Not implemeted yet
    float getBlendFactor() const noexcept { return mImpl->getBlendFactor(); }

    //!
    //! \brief Not implemented yet
    void setAverageCountExcludesPadding(bool exclusive) noexcept { mImpl->setAverageCountExcludesPadding(exclusive); }

    //!
    //! \brief Not implemented yet
    bool getAverageCountExcludesPadding() const noexcept { return mImpl->getAverageCountExcludesPadding(); }

    //!
    //! \brief Not implemented yet
    void setPrePadding(Dims padding) noexcept { mImpl->setPrePadding(padding); }

    //!
    //! \brief Not implemented yet
    Dims getPrePadding() const noexcept { return mImpl->getPrePadding(); }

    //!
    //! \brief Not implemented yet
    void setPostPadding(Dims padding) noexcept { mImpl->setPostPadding(padding); }

    //!
    //! \brief Not implemented yet
    Dims getPostPadding() const noexcept { return mImpl->getPostPadding(); }

    //!
    //! \brief Set the padding mode.
    //!
    //! Padding mode takes precedence if both setPaddingMode and setPre/PostPadding are used.
    //!
    //! Default: kEXPLICIT_ROUND_DOWN
    //!
    //! \see getPaddingMode()
    void setPaddingMode(PaddingMode paddingMode) noexcept { mImpl->setPaddingMode(paddingMode); }

    //!
    //! \brief Get the padding mode.
    //!
    //! Default: kEXPLICIT_ROUND_DOWN
    //!
    //! \see setPaddingMode()
    PaddingMode getPaddingMode() const noexcept { return mImpl->getPaddingMode(); }

    //!
    //! \brief Set the window size for the pooling layer.
    //! \see getWindowSizeNd() setWindowSize() getWindowSize()
    //!
    void setWindowSizeNd(Dims windowSize) noexcept { mImpl->setWindowSizeNd(windowSize); }

    //!
    //! \brief Get the window size of the pooling layer.
    //!
    //! \see setWindowSizeNd()
    //!
    Dims getWindowSizeNd() const noexcept { return mImpl->getWindowSizeNd(); }

    //!
    //! \brief Set stride for the pooling layer.
    //!
    //! Default: (1, 1, ..., 1)
    //! \see getStrideNd() setStride() getStride()
    //!
    void setStrideNd(Dims stride) noexcept { mImpl->setStrideNd(stride); }

    //!
    //! \brief Get stride for the pooling layer.
    //!
    //! \see setStrideNd()
    //!
    Dims getStrideNd() const noexcept { return mImpl->getStrideNd(); }

    //!
    //! \brief Set padding for the pooling layer.
    //! \see getPaddingNd() setPadding() getPadding()
    //!
    void setPaddingNd(Dims padding) noexcept { mImpl->setPaddingNd(padding); }

    //!
    //! \brief Get padding for the pooling layer.
    //!
    //! If the padding is asymmetric, the pre-padding is returned.
    //!
    //! \see setPaddingNd()
    //!
    Dims getPaddingNd() const noexcept { return mImpl->getPaddingNd(); }

   protected:
    virtual ~IPoolingLayer() noexcept = default;
    apiv::VPoolingLayer* mImpl;
};

//!
//! \class IConcatenationLayer
//!
//! \brief A concatenation layer
//!
//! The output dimension axis is the sum of the corresponding input dimensions specified by concatenation axis.
//! Every other output dimension will keep same as the corresponding dimension of the inputs.
//!
//! \warning All input tensors must have the equal dimensions without the concatenation axis.
//!
//!
class IConcatenationLayer : public ILayer {
   public:
    //!
    //! \brief Set the axis along which concatenation occurs.
    //!
    //! \param axis The axis along which concatenation occurs.
    //!
    void setAxis(int32_t axis) noexcept { mImpl->setAxis(axis); }

    //!
    //! \brief Get the concatenation axis
    //!
    int32_t getAxis() const noexcept { return mImpl->getAxis(); }

   protected:
    virtual ~IConcatenationLayer() noexcept = default;
    apiv::VConcatenationLayer* mImpl;
};
//!
//! \class IQuantizeLayer
//!
//! \brief A Quantize layer in a network definition.
class IQuantizeLayer : public ILayer {
   public:
    //!
    //! \brief Get the quantization axis.
    //!
    //! \return axis parameter of the quantize layer
    //! The return value means which axis to quantize, if returns -1, it means per-tensor quantization
    //!
    int32_t getAxis() const noexcept { return mImpl->getAxis(); }
    //!
    //! \brief Set the quantization axis.
    //!
    //! Set the quantization axis of input tensor
    void setAxis(int32_t axis) noexcept { mImpl->setAxis(axis); }

   protected:
    virtual ~IQuantizeLayer() noexcept = default;
    apiv::VQuantizeLayer* mImpl;
};
//!
//! \class IDequantizeLayer
//!
//! \brief A Dequantize layer in a network definition.
//!
class IDequantizeLayer : public ILayer {
   public:
    //!
    //! \brief Get the quantization axis.
    //!
    //! \return axis parameter of the quantize layer
    //! The return value means which axis to quantize, if returns -1, it means per-tensor quantization
    //!
    int32_t getAxis() const noexcept { return mImpl->getAxis(); }
    //!
    //! \brief Set the quantization axis.
    //!
    //! Set the quantization axis of input tensor
    void setAxis(int32_t axis) noexcept { mImpl->setAxis(axis); }

   protected:
    virtual ~IDequantizeLayer() noexcept = default;
    apiv::VDequantizeLayer* mImpl;
};
//!
//! \enum FillOperation
//!
//! \brief Specifys how output tensor is filled
//!
//! \see IFillLayer
//!
enum class FillOperation : int32_t {
    kLINSPACE = 0,        //!< Generate evenly spaced numbers over a specified interval.
    kRANDOM_UNIFORM = 1,  //!< Generate a tensor with random values drawn from a uniform distribution.
    kRANDOM_NORMAL = 2    //!< Generate a tensor with random values drawn from a normal distribution.
};
//!
//! \brief Generate an output tensor with specified mode.
//!
//! \see FillOperation
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
class IFillLayer : public ILayer {
   public:
    //!
    //! \brief Set the dimensions of output tensor.
    //!
    //! \param dimensions The dimensions of output tensor.
    //!
    //! If the first input had been used to create this layer, that input is reset to null by this method.
    //!
    //! \see getDimensions
    //
    void setDimensions(Dims dimensions) noexcept { mImpl->setDimensions(dimensions); }

    //!
    //! \brief Get the output tensor's dimensions.
    //!
    //! \return The output tensor's dimensions, or an invalid Dims structure.
    //!
    //! If the first input is present and non-null,
    //! this function returns a Dims with nbDims = -1.
    //!
    //! \see setDimensions
    //!
    Dims getDimensions() const noexcept { return mImpl->getDimensions(); }

    //!
    //! \brief Set the fill operation for the layer.
    //!
    //! \see getOperation(), FillOperation
    //!
    void setOperation(FillOperation op) noexcept { mImpl->setOperation(op); }

    //!
    //! \brief Get the fill operation for the layer.
    //!
    //! \see setOperation(), FillOperation
    //!
    FillOperation getOperation() const noexcept { return mImpl->getOperation(); }

    //!
    //! \brief Set the alpha parameter.
    //!
    //! \param alpha has different meanings for each operator:
    //!
    //! Operation          | Usage
    //! kLINSPACE          | the start value, defaults to 0.0;
    //! kRANDOM_UNIFORM    | the minimum value, defaults to 0.0;
    //! kRANDOM_NORMAL     | the mean of the normal distribution, default is 0.0;
    //!
    //! If a second input had been used to create this layer, that input is reset to null by this method.
    //!
    //! \see getAlpha
    //
    void setAlpha(double alpha) noexcept { mImpl->setAlpha(alpha); }

    //!
    //! \brief Get the value of alpha parameter.
    //!
    //! \return A double value of alpha.
    //!
    //! If the second input is present and non-null,
    //! this function returns -1.0.
    //!
    //! \see setAlpha
    //!
    double getAlpha() const noexcept { return mImpl->getAlpha(); }

    //!
    //! \brief Set the beta parameter.
    //!
    //! \param beta has different meanings for each operator:
    //!
    //! Operation          | Usage
    //! kLINSPACE          | the delta value, defaults to 1.0;
    //! kRANDOM_UNIFORM    | the maximal value, defaults to 1.0;
    //! kRANDOM_NORMAL     | the standard deviation of the normal distribution, default is 1.0;
    //!
    //! If a third input had been used to create this layer, that input is reset to null by this method.
    //!
    //! \see getBeta
    //!
    void setBeta(double beta) noexcept { mImpl->setBeta(beta); }

    //!
    //! \brief Get the value of beta parameter.
    //!
    //! \return A double value of beta.
    //!
    //! If the third input is present and non-null,
    //! this function returns -1.0.
    //!
    //! \see setBeta
    //!
    double getBeta() const noexcept { return mImpl->getBeta(); }

    //!
    //! \brief replace an input of this layer with a specific tensor.
    //!
    //! \param index the index of the input to set.
    //! \param tensor the new input tensor
    //!
    //! Indices for kLINSPACE are described as:
    //!
    //! - 0: Shape tensor, represents the output tensor's dimensions.
    //! - 1: Start, a scalar, represents the start value.
    //! - 2: Delta, a 1D tensor, length equals to shape tensor's nbDims, represents the delta value for each dimension.
    //!
    //! Indices for kRANDOM_UNIFORM are described as:
    //!
    //! - 0: Shape tensor, represents the output tensor's dimensions.
    //! - 1: Minimum, a scalar, represents the minimum random value.
    //! - 2: Maximum, a scalar, represents the maximal random value.
    //!
    //! Indices for kRANDOM_NORMAL are described as:
    //!
    //! - 0: Shape tensor, represents the output tensor's dimensions.
    //! - 1: Mean, a scalar, represents the mean of the normal distribution,.
    //! - 2: Scale, a scalar, represents the standard deviation of the normal distribution.
    //!
    //! Using the corresponding setter resets the input to null.
    //!
    //! If either inputs 1 or 2, is non-null, then both must be non-null and have the same data type.
    //!
    //! If this function is called for an index greater or equal to getNbInputs(),
    //! then afterwards getNbInputs() returns index + 1, and any missing intervening
    //! inputs are set to null.
    //!
    using ILayer::setInput;

   protected:
    virtual ~IFillLayer() noexcept = default;
    apiv::VFillLayer* mImpl;
};
//! \class ICastLayer
//!
//! \brief Add a cast layer to a network.
//!
//! This layer casts a given tensor's datatype to the \p toType.
//!
class ICastLayer : public ILayer {
   public:
    //!
    //! \brief Set cast layer output type.
    //!
    void setToType(DataType toType) noexcept { mImpl->setToType(toType); }

    //!
    //! \brief Return cast layer output type.
    //!
    DataType getToType() const noexcept { return mImpl->getToType(); }

   protected:
    apiv::VCastLayer* mImpl;
    virtual ~ICastLayer() noexcept = default;
};

//! \enum MatrixOperation
//!
//! \brief Enumerates the operations that can be performed on a tensor
//!
enum class MatrixOperation : int32_t {
    //! Default behavior
    kNONE,

    //! Transpose the tensor.
    kTRANSPOSE,

    //! Treat tensor as a vector, or as a collection of vectors
    kVECTOR
};

//!
//! Maximum number of elements in MatrixOperation enum.
//!
//! \see DataType
//!
template <>
constexpr inline int32_t EnumMax<MatrixOperation>() noexcept {
    return 3;
}

//! \class IEinsumLayer
//!
//! \brief An Einsum layer in a network
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IEinsumLayer : public ILayer {
   public:
    //!
    //! \brief Set the equation.
    //!
    //! \see setEquation()
    //!
    bool setEquation(char const* equation) noexcept { return mImpl->setEquation(equation); }

    //!
    //! \brief Return the equation.
    //!
    //! \see setEquation()
    //!
    char const* getEquation() const noexcept { return mImpl->getEquation(); }

   protected:
    virtual ~IEinsumLayer() noexcept = default;
    apiv::VEinsumLayer* mImpl;
};

//!
//! \class IMatrixMultiplyLayer
//!
//! \brief The layer to Matrix Multiplication.
//!
class IMatrixMultiplyLayer : public ILayer {
   public:
    //!
    //! \brief Set the operation for an input tensor.
    //! \param index Input tensor number (0 or 1).
    //! \param op Operation.
    //! \see getOperation()
    //!
    void setOperation(int32_t index, MatrixOperation op) noexcept { mImpl->setOperation(index, op); }

    //!
    //! \brief Get the operation for an input tensor.
    //!
    //! \param index Input tensor number (0 or 1).
    //!
    //! \see setOperation()
    //!
    MatrixOperation getOperation(int32_t index) const noexcept { return mImpl->getOperation(index); }

   protected:
    apiv::VMatrixMultiplyLayer* mImpl;
    virtual ~IMatrixMultiplyLayer() noexcept = default;
};

//!
//! \brief Slices an input tensor into an output tensor based on the offset and strides.
//!
//! The slice layer has two variants, static and dynamic. Static slice specifies the start, size, and stride
//! dimensions at layer creation time via Dims and can use the get/set accessor functions of the ISliceLayer.
//! Dynamic slice specifies one or more of start, size or stride as ITensors, by using ILayer::setInput to add
//! a second, third, or fourth input respectively. The corresponding Dims are used if an input
//! is missing or null.
//!
//! An application can determine if the ISliceLayer has a dynamic output shape based on whether
//! the size input (third input) is present and non-null.
//!
//! The slice layer selects for each dimension a start location from within the input tensor, and
//! copies elements to the output tensor using the specified stride across the input tensor.
//! Start, size, and stride tensors must be 1D Int32 shape tensors if not specified via Dims.
//!
//! An example of using slice on a tensor:
//! input = {{0, 2, 4}, {1, 3, 5}}
//! start = {1, 0}
//! size = {1, 2}
//! stride = {1, 2}
//! output = {{1, 5}}
//!
//! When the sliceMode is kCLAMP or kREFLECT, for each input dimension, if its size is 0 then the corresponding output
//! dimension must be 0 too.
//!
//! A slice layer can produce a shape tensor if the following conditions are met:
//!
//! * start, size, and stride are build time constants, either as static Dims or as constant input tensors.
//! * The number of elements in the output tensor does not exceed 2 * Dims::MAX_DIMS.
//!
//! The input tensor is a shape tensor if the output is a shape tensor.
class ISliceLayer : public ILayer {
   public:
    //!
    //! \brief Set the start offset that the slice layer uses to create the output slice.
    //!
    //! \param start The start offset to read data from the input tensor.
    //!
    //! If a second input had been used to create this layer, that input is reset to null by this method.
    //!
    //! \see getStart
    //!
    void setStart(Dims start) noexcept { mImpl->setStart(start); }

    //!
    //! \brief Set the start offset that the slice layer uses to create the output slice.
    //!
    //! \param start The start offset to read data from the input tensor.
    //!
    //! If a second input had been used to create this layer, that input is reset to null by this method.
    //!
    //! \see getStart
    //!
    Dims getStart() const noexcept { return mImpl->getStart(); }

    //!
    //! \brief Set the dimensions of the output slice.
    //!
    //! \param size The dimensions of the output slice.
    //!
    //! If a third input had been used to create this layer, that input is reset to null by this method.
    //!
    //! \see getSize
    //!
    void setSize(Dims size) noexcept { mImpl->setSize(size); }

    //!
    //! \brief Get dimensions of the output slice.
    //!
    //! \return The output dimension, or an invalid Dims structure.
    //!
    //! If the third input is present and non-null,
    //! this function returns a Dims with nbDims = -1.
    //!
    //! \see setSize
    //!
    Dims getSize() const noexcept { return mImpl->getSize(); }

    //!
    //! \brief Set the stride for computing the output slice data.
    //!
    //! \param stride The dimensions of the stride to compute the values to store in the output slice.
    //!
    //! If a fourth input had been used to create this layer, that input is reset to null by this method.
    //!
    //! \see getStride
    //!
    void setStride(Dims stride) noexcept { mImpl->setStride(stride); }

    //!
    //! \brief Get the stride for the output slice.
    //!
    //! \return The slicing stride, or an invalid Dims structure.
    //!
    //! If the fourth input is present and non-null,
    //! this function returns a Dims with nbDims = -1.
    //!
    //! \see setStride
    //!
    Dims getStride() const noexcept { return mImpl->getStride(); }

    //!
    //! \brief Set the slice mode.
    //!
    //! \see getMode()
    //!
    void setMode(SliceMode mode) noexcept { mImpl->setMode(mode); }

    //!
    //! \brief Set the slice mode.
    //!
    //! \see getMode()
    //!
    SliceMode getMode() const noexcept { return mImpl->getMode(); }

    //!
    //! \brief Append or replace an input of this layer with a specific tensor
    //!
    //! \param index the index of the input to modify.
    //! \param tensor the new input tensor
    //!
    //! For a slice layer, the values 0-4 are valid.
    //! The indices are as follows:
    //!
    //! - 0: Tensor to be sliced.
    //! - 1: The start tensor to begin slicing, as a 1D Int32 shape tensor.
    //! - 2: The size tensor of the resulting slice, as a 1D Int32 shape tensor.
    //! - 3: The stride of the slicing operation, as a 1D Int32 shape tensor.
    //! - 4: Value for the kFILL slice mode. The fill value data type should either be the same
    //!      or be implicitly convertible to the input data type.
    //!      Implicit data type conversion is supported among kFLOAT, kHALF, kINT8, and kFP8 data types.
    //!      This input is disallowed for other modes.
    //!
    //! Using the corresponding setter resets the input to null.
    //!
    //! If this function is called with a value greater than 0, then the function getNbInputs() changes
    //! from returning 1 to index + 1.
    //!
    using ILayer::setInput;

   protected:
    apiv::VSliceLayer* mImpl;
    virtual ~ISliceLayer() noexcept = default;
};

//! \class IConstantLayer
//!
//! \brief Layer that represents a constant value.
//!
//! \note This layer does not support boolean types.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IConstantLayer : public ILayer {
   public:
    //!
    //! \brief Set the weights for the layer.
    //!
    //! If weights.type is DataType::kINT32, the output is a tensor of 32-bit indices.
    //! Otherwise the output is a tensor of real values and the output type will be
    //! follow TensorRT's normal precision rules.
    //!
    //! \see getWeights()
    //!
    void setWeights(Weights weights) noexcept { mImpl->setWeights(weights); }

    //!
    //! \brief Get the weights for the layer.
    //!
    //! \see setWeights
    //!
    Weights getWeights() const noexcept { return mImpl->getWeights(); }

    //!
    //! \brief Set the dimensions for the layer.
    //!
    //! \param dimensions The dimensions of the layer
    //!
    //! \see setDimensions
    //!
    void setDimensions(Dims dimensions) noexcept { mImpl->setDimensions(dimensions); }

    //!
    //! \brief Get the dimensions for the layer.
    //!
    //! \return the dimensions for the layer
    //!
    //! \see getDimensions
    //!
    Dims getDimensions() const noexcept { return mImpl->getDimensions(); }

   protected:
    apiv::VConstantLayer* mImpl;
    virtual ~IConstantLayer() noexcept = default;
};

//! \class INormalizationLayer
//!
//! \brief The normalization layer in a network definition.
//!
//! Output = (Input - Mean(Input, axes)) / Sqrt(Variance(Input) + epsilon) * Scale + Bias
//!
//! axes is the reduction axes.
//!
class INormalizationLayer : public ILayer {
   public:
    //! \brief Set the epsilon value.
    //!
    //! The default value of \p eps is 1e-5F.
    //!
    //! \param eps The epsilon value.
    //!
    void setEpsilon(float eps) noexcept { return mImpl->setEpsilon(eps); }

    //! \brief Get the epsilon value.
    //!
    //! \return The epsilon value.
    //!
    float getEpsilon() const noexcept { return mImpl->getEpsilon(); }

    //! \brief Set the reduction axes.
    //!
    //! \param axesMask The reduction axes.
    //!
    void setAxes(uint32_t axesMask) noexcept { return mImpl->setAxes(axesMask); }

    //! \brief Get the reduction axes.
    //!
    //! \return The reduction axes.
    //!
    uint32_t getAxes() const noexcept { return mImpl->getAxes(); }

    //! \brief Set the number of groups used to split the channels.
    //!
    //! The input tensor channels are divided into \p nbGroups groups, normalization is performed on each group.
    //! The channel dimension is considered to be the second dimension in a [N, C, H, W, ...] formatted tensor.
    //!
    //! The default \p nbGroups is 1.
    //!
    //! \warning  \p nbGroups must be a value that can divide the number of channels of the input tensor.
    //!
    //! \warning When \p nbGroups is != 1, it is expected that the provided axesMask will have all bits corresponding
    //! to dimensions after the channel dimension set to 1, with all other bits set to 0.
    //!
    //! \param nbGroups The number of groups to split the channels.
    //!
    void setNbGroups(int32_t nbGroups) noexcept { return mImpl->setNbGroups(nbGroups); }

    //! \brief Get the number of groups used to split the channels.
    //!
    //! \return The number of groups used to split the channels.
    //!
    int32_t getNbGroups() const noexcept { return mImpl->getNbGroups(); }

    //! \brief Set the compute precision of this layer.
    //!
    //! \param type The datatype used for the compute precision of this layer.
    //!
    //! Only DataType::kFLOAT32 and DataType::kHALF are valid types for \p type.
    //!
    void setComputePrecision(DataType type) noexcept { return mImpl->setComputePrecision(type); }

    //! \brief Get the compute precision of this layer.
    //!
    //! \return The datatype used for the compute precision of this layer.
    //!
    DataType getComputePrecision() const noexcept { return mImpl->getComputePrecision(); }

   protected:
    apiv::VNormalizationLayer* mImpl;
    virtual ~INormalizationLayer() noexcept = default;
};

struct Permutation {
    //!
    //! The elements of the permutation.
    //! To permute from CHW order to HWC order, the required permutation is [1, 2, 0].
    //! To permute from HWC to CHW, the required permutation is [2, 0, 1].
    //!
    int32_t order[Dims::MAX_DIMS];
};

//! \class IShuffleLayer
//!
//! \brief Layer type for shuffling data.
//!
//! This layer shuffles data by applying in sequence: a transpose operation, a reshape operation
//! and a second transpose operation.
//!
//! The layer has an optional second input.  If present, it must be a 1D Int32 shape tensor,
//! and the reshape dimensions are taken from it.
//!
class IShuffleLayer : public ILayer {
   public:
    //!
    //! \brief Set the permutation applied by the first transpose operation.
    //!
    //! \param permutation The dimension permutation applied before the reshape.
    //!
    //! The default is the identity permutation.
    //!
    //! \see getFirstTranspose
    //!
    void setFirstTranspose(Permutation permutation) noexcept { mImpl->setFirstTranspose(permutation); }

    //!
    //! \brief Get the permutation applied by the first transpose operation.
    //!
    //! \return The dimension permutation applied before the reshape.
    //!
    //! \see setFirstTranspose
    //!
    Permutation getFirstTranspose() const noexcept { return mImpl->getFirstTranspose(); }

    //!
    //! \brief Set the reshaped dimensions.
    //!
    //! \param dimensions The reshaped dimensions.
    //!
    //! Two special values can be used as dimensions.
    //!
    //! Value 0 copies the corresponding dimension from input. This special value
    //! can be used more than once in the dimensions. If number of reshape
    //! dimensions is less than input, 0s are resolved by aligning the most
    //! significant dimensions of input.
    //!
    //! Value -1 infers that particular dimension by looking at input and rest
    //! of the reshape dimensions. Note that only a maximum of one dimension is
    //! permitted to be specified as -1.
    //!
    //! The product of the new dimensions must be equal to the product of the old.
    //!
    //! If a second input had been used to create this layer, that input is reset to null by this method.
    //!
    void setReshapeDimensions(Dims dimensions) noexcept { mImpl->setReshapeDimensions(dimensions); }

    //!
    //! \brief Get the reshaped dimensions.
    //!
    //! \return The reshaped dimensions.
    //!
    //! If a second input is present and non-null, or setReshapeDimensions has
    //! not yet been called, this function returns Dims with nbDims == -1.
    //!
    Dims getReshapeDimensions() const noexcept { return mImpl->getReshapeDimensions(); }

    //!
    //! \brief Append or replace an input of this layer with a specific tensor
    //!
    //! \param index the index of the input to modify.
    //! \param tensor the new input tensor
    //
    //! Sets the input tensor for the given index. The index must be 0 for a static shuffle layer.
    //! A static shuffle layer is converted to a dynamic shuffle layer by calling setInput with an index 1.
    //! A dynamic shuffle layer cannot be converted back to a static shuffle layer.
    //!
    //! For a dynamic shuffle layer, the values 0 and 1 are valid.
    //! The indices in the dynamic case are as follows:
    //!
    //! - 0: Data or Shape tensor to be shuffled.
    //! - 1: The dimensions for the reshape operation, as a 1D Int32 shape tensor.
    //!
    //! If this function is called with the value 1, then the function getNbInputs() changes
    //! from returning 1 to 2.
    //!
    //! The reshape dimensions are treated identically to how they are treated if set statically
    //! via setReshapeDimensions. In particular, a -1 is treated as a wildcard even if dynamically
    //! supplied at runtime, and a 0 is treated as a placeholder if getZeroIsPlaceholder() = true,
    //! which is the default. If the placeholder interpretation of 0 is unwanted because the
    //! runtime dimension should be 0 when the reshape dimension is 0, be sure to call
    //! setZeroIsPlacholder(false) on the IShuffleLayer.
    //!
    //! \see setReshapeDimensions.
    //!
    using ILayer::setInput;

    //!
    //! \brief Set the permutation applied by the second transpose operation.
    //!
    //! \param permutation The dimension permutation applied after the reshape.
    //!
    //! The default is the identity permutation.
    //!
    //! \see getFirstTranspose
    //!
    void setSecondTranspose(Permutation permutation) noexcept { mImpl->setSecondTranspose(permutation); }

    //!
    //! \brief Get the permutation applied by the second transpose operation.
    //!
    //! \return The dimension permutation applied after the reshape.
    //!
    //! \see setSecondTranspose
    //!
    Permutation getSecondTranspose() const noexcept { return mImpl->getSecondTranspose(); }

    //!
    //! \brief Set meaning of 0 in reshape dimensions.
    //!
    //! If true, then a 0 in the reshape dimensions denotes copying the corresponding
    //! dimension from the first input tensor.  If false, then a 0 in the reshape
    //! dimensions denotes a zero-length dimension.
    //!
    //! Default: true
    //!
    //! \see getZeroIsPlaceholder();
    //!
    void setZeroIsPlaceholder(bool zeroIsPlaceholder) noexcept {
        return mImpl->setZeroIsPlaceholder(zeroIsPlaceholder);
    }

    //!
    //! \brief Get meaning of 0 in reshape dimensions.
    //!
    //! \return true if 0 is placeholder for corresponding input dimension,
    //!         false if 0 denotes a zero-length dimension.
    //!
    //! \see setZeroIsPlaceholder
    //!
    bool getZeroIsPlaceholder() const noexcept { return mImpl->getZeroIsPlaceholder(); }

   protected:
    apiv::VShuffleLayer* mImpl;
    virtual ~IShuffleLayer() noexcept = default;
};

//!
//! \class ITopKLayer
//!
//! \brief Layer that represents a TopK reduction.
//!
class ITopKLayer : public ILayer {
   public:
    //!
    //! \brief Set the operation for the layer.
    //!
    void setOperation(TopKOperation op) noexcept { mImpl->setOperation(op); }

    //!
    //! \brief Get the operation for the layer.
    //!
    TopKOperation getOperation() const noexcept { return mImpl->getOperation(); }

    //!
    //! \brief Set the static k value for the layer.
    //!
    void setK(int32_t k) noexcept { mImpl->setK(k); }

    //!
    //! \brief Get the k value for the layer.
    //!
    int32_t getK() const noexcept { return mImpl->getK(); }

    //!
    //! \brief Set which axes to reduce for the layer.
    //!
    void setReduceAxes(uint32_t reduceAxes) noexcept { mImpl->setReduceAxes(reduceAxes); }

    //!
    //! \brief Get the axes to reduce for the layer.
    //!
    uint32_t getReduceAxes() const noexcept { return mImpl->getReduceAxes(); }

    //!
    //! \brief Append or replace an input of this layer with a specific tensor
    //!
    using ILayer::setInput;

   protected:
    apiv::VTopKLayer* mImpl;
    virtual ~ITopKLayer() noexcept = default;
};

//! \class IFullyConnectedLayer
//!
//! \brief A fully connected layer in a network definition.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class TRT_DEPRECATED IFullyConnectedLayer : public ILayer {
   public:
    //!
    //! \brief Set the number of output channels `K` from the fully connected layer.
    //!
    //! \see getNbOutputChannels()
    //!
    void setNbOutputChannels(int32_t nbOutputs) noexcept { mImpl->setNbOutputChannels(nbOutputs); }

    //!
    //! \brief Get the number of output channels `K` from the fully connected layer.
    //!
    //! \see setNbOutputChannels()
    //!
    int32_t getNbOutputChannels() const noexcept { return mImpl->getNbOutputChannels(); }

    // //!
    // //! \brief Set the kernel weights, given as a `KxC` matrix in row-major order.
    // //!
    // //! \see getKernelWeights()
    // //!
    // void setKernelWeights(Weights weights) noexcept
    // {
    //     mImpl->setKernelWeights(weights);
    // }

    // //!
    // //! \brief Get the kernel weights.
    // //!
    // //! \see setKernelWeights()
    // //!
    // Weights getKernelWeights() const noexcept
    // {
    //     return mImpl->getKernelWeights();
    // }

    // //!
    // //! \brief Set the bias weights.
    // //!
    // //! Bias is optional. To omit bias, set the count value in the weights structure to zero.
    // //!
    // //! \see getBiasWeightsWeights()
    // //!
    // void setBiasWeights(Weights weights) noexcept
    // {
    //     mImpl->setBiasWeights(weights);
    // }

    // //!
    // //! \brief Get the bias weights.
    // //!
    // //! \see setBiasWeightsWeights()
    // //!
    // Weights getBiasWeights() const noexcept
    // {
    //     return mImpl->getBiasWeights();
    // }

    //! \param index the index of the input to modify.
    //! \param tensor the new input tensor
    //!
    //! Only index 0 (data input) is valid
    //!
    using ILayer::setInput;

   protected:
    virtual ~IFullyConnectedLayer() noexcept = default;
    apiv::VFullyConnectedLayer* mImpl;
};

//! \class IShapeLayer
//!
//! \brief Layer type for getting shape of a tensor.
//!
//! This layer sets the output to a 1D tensor of type Int32 with the dimensions of the input tensor.
//!
class IShapeLayer : public ILayer {
   protected:
    apiv::VShapeLayer* mImpl;
    virtual ~IShapeLayer() noexcept = default;
};

//! \class IAssertionLayer
//!
//! \brief An assertion layer in a network
//!
//! The layer has one input and no output.
//! The input must be a boolean shape tensor.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IAssertionLayer : public ILayer {
   public:
    //!
    //! \brief Set the message to print if the assertion fails.
    //!
    //! \see getMessage()
    //!
    void setMessage(char const* message) noexcept { mImpl->setMessage(message); }

    //!
    //! \brief Return the assertion message.
    //!
    //! \see setMessage()
    //!
    char const* getMessage() const noexcept { return mImpl->getMessage(); }

   protected:
    virtual ~IAssertionLayer() noexcept = default;

    apiv::VAssertionLayer* mImpl;
};

class INetworkDefinition : public INoCopy {
   public:
    virtual ~INetworkDefinition() noexcept = default;

    //!
    //! \brief Get the number of layers in the network.
    //!
    //! \return The number of layers in the network.
    //!
    int32_t getNbLayers() const noexcept { return mImpl->getNbLayers(); }

    //!
    //! \brief Get the layer specified by the given index.
    //!
    //! \param index The index of the layer.
    //!
    //! \return The layer, or nullptr if the index is out of range.
    //!
    ILayer* getLayer(int32_t index) const noexcept { return mImpl->getLayer(index); }

    //!
    //! \brief Get the number of inputs in the network.
    //!
    //! \return The number of inputs in the network.
    //!
    int32_t getNbInputs() const noexcept { return mImpl->getNbInputs(); }

    //!
    //! \brief Get the network output tensor specified by the given index.
    //!
    //! \param index The index of the input tensor.
    //!
    //! \return The input tensor, or nullptr if the index is out of range.
    //!
    ITensor* getInput(int32_t index) const noexcept { return mImpl->getInput(index); }

    //!
    //! \brief Get the number of outputs in the network.
    //!
    //! \return The number of outputs in the network.
    //!
    int32_t getNbOutputs() const noexcept { return mImpl->getNbOutputs(); }

    //!
    //! \brief Get the network output tensor specified by the given index.
    //!
    //! \param index The index of the output tensor.
    //!
    //! \return The output tensor, or nullptr if the index is out of range.
    //!
    ITensor* getOutput(int32_t index) const noexcept { return mImpl->getOutput(index); }

    //!
    //! \brief Sets the name of the network.
    //!
    //! \param name The name to assign to this network.
    //!
    void setName(char const* name) noexcept { mImpl->setName(name); }

    //!
    //! \brief Gets the name of the network.
    //!
    char const* getName() const noexcept { return mImpl->getName(); }

    ITensor* addInput(char const* name, DataType type, Dims dimensions) noexcept {
        return mImpl->addInput(name, type, dimensions);
    }

    void markOutput(ITensor& tensor) noexcept { mImpl->markOutput(tensor); }

    IConvolutionLayer* addConvolution(ITensor& input, int32_t nbOutputMaps, DimsHW kernelSize, Weights kernelWeights,
                                      Weights biasWeights) noexcept {
        return mImpl->addConvolution(input, nbOutputMaps, kernelSize, kernelWeights, biasWeights);
    }

    //!
    //! \brief Add a 2D convolution layer to the network.
    //!
    //! \param input The input tensor.
    //! \param nbOutputMaps The number of output channels.
    //! \param kernelSize The multi-dimensions of the convolution kernel.
    //! \param kernelWeights The kernel weights.
    //! \param biasWeights The bias weights. Weights{} represents no bias.
    //!
    //! \warning It is an error to specify a wildcard value for the 'C' dimension of the input tensor.
    //! \warning Only 2D convolution is supported. 3D convolution isn't supported.
    //!
    //! \return The new convolution layer, or nullptr if it could not be created.
    //!
    IConvolutionLayer* addConvolutionNd(ITensor& input, int32_t nbOutputMaps, Dims kernelSize, Weights kernelWeights,
                                        Weights biasWeights) noexcept {
        return mImpl->addConvolutionNd(input, nbOutputMaps, kernelSize, kernelWeights, biasWeights);
    }

    //!
    //! \brief Add a 2D deconvolution layer to the network.
    //!
    //! \param input The input tensor.
    //! \param nbOutputMaps The number of output channels.
    //! \param kernelSize The multi-dimensions of the convolution kernel.
    //! \param kernelWeights The kernel weights.
    //! \param biasWeights The bias weights. Weights{} represents no bias.
    //!
    //! \warning It is an error to specify a wildcard value for the 'C' dimension of the input tensor.
    //! \warning Int32 tensors are not valid input tensors.
    //! \warning Only 2D deconvolution is supported. 3D deconvolution isn't supported.
    //
    //! \return The new deconvolution layer, or nullptr if it could not be created.
    //!
    IDeconvolutionLayer* addDeconvolutionNd(ITensor& input, int32_t nbOutputMaps, Dims kernelSize,
                                            Weights kernelWeights, Weights biasWeights) noexcept {
        return mImpl->addDeconvolutionNd(input, nbOutputMaps, kernelSize, kernelWeights, biasWeights);
    }

    //!
    //! \brief Add a reduce layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //! \param operation The reduction operation to perform.
    //! \param reduceAxes The reduction dimensions.
    //! \param keepDimensions The boolean that specifies whether or not to keep the reduced dimensions in the
    //! output of the layer.
    //!
    //! \return The new reduce layer, or nullptr if it could not be created.
    //!
    IReduceLayer* addReduce(ITensor& input, ReduceOperation operation, uint32_t reduceAxes,
                            bool keepDimensions) noexcept {
        return mImpl->addReduce(input, operation, reduceAxes, keepDimensions);
    }

    //!
    //! \brief Add a SoftMax layer to the network.
    //!
    //! \return The new SoftMax layer, or nullptr when failed.
    //!
    ISoftMaxLayer* addSoftMax(ITensor& input) noexcept { return mImpl->addSoftMax(input); }

    //!
    //! \brief Add an activation layer to the network.
    //!
    //! \param input The layer input tensor
    //! \param type Activation function type
    //!
    //! If activation type require parameters like alpha or beta, use setAlpha() and setBeta() methods must be called
    //!
    //! \see IActivationLayer ActivationType
    //! \warning Int32 tensors are invalid input tensors.
    //!
    //! \return The new activation layer pointer, or nullptr if create layer failed.
    //!
    IActivationLayer* addActivation(ITensor& input, ActivationType type) noexcept {
        return mImpl->addActivation(input, type);
    }

    //!
    //! \brief Add a parametric ReLU layer to the network.
    //!
    //! \param input Input tensor of layer.
    //! \param slope Slope tensor of layer. This tensor should be keep same shape with input tensor or
    //! unidirectionally broadcastable to the input tensor.
    //!
    //! \warning Int32 tensors are invalid input tensors.
    //!
    //! \return The new parametric ReLU layer, or nullptr if it create layer failed
    //!
    IParametricReLULayer* addParametricReLU(ITensor& input, ITensor& slope) noexcept {
        return mImpl->addParametricReLU(input, slope);
    }

    //!
    //! \brief Add an elementwise layer to the network.
    //!
    //! \param input1 The first input tensor.
    //! \param input2 The second input tensor.
    //! \param op The binary operation that the layer applies.
    //!
    //! Supports broadcast.
    //!
    //! \return The new elementwise layer, or nullptr if it could not be created.
    //!
    IElementWiseLayer* addElementWise(ITensor& input1, ITensor& input2, ElementWiseOperation op) noexcept {
        return mImpl->addElementWise(input1, input2, op);
    }

    //!
    //! \brief Add gather with mode GatherMode::kDEFAULT and specified axis and nbElementWiseDims=0.
    //!
    //! \param data The input tensor to gather values from.
    //! \param indices The tensor to get indices from to populate the output tensor.
    //! \param axis The axis in the data tensor to gather on.
    //!
    //! \see IGatherLayer
    //!
    //! \return The new gather layer, or nullptr if it create failed
    //!
    IGatherLayer* addGather(ITensor& data, ITensor& indices, int32_t axis) noexcept {
        return mImpl->addGather(data, indices, axis);
    }

    //!
    //! \brief Add gather with specified mode, default value will be axis=0 and nbElementWiseDims=0.
    //!
    //! \param data The input tensor to gather values from.
    //! \param indices The tensor to get indices from to populate the output tensor.
    //! \param mode The gather mode.
    //!
    //! \see IGatherLayer
    //!
    //! \return The new gather layer, or nullptr if it could not be created.
    //!
    IGatherLayer* addGatherV2(ITensor& data, ITensor& indices, GatherMode mode) {
        return mImpl->addGatherV2(data, indices, mode);
    }

    //!
    //! \brief Add plugin layer to the network.
    //!
    //! \param inputs The input tensors.
    //! \param nbInputs The number of input tensors.
    //! \param plugin The plugin.
    //!
    //! \see IPluginV2Layer
    //!
    //! \return The plugin layer created; can be nullptr if it did not created successfully.
    //!
    IPluginV2Layer* addPluginV2(ITensor* const* inputs, int32_t nbInputs, IPluginV2& plugin) noexcept {
        return mImpl->addPluginV2(inputs, nbInputs, plugin);
    }

    //!
    //! \brief Add pooling layer to the network.
    //!
    //! \param inputs The input tensors.
    //! \param type The type to the pooling layer
    //! \param windowSize Symmetric window size of pooling window, either 2 or 3 dimensions supported
    //!
    //! \return The pooling layer created; can be nullptr if it did not created successfully.
    //!
    IPoolingLayer* addPoolingNd(ITensor& input, PoolingType type, Dims windowSize) noexcept {
        return mImpl->addPoolingNd(input, type, windowSize);
    }

    //!
    //! \brief Add a resize layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //!
    //! \return The resize layer created, or nullptr if it did not created successfully.
    //!
    IResizeLayer* addResize(ITensor& input) noexcept { return mImpl->addResize(input); }

    //!
    //! \brief Add a unary layer to network.
    //!
    //! \param input The input tensor to the layer.
    //! \param operation The operation to apply.
    //!
    //! \return The new unary layer, or or nullptr if it create failed
    //!
    IUnaryLayer* addUnary(ITensor& input, UnaryOperation operation) noexcept {
        return mImpl->addUnary(input, operation);
    }

    //!
    //! \brief Add a concatenation layer to the model.
    //!
    //! \param inputs The input tensors list
    //! \param nbInputs The length of input tensor list.
    //!
    //! \return The new concatenation layer, or nullptr if create failed
    //!
    IConcatenationLayer* addConcatenation(ITensor* const* inputs, int32_t nbInputs) noexcept {
        return mImpl->addConcatenation(inputs, nbInputs);
    }
    //!
    //! \brief Add a quantization layer to the network.
    //!
    //! \param input The input tensor to be quantized.
    //! \param scale The scale value calculated within quantization formula.
    //!
    //! \see IQuantizeLayer
    //!
    //! \p data type of input tensor must be DataType::kFLOAT.
    //! \p data type of scale tensor must be DataType::kFLOAT.
    //!
    //! \return The new quantization layer, or nullptr if it could not be created.
    //!
    IQuantizeLayer* addQuantize(ITensor& input, ITensor& scale) noexcept { return mImpl->addQuantize(input, scale); }
    //!
    //! \brief Add a dequantization layer to the network.
    //!
    //! \param input The input tensor to be dequantized.
    //! \param scale The scale value calculated within quantization formula.
    //!
    //! \see IDequantizeLayer
    //!
    //! \p data type of input tensor must be DataType::kFLOAT.
    //! \p data type of scale tensor must be DataType::kFLOAT.
    //!
    //! \return The new dequantization layer, or nullptr if it could not be created.
    //!
    IDequantizeLayer* addDequantize(ITensor& input, ITensor& scale) noexcept {
        return mImpl->addDequantize(input, scale);
    }

    //! \brief Add a fill layer to the network.
    //!
    //! \param dimensions Shape of output tensor to be generated
    //! \param op The fill operation that the layer applies.
    //!
    //! \see IFillLayer
    //!
    //! \return The new fill layer, or nullptr if it could not be created.
    //!
    IFillLayer* addFill(Dims dimensions, FillOperation op) noexcept { return mImpl->addFill(dimensions, op); }

    //!
    //! \brief Add an identity layer to the model.
    //!
    //! \param input The input tensor to the layer.
    //!
    //! \see IIdentityLayer
    //!
    //! \return The new identity layer, or nullptr if create failed
    //!
    IIdentityLayer* addIdentity(ITensor& input) noexcept { return mImpl->addIdentity(input); }

    //! \brief Add MatrixMultiply layer to the network.
    //!
    //! \param input0 The first input tensor.
    //! \param op0 The operation apply to input0.
    //! \param input1 The second input tensor.
    //! \param op1 The operation apply to input1.
    //!
    //! \see IMatrixMultiplyLayer
    //!
    //! \return The new matrix multiply layer, or nullptr if it could not be created.
    //!
    IMatrixMultiplyLayer* addMatrixMultiply(ITensor& input0, MatrixOperation op0, ITensor& input1,
                                            MatrixOperation op1) noexcept {
        return mImpl->addMatrixMultiply(input0, op0, input1, op1);
    }
    //!
    //! \brief Add a cast layer.
    //!
    //! \param input The input tensor to the layer.
    //! \param toType The DataType of the output tensor
    //!
    //! \see ICastLayer
    //!
    //! \return The new cast layer, or nullptr if it could not be created.
    //!
    ICastLayer* addCast(ITensor& input, DataType toType) noexcept { return mImpl->addCast(input, toType); }

    //! \brief Add a select layer to the network.
    //!
    //! \param condition The condition tensor to the layer. Must have type DataType::kBOOL.
    //! \param thenInput The "then" input tensor to the layer.
    //! \param elseInput The "else" input tensor to the layer.
    //!
    //! All input tensors must have the same rank, and along each axis
    //! must have the same length or a length of one for broadcasting.
    //
    //! \return The new select layer, or nullptr if it create failed
    ISelectLayer* addSelect(ITensor& condition, ITensor& thenInput, ITensor& elseInput) noexcept {
        return mImpl->addSelect(condition, thenInput, elseInput);
    }

    //!
    //! \brief Add slice layer to the network.
    //!
    //! \param inputs The input tensors.
    //! \param start The start dims of the input, each dim is the slice start index.
    //! \param size The size dims of the input, each dim is the slice shape size.
    //! \param stride The stride dims of the input, each dim is the slice step index
    //!
    //! \return The slice layer created; can be nullptr if it did not created successfully.
    //!
    ISliceLayer* addSlice(ITensor& input, Dims start, Dims size, Dims stride) noexcept {
        return mImpl->addSlice(input, start, size, stride);
    }

    //!
    //! \brief Add a constant layer to the network.
    //!
    //! \param dimensions The dimensions of the constant.
    //! \param weights The constant value, represented as weights.
    //!
    //! \see IConstantLayer
    //!
    //! \return The new constant layer, or nullptr if it could not be created.
    //!
    //! If weights.type is DataType::kINT32, the output is a tensor of 32-bit indices.
    //! Otherwise the output is a tensor of real values and the output type will be
    //! follow TensorRT's normal precision rules.
    //!
    //! If tensors in the network have an implicit batch dimension, the constant
    //! is broadcast over that dimension.
    //!
    //! If a wildcard dimension is used, the volume of the runtime dimensions must equal
    //! the number of weights specified.
    //!
    //! \warning DataType::kUINT8 not supported.
    //!
    IConstantLayer* addConstant(Dims dimensions, Weights weights) noexcept {
        return mImpl->addConstant(dimensions, weights);
    }
    //! \brief Add an Einsum layer to the network.
    //!
    //! \param inputs The input tensors to the layer.
    //! \param nbInputs The number of input tensors.
    //! \param equation The equation of the layer
    //! \see IEinsumLayer
    //!
    //! \return The new Einsum layer, or nullptr if it could not be created.
    //!
    IEinsumLayer* addEinsum(ITensor* const* inputs, int32_t nbInputs, char const* equation) noexcept {
        return mImpl->addEinsum(inputs, nbInputs, equation);
    }
    //!
    //! \brief Add a normalization layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //! \param scale The scale tensor multiplying to the normalized output.
    //! \param bias The bias tensor added to the normalized output.
    //! \param axesMask The axes on which to perform mean calculations.
    //!
    //! The normalization layer works by performing normalization of the tensor \p input on the specified \p axesMask.
    //! The result is then scaled by multiplying with \p scale and adding \p bias.
    //!
    //! \see INormalizationLayer
    //!
    //! \return The new normalization layer, or nullptr if it could not be created.
    //!
    INormalizationLayer* addNormalization(ITensor& input, ITensor& scale, ITensor& bias, uint32_t axesMask) noexcept {
        return mImpl->addNormalization(input, scale, bias, axesMask);
    }

    //! \brief Add a shuffle layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //!
    //! \return The new shuffle layer, or nullptr if it could not be created.
    //!
    IShuffleLayer* addShuffle(ITensor& input) noexcept { return mImpl->addShuffle(input); }

    //!
    //! \brief Add a TopK layer to the network.
    //!
    //! The TopK layer has two outputs of the same dimensions.
    //! Output 0 : data values,
    //! Output 1 : index positions for the values.
    //!
    //! \param input The input tensor to the layer.
    //! \param op Operation to perform.
    //! \param k Number of elements to keep.
    //! \param reduceAxes The reduction dimensions.
    //!
    //! \return New TopK layer, or nullptr when failed.
    //!
    ITopKLayer* addTopK(ITensor& input, TopKOperation op, int32_t k, uint32_t reduceAxes) noexcept {
        return mImpl->addTopK(input, op, k, reduceAxes);
    }

    //! \brief Add a GridSample layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //! \param grid The grid tensor to the layer.
    //!
    //! \return The new GridSample layer, or nullptr if it could not be created.
    //!
    IGridSampleLayer* addGridSample(ITensor& input, ITensor& grid) noexcept {
        return mImpl->addGridSample(input, grid);
    }

    //!
    //! \brief Add a shape layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //!
    //! \see IShapeLayer
    //!
    IShapeLayer* addShape(ITensor& input) noexcept { return mImpl->addShape(input); }

    //! \brief Add a fully connected layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //! \param nbOutputs The number of outputs of the layer.
    //! \param kernelWeights The kernel weights for the fully connected layer.
    //! \param biasWeights The bias weights for the fully connected layer. Weights{} represents no bias.
    //!
    //! \see IFullyConnectedLayer
    //!
    //! \warning It is an error to specify a wildcard value for the 'C' dimension of the input tensor.
    //! \warning Int32 tensors are not valid input tensors.
    //!
    //! \return The new fully connected layer, or nullptr if it could not be created.
    //!
    TRT_DEPRECATED IFullyConnectedLayer* addFullyConnected(ITensor& input, int32_t nbOutputs, Weights kernelWeights,
                                                           Weights biasWeights) noexcept {
        return mImpl->addFullyConnected(input, nbOutputs, kernelWeights, biasWeights);
    }
    //!
    //! \brief Add an assertion layer to the network.
    //!
    //! \param condition The input tensor to the layer.
    //! \param message A message to print if the assertion fails.
    //!
    //! \see IAssertionLayer
    //!
    //! \return The new assertion layer, or nullptr if it could not be created.
    //!
    //! The input tensor must be a boolean shape tensor.
    //!
    IAssertionLayer* addAssertion(ITensor& condition, char const* message) noexcept {
        return mImpl->addAssertion(condition, message);
    }
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
    apiv::VNetworkDefinition* mImpl;
};

enum class CalibrationAlgoType : int32_t {
    kENTROPY_CALIBRATION_2 = 2,
    kMINMAX_CALIBRATION = 3,
};

template <>
constexpr inline int32_t EnumMax<CalibrationAlgoType>() noexcept {
    return 2;
}

class IInt8Calibrator {
   public:
    virtual int32_t getBatchSize() const noexcept = 0;

    virtual bool getBatch(void* bindings[], char const* names[], int32_t nbBindings) noexcept = 0;

    // TODO: Remove this Interface.
    virtual bool getBatch(void** bindings, const char* names, int nbBindings) noexcept = 0;

    virtual void const* readCalibrationCache(std::size_t& length) noexcept = 0;

    virtual void writeCalibrationCache(void const* ptr, std::size_t length) noexcept = 0;

    virtual CalibrationAlgoType getAlgorithm() noexcept = 0;

    virtual ~IInt8Calibrator() noexcept = default;
};

class IInt8EntropyCalibrator2 : public IInt8Calibrator {
   public:
    CalibrationAlgoType getAlgorithm() noexcept override { return CalibrationAlgoType::kENTROPY_CALIBRATION_2; }

    virtual ~IInt8EntropyCalibrator2() noexcept = default;
};

class IInt8MinMaxCalibrator : public IInt8Calibrator {
   public:
    CalibrationAlgoType getAlgorithm() noexcept override { return CalibrationAlgoType::kMINMAX_CALIBRATION; }

    virtual ~IInt8MinMaxCalibrator() noexcept = default;
};

//!
//! \enum BuilderFlag
//! \brief Layer precision used by builder when creating engine.
//!
enum class BuilderFlag : int32_t {
    //!< Enable float16 layer precision.
    kFP16 = 0,
    //!< Enable int8 layer precision.
    kINT8 = 1,
    //!< Disable reuse of timing information across identical layers.
    kDISABLE_TIMING_CACHE = 6,

    //!< Require that layers execute in specified precisions. Build fails otherwise.
    kOBEY_PRECISION_CONSTRAINTS = 10,

    //!< Prefer that layers execute in specified precisions.
    //!< Fall back (with warning) to another precision if build would otherwise fail.
    kPREFER_PRECISION_CONSTRAINTS = 11,
};

template <>
constexpr inline int32_t EnumMax<BuilderFlag>() noexcept {
    return 3;
}
//!
//! \class ITimingCache
//!
//! \brief Class to handle tactic timing info collected from builder.
//!
//! The timing cache is created or initialized by IBuilderConfig. It can be shared across builder instances
//! to accelerate the builder wallclock time.
//!
//! \see IBuilderConfig
//!
class ITimingCache : public INoCopy {
   public:
    virtual ~ITimingCache() noexcept = default;

    //!
    //! \brief Serialize a timing cache to IHostMemory object.
    //!
    //! This function allows serialization of current timing cache.
    //!
    //! \return A pointer to a IHostMemory object that contains a serialized timing cache.
    //!
    //! \see IHostMemory
    //!
    nvinfer1::IHostMemory* serialize() const noexcept { return mImpl->serialize(); }

    //!
    //! \brief Combine input timing cache into local instance.
    //!
    //! This function allows combining entries in the input timing cache to local cache object.
    //!
    //! \param inputCache The input timing cache.
    //! \param ignoreMismatch Whether or not to allow cache verification header mismatch.
    //!
    //! \return True if combined successfully, false otherwise.
    //!
    //! Append entries in input cache to local cache. Conflicting entries will be skipped
    //! The input cache must be generated by a TensorRT build of exact same version, otherwise
    //! combine will be skipped and return false.
    //! ignoreMismatch must be set to true if combining a timing cache created from a
    //! different device.
    //!
    //! \warning Combining caches generated from devices with different device properties may
    //!          lead to functional/performance bugs!
    //!
    bool combine(ITimingCache const& inputCache, bool ignoreMismatch) noexcept {
        return mImpl->combine(inputCache, ignoreMismatch);
    }

    //!
    //! \brief Empty the timing cache
    //!
    //! \return True if reset successfully, false otherwise.
    //!
    bool reset() noexcept { return mImpl->reset(); }

   protected:
    apiv::VTimingCache* mImpl;
};

//!
//! \enum PreviewFeature
//!
//! \brief Define preview features
//!
enum class PreviewFeature : int32_t {
    //!
    //! Allows optimization profiles to be shared across execution contexts.
    //!
    kPROFILE_SHARING_0806 = 0,
};

//!
//! \class IBuilderConfig
//!
//! \brief Specifys the way to build the engine
//!
//! \see BuilderFlags
//!
class IBuilderConfig : public INoCopy {
   public:
    virtual ~IBuilderConfig() noexcept = default;

    void setFlags(BuilderFlags builderFlags) noexcept { mImpl->setFlags(builderFlags); }

    //!
    //! \brief Get bitmask build options. Defaults to 0.
    //!
    //! \return The bitmask build options.
    //!
    //! \see setFlags()
    //!
    BuilderFlags getFlags() const noexcept { return mImpl->getFlags(); }

    //!
    //! \brief clear only a single builder config
    //!
    //! clears the builder mode flag from the enabled flags.
    //!
    //! \see setFlags()
    //!
    void clearFlag(BuilderFlag builderFlag) noexcept { mImpl->clearFlag(builderFlag); }

    //!
    //! \brief Set a single build mode flag.
    //!
    //! Add the input builder mode flag to the already enabled flags.
    //!
    //! \see setFlags()
    //!
    void setFlag(BuilderFlag builderFlag) noexcept { mImpl->setFlag(builderFlag); }

    //!
    //! \brief Test if the build mode flag is set
    //!
    //! \see getFlags()
    //!
    //! \return True if flag is set, false if unset.
    //!
    bool getFlag(BuilderFlag builderFlag) const noexcept { return mImpl->getFlag(builderFlag); }

    //!
    //! \brief Resets the builder configuration to defaults.
    //!
    //! Can be used to initialize the original state of the configuration.
    //!
    void reset() noexcept { mImpl->reset(); }

    //!
    //! \brief Add an optimization profile.
    //!
    //! \param profile The new optimization profile, which must satisfy profile->isValid() == true
    //! \return The index of the optimization profile (starting from 0) if the input is valid, or -1 if the input is
    //!         not valid.
    //!
    int32_t addOptimizationProfile(IOptimizationProfile const* profile) noexcept {
        return mImpl->addOptimizationProfile(profile);
    }

    //!
    //! \brief Get number of optimization profiles.
    //!
    //! This is one higher than the index of the last optimization profile that has be defined (or
    //! zero, if none has been defined yet).
    //!
    //! \return The number of the optimization profiles.
    //!
    int32_t getNbOptimizationProfiles() const noexcept { return mImpl->getNbOptimizationProfiles(); }

    //!
    //! \brief Set verbosity level of layer information exposed in NVTX annotations and IEngineInspector.
    //!
    //! Control how much layer information will be exposed in NVTX annotations and IEngineInspector.
    //!
    void setProfilingVerbosity(ProfilingVerbosity verbosity) noexcept { mImpl->setProfilingVerbosity(verbosity); }

    //!
    //! \brief Get verbosity level of layer information exposed in NVTX annotations and IEngineInspector.
    //!
    //! Get the current setting of verbosity level of layer information exposed in
    //! NVTX annotations and IEngineInspector. Default value is ProfilingVerbosity::kLAYER_NAMES_ONLY.
    //!
    //! \return ProfilingVerbosity
    //!
    ProfilingVerbosity getProfilingVerbosity() const noexcept { return mImpl->getProfilingVerbosity(); }

    //!
    //! \brief Get Int8 Calibration interface.
    //!
    IInt8Calibrator* getInt8Calibrator() const noexcept { return mImpl->getInt8Calibrator(); }

    //!
    //! \brief Set Int8 Calibration interface.
    //!
    void setInt8Calibrator(IInt8Calibrator* calibrator) noexcept { mImpl->setInt8Calibrator(calibrator); }

    //!
    //! \brief Create timing cache
    //!
    //! \param blob A pointer to the raw buffer of serialized timing cache
    //! \param size The size of the serialized timing cache. Size 0 means create a new cache from scratch
    //!
    //! \return the pointer to ITimingCache created
    //!
    nvinfer1::ITimingCache* createTimingCache(void const* blob, std::size_t size) const noexcept {
        return mImpl->createTimingCache(blob, size);
    }

    //!
    //! \brief Attach a timing cache to IBuilderConfig
    //!
    //! The cache must not be destroyed until the engine has been built.
    //!
    //! \param cache the timing cache that will be used
    //! \param ignoreMismatch whether or not allow using a cache that contains different CUDA device property
    //!
    //! \return true if set successfully, false otherwise
    //!
    //!
    bool setTimingCache(ITimingCache const& cache, bool ignoreMismatch) noexcept {
        return mImpl->setTimingCache(cache, ignoreMismatch);
    }

    //!
    //! \brief Get the pointer of the timing cache from current IBuilderConfig
    //!
    //! \return pointer to the current timing cache
    //!
    nvinfer1::ITimingCache const* getTimingCache() const noexcept { return mImpl->getTimingCache(); }

    //!
    //! \brief Enable or disable a specific preview feature
    //!
    //! Allows enabling or disabling experimental features, which are not enabled by default in the
    //! current release.
    //!
    //! Refer to PreviewFeature for additional information, and a list of the available features.
    //!
    //! \param feature the feature to enable / disable
    //! \param enable true for enable, false for disable
    //!
    //! \see PreviewFeature, getPreviewFeature
    //!
    void setPreviewFeature(PreviewFeature feature, bool enable) noexcept { mImpl->setPreviewFeature(feature, enable); }

    //!
    //! \brief Get status of preview feature
    //!
    //! \param feature the feature to query
    //!
    //! \returns true if the \p feature is enabled, false otherwise
    //!
    //! \see PreviewFeature, setPreviewFeature
    //!
    bool getPreviewFeature(PreviewFeature feature) const noexcept { return mImpl->getPreviewFeature(feature); }

   protected:
    apiv::VBuilderConfig* mImpl;
};

using NetworkDefinitionCreationFlags = uint32_t;
//!
//! \enum NetworkDefinitionCreationFlag
//! \brief Flags used by builder when creating network.
//!
enum class NetworkDefinitionCreationFlag : int32_t {
    //! Batch dimension of network should be explicit.
    kEXPLICIT_BATCH = 0,
};

//!
//! \brief Build IxRT Network to engine
//!
class IBuilder : public INoCopy {
   public:
    virtual ~IBuilder() noexcept = default;
    //!
    //! \brief Test if platform has fast fp16
    //!
    bool platformHasFastFp16() const noexcept { return mImpl->platformHasFastFp16(); }

    //!
    //! \brief Test if platform has fast int8
    //!
    bool platformHasFastInt8() const noexcept { return mImpl->platformHasFastInt8(); }
    //!
    //! \brief Set the GPU allocator.
    //! \param allocator The GPU allocator to be used by the builder. All GPU memory acquired will use this
    //! allocator. If NULL is passed, use cudaMalloc/cudaFree as default.
    //!
    void setGpuAllocator(IGpuAllocator* allocator) noexcept { mImpl->setGpuAllocator(allocator); }
    //!
    //! \brief Create a configuration object of current builder.
    //!
    //! \see IBuilderConfig
    //!
    IBuilderConfig* createBuilderConfig() noexcept { return mImpl->createBuilderConfig(); }
    //!
    //! \brief Create a network definition object
    //!
    //! \param flags Bitset of NetworkDefinitionCreationFlags specifying network properties combined with bitwise OR.
    //!             e.g., 1U << NetworkDefinitionCreationFlag::kEXPLICIT_BATCH
    INetworkDefinition* createNetworkV2(NetworkDefinitionCreationFlags flags) noexcept {
        return mImpl->createNetworkV2(flags);
    }
    //!
    //! \brief Create an optimization profile
    //!
    IOptimizationProfile* createOptimizationProfile() noexcept { return mImpl->createOptimizationProfile(); }
    //!
    //! \brief Reset internal state of the builder to default values
    //!
    void reset() noexcept { mImpl->reset(); }
    //!
    //! \brief Build and serialize a network without creating an engine
    //! \param network The network definition for building
    //! \param config The config specifying how to build the network
    //!
    IHostMemory* buildSerializedNetwork(INetworkDefinition& network, IBuilderConfig& config) noexcept {
        return mImpl->buildSerializedNetwork(network, config);
    }
    //!
    //! \brief Check the network is available with the given config
    //! \param network The network definition for checking
    //! \param config The config of the network
    //!
    bool isNetworkSupported(INetworkDefinition const& network, IBuilderConfig const& config) const noexcept {
        return mImpl->isNetworkSupported(network, config);
    }
    //!
    //! \brief Get logger of the builder
    //!
    //! \return Pointer to the logger, nullptr if there's no logger
    ILogger* getLogger() const noexcept { return mImpl->getLogger(); }
    //!
    //! \brief Set max threads for builder
    //!
    //! \param maxThreads  Number of max threads used by builder
    //! \return true if set done, false for fail
    //!
    bool setMaxThreads(int32_t maxThreads) noexcept { return mImpl->setMaxThreads(maxThreads); }

    //!
    //! \brief Get number of max threads used by builder
    //!
    int32_t getMaxThreads() const noexcept { return mImpl->getMaxThreads(); }
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
    apiv::VBuilder* mImpl;
};

}  // namespace nvinfer1

extern "C" void* createInferBuilder_INTERNAL(void* logger, int32_t version) noexcept;

namespace nvinfer1 {
//!
//! \brief Create the builder object
//!
inline IBuilder* createInferBuilder(ILogger& logger) noexcept {
    return static_cast<IBuilder*>(createInferBuilder_INTERNAL(&logger, 0));
}
}  // namespace nvinfer1
