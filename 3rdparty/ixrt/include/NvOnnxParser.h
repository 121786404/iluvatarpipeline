#pragma once
#include <cstddef>
#include <vector>

#include "NvInfer.h"

extern "C" void* createOnnxParser_INTERNAL(void* network, void* logger, int version);

typedef std::pair<std::vector<size_t>, bool> SubGraph_t;

typedef std::vector<SubGraph_t> SubGraphCollection_t;

namespace nvonnxparser {
//!
//! \class IParser
//! \brief Onnx parser class to process a onnx (or maybe quantization json file) to IxRT network definition
//!
class IParser {
   public:
    //!
    //! \brief Parse onnx file
    //!
    //! \param onnxModelFile c string representing onnx path
    //! \param verbosity extent of showing the log
    //!
    //! \see parseFromFiles(const char* onnxModelFile, const char* quantParamFile, int verbosity)
    //!
    virtual bool parseFromFile(const char* onnxModelFile, int verbosity) = 0;
    //!
    //! \brief Parse serialized onnx model to build IxRT network
    //!
    //! \param serialized_onnx_model pointer to onnx model data memory
    //! \param serialized_onnx_model_size byte size of the serialized_onnx_model
    //! \param model_path if specified, load weights from it, note is has to be absolute path
    //!
    //! \see parseFromFiles(void const* serialized_onnx_model, size_t serialized_onnx_model_size, const char* quantFile,
    //! const char* model_path = nullptr)
    virtual bool parse(void const* serialized_onnx_model, size_t serialized_onnx_model_size,
                       const char* model_path = nullptr) = 0;
    //!
    //! \brief Parse onnx file with external quantized params
    //!
    //! \param onnxModelFile c string representing onnx path
    //! \param quantParamFile c string representing quantization param path, must be ppq style
    //! \param verbosity extent of showing the log
    //!
    //! \see parseFromFile(const char* onnxModelFile, int verbosity)
    //!
    virtual bool parseFromFiles(const char* onnxModelFile, const char* quantParamFile, int verbosity) = 0;
    //!
    //! \brief Parse serialized onnx model and quantization params to build IxRT network
    //!
    //! \param serialized_onnx_model pointer to onnx model data memory
    //! \param serialized_onnx_model_size byte size of the serialized_onnx_model
    //! \param quantParamFile c string representing quantization param path, must be ppq style
    //! \param model_path if specified, load weights from it, note is has to be absolute path
    //!
    //! \see parseFromFiles(const char* onnxModelFile, const char* quantParamFile, int verbosity)
    //!
    virtual bool parseFromFiles(void const* serialized_onnx_model, size_t serialized_onnx_model_size,
                                const char* quantFile, const char* model_path = nullptr) = 0;

    //!
    //!\brief Check whether a ONNX model is supported.
    //!
    //! \param serialized_onnx_model Pointer to the serialized ONNX model
    //! \param serialized_onnx_model_size Size of the serialized ONNX model in bytes
    //! \param sub_graph_collection Container to hold supported subgraphs
    //! \param model_path Absolute path to the model file for loading external weights if required
    //! \return true if the model is supported
    //!
    virtual bool supportsModel(void const* serialized_onnx_model, size_t serialized_onnx_model_size,
                               SubGraphCollection_t& sub_graph_collection, const char* model_path = nullptr) = 0;

    virtual ~IParser() noexcept = default;
};

//!
//! \brief Create IxRT onnx parser
//!
//! \param network INetworkDefinition
//! \param logger logger
//!
//! \return IParser*
//!
inline IParser* createParser(nvinfer1::INetworkDefinition& network, nvinfer1::ILogger& logger) {
    return static_cast<IParser*>(createOnnxParser_INTERNAL(&network, &logger, 0));
}
}  // namespace nvonnxparser
