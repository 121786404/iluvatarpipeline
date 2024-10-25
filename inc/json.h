#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "nlohmann/json.hpp"
using json = nlohmann::json;

struct ModelParams
{
    int                      resize_h       = 0;
    int                      resize_w       = 0;
    int                      pre_maxBatch   = 0;
    int                      pre_qsz        = 0;
    int                      infer_maxBatch = 0;
    int                      infer_qsz      = 0;
    std::vector<std::string> custom_outputs;
    std::string              onnx_file;
    std::string              engine_file;
};

struct RtspUrlParams
{
    std::string rtsp_url;
    int         rate;
};

namespace nlohmann
{
template <>
struct adl_serializer<ModelParams>
{
    static void from_json(const json& j, ModelParams& params)
    {
        j.at("resize_h").get_to(params.resize_h);
        j.at("resize_w").get_to(params.resize_w);
        j.at("pre_maxBatch").get_to(params.pre_maxBatch);
        j.at("pre_qsz").get_to(params.pre_qsz);
        j.at("infer_maxBatch").get_to(params.infer_maxBatch);
        j.at("infer_qsz").get_to(params.infer_qsz);
        j.at("custom_outputs").get_to(params.custom_outputs);
        j.at("onnx_file").get_to(params.onnx_file);
        j.at("engine_file").get_to(params.engine_file);
    }
};

template <>
struct adl_serializer<RtspUrlParams>
{
    static void to_json(json& j, const RtspUrlParams& params)
    {
        j = json{{"rtsp_url", params.rtsp_url}, {"rate", params.rate}};
    }

    static void from_json(const json& j, RtspUrlParams& params)
    {
        j.at("rtsp_url").get_to(params.rtsp_url);
        j.at("rate").get_to(params.rate);
    }
};
}  // namespace nlohmann

void from_json(const json& j, std::vector<std::string>& vec);

class RtspUrlManager
{
public:
    RtspUrlManager(const std::string& file_path) { loadFromFile(file_path); }

    const std::vector<RtspUrlParams>& getUrls() const { return rtsp_params; }

private:
    std::vector<RtspUrlParams> rtsp_params;

    void loadFromFile(const std::string& file_path)
    {
        std::ifstream file(file_path);
        if (file.is_open())
        {
            json j;
            file >> j;
            j.at("rtsp_params").get_to(rtsp_params);
        }
    }
};