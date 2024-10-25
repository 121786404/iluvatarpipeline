#include <cstdlib>  // for std::getenv
#include <filesystem>
#include "Logger.h"

std::shared_ptr<spdlog::logger> logger;

static std::string GetLocalTime(bool file = false)
{
    auto now = std::chrono::system_clock::now();

    std::time_t now_c = std::chrono::system_clock::to_time_t(now);

    struct tm parts;
    localtime_r(&now_c, &parts);

    char buffer[80];
    if (file)
        strftime(buffer, sizeof(buffer), "%Y-%m-%d/%H:%M:%S", &parts);
    else
        strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &parts);
    // 返回格式化的时间字符串
    return std::string(buffer);
}

spdlog::level::level_enum getLogLevelFromEnv()
{
    const char* log_level_env = std::getenv("LOG_LEVEL");
    if (!log_level_env)
    {
        return spdlog::level::info;  // 默认日志级别
    }

    std::string log_level_str(log_level_env);
    if (log_level_str == "trace")
        return spdlog::level::trace;
    if (log_level_str == "debug")
        return spdlog::level::debug;
    if (log_level_str == "info")
        return spdlog::level::info;
    if (log_level_str == "warn")
        return spdlog::level::warn;
    if (log_level_str == "error")
        return spdlog::level::err;
    if (log_level_str == "critical")
        return spdlog::level::critical;
    if (log_level_str == "off")
        return spdlog::level::off;

    return spdlog::level::info;  // 如果无效，则返回默认日志级别
}

void initializeLogger()
{
    std::string           foldtime = GetLocalTime(true);
    std::string           log_name = "../log/" + foldtime + ".log";
    std::filesystem::path log_dir  = std::filesystem::path("../log/");
    if (!std::filesystem::exists(log_dir))
    {
        // Create the directory
        std::filesystem::create_directories(log_dir);
    }

    logger         = spdlog::basic_logger_mt("iluvatarpipeline", log_name);
    auto log_level = getLogLevelFromEnv();
    logger->set_level(log_level);  // 根据环境变量设置日志级别
    logger->flush_on(log_level);

    // 可选：设置控制台日志输出
    auto console_logger = spdlog::stdout_color_mt("console");
    console_logger->set_level(log_level);
    console_logger->flush_on(log_level);

    // 将文件和控制台输出同步
    logger->sinks().push_back(console_logger->sinks()[0]);
}
