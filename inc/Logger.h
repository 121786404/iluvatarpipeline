// Logger.h
#ifndef LOGGER_H
#define LOGGER_H

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include "spdlog/sinks/stdout_color_sinks.h"
#include <memory>

extern std::shared_ptr<spdlog::logger> logger;

void initializeLogger();

#endif // LOGGER_H
