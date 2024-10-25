#pragma once
#ifndef NV_INFER_PLUGIN_H
#define NV_INFER_PLUGIN_H

#include "NvInfer.h"
//!
//! \file NvInferPlugin.h
//!
//! This is the API for the plugins.
//!

extern "C" {
//!
//! \brief Initialize and register all the existing plugins to the Plugin Registry
//! This function should be called once before accessing the Plugin Registry.
//! \param logger Logger pointer to print plugin registration information
//! \param libNamespace Namespace that plugins need in this library
//!
bool initLibNvInferPlugins(void* logger, char const* libNamespace);

}  // extern "C"

#endif  // NV_INFER_PLUGIN_H
