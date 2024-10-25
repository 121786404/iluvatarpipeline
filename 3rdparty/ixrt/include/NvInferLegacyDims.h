#pragma once
#include "NvInferRuntimeCommon.h"

namespace nvinfer1 {

class Dims2 : public Dims {
   public:
    Dims2() : Dims{2, {}} {}

    Dims2(int32_t d0, int32_t d1) : Dims{2, {d0, d1}} {}
};

class DimsHW : public Dims2 {
   public:
    DimsHW() : Dims2() {}

    DimsHW(int32_t height, int32_t width) : Dims2(height, width) {}

    int32_t& h() { return d[0]; }

    int32_t h() const { return d[0]; }

    int32_t& w() { return d[1]; }

    int32_t w() const { return d[1]; }
};

class Dims3 : public Dims {
   public:
    Dims3() : Dims{3, {}} {}

    Dims3(int32_t d0, int32_t d1, int32_t d2) : Dims{3, {d0, d1, d2}} {}
};

class Dims4 : public Dims {
   public:
    Dims4() : Dims{4, {}} {}

    Dims4(int32_t d0, int32_t d1, int32_t d2, int32_t d3) : Dims{4, {d0, d1, d2, d3}} {}
};

}  // namespace nvinfer1
