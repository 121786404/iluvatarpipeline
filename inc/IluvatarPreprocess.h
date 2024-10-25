#pragma once
#include "memoryInterface.h"
#include "util.h"

struct Rect
{
    int x;
    int y;
    int width;
    int height;
};

class DetectPreprocessor
{
public:
    DetectPreprocessor()                                 = delete;
    DetectPreprocessor(const DetectPreprocessor& other) = delete;
    DetectPreprocessor& operator=(DetectPreprocessor& other) = delete;

    DetectPreprocessor(int resize_h, int resize_w, CUcontext context, int maxInputs);
    ~DetectPreprocessor();

    SurfaceCudaBuff Process(std::vector<ViDecSurfaceCudaBuff>& inputImage);

private:
    struct PreProcessImpl* p_impl;
};

class ClassifyPreprocessor
{
public:
    ClassifyPreprocessor()                                 = delete;
    ClassifyPreprocessor(const ClassifyPreprocessor& other) = delete;
    ClassifyPreprocessor& operator=(ClassifyPreprocessor& other) = delete;

    ClassifyPreprocessor(int resize_h, int resize_w, CUcontext context, int maxInputs);
    ~ClassifyPreprocessor();

    SurfaceCudaBuff Process(std::vector<ViDecSurfaceCudaBuff>& inputImage, std::vector<std::vector<Rect>>& rects);

private:
    struct PreProcessImpl* p_impl;
};