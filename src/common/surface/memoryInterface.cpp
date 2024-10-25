/*
 * Copyright 2019 NVIDIA Corporation
 * Copyright 2021 Videonetics Technology Private Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *    http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "memoryInterface.h"

using namespace std;

static void ThrowOnCudaError(CUresult res, int lineNum = -1)
{
    if (CUDA_SUCCESS != res)
    {
        stringstream ss;

        if (lineNum > 0)
        {
            ss << __FILE__ << ":";
            ss << lineNum << endl;
        }

        const char* errName = nullptr;
        if (CUDA_SUCCESS != cuGetErrorName(res, &errName))
        {
            ss << "CUDA error with code " << res << endl;
        }
        else
        {
            ss << "CUDA error: " << errName << endl;
        }

        const char* errDesc = nullptr;
        cuGetErrorString(res, &errDesc);

        if (!errDesc)
        {
            ss << "No error string available" << endl;
        }
        else
        {
            ss << errDesc << endl;
        }

        throw runtime_error(ss.str());
    }
};

CudaBuffer* CudaBuffer::Make(size_t elemSize, size_t numElems, CUdeviceptr srcMem, CUcontext context, Buffer::Flag flag)
{
    return new CudaBuffer(elemSize, numElems, srcMem, context, flag);
}

CudaBuffer* CudaBuffer::Make(size_t elemSize, size_t numElems, CUcontext context, Buffer::Flag flag)
{
    return new CudaBuffer(elemSize, numElems, context, flag);
}

CudaBuffer* CudaBuffer::Make(const void*  ptr,
                             size_t       elemSize,
                             size_t       numElems,
                             CUcontext    context,
                             CUstream     str,
                             Buffer::Flag flag)
{
    return new CudaBuffer(ptr, elemSize, numElems, context, str, flag);
}

CudaBuffer* CudaBuffer::Clone()
{
    auto pCopy = CudaBuffer::Make(elem_size, num_elems, ctx, buffer_flag);

    if (CUDA_SUCCESS != cuMemcpyDtoD(pCopy->GpuMem(), GpuMem(), GetRawMemSize()))
    {
        delete pCopy;
        return nullptr;
    }

    return pCopy;
}

CudaBuffer::~CudaBuffer()
{
    Deallocate();
}

CudaBuffer::CudaBuffer(size_t elemSize, size_t numElems, CUcontext context, Buffer::Flag flag)
{
    elem_size   = elemSize;
    num_elems   = numElems;
    ctx         = context;
    buffer_flag = flag;

    if (!Allocate())
    {
        throw bad_alloc();
    }
}

CudaBuffer::CudaBuffer(size_t elemSize, size_t numElems, CUdeviceptr srcMem, CUcontext context, Buffer::Flag flag)
{
    elem_size   = elemSize;
    num_elems   = numElems;
    ctx         = context;
    gpuMem      = srcMem;
    buffer_flag = flag;
}

CudaBuffer::CudaBuffer(const void*  ptr,
                       size_t       elemSize,
                       size_t       numElems,
                       CUcontext    context,
                       CUstream     str,
                       Buffer::Flag flag)
{
    elem_size   = elemSize;
    num_elems   = numElems;
    ctx         = context;
    buffer_flag = flag;

    if (!Allocate())
    {
        throw bad_alloc();
    }

    CudaCtxPush lock(ctx);
    auto        res = cuMemcpyHtoDAsync(gpuMem, ptr, GetRawMemSize(), str);
    ThrowOnCudaError(res, __LINE__);

    res = cuStreamSynchronize(str);
    ThrowOnCudaError(res, __LINE__);
}

bool CudaBuffer::Allocate()
{
    if (GetRawMemSize())
    {
        CudaCtxPush lock(ctx);
        auto        res = cuMemAlloc(&gpuMem, GetRawMemSize());

        std::ostringstream oss;
        oss << "[" << __FUNCTION__ << " " << __LINE__ << "]:"
            << "cuMemAlloc:" << std::hex << gpuMem << ", size:" << GetRawMemSize() << std::dec;
        logger->debug("[{} {}]: {}", __FUNCTION__, __LINE__, oss.str());
        ThrowOnCudaError(res, __LINE__);

        if (0U != gpuMem)
        {
            return true;
        }
    }
    return false;
}

void CudaBuffer::Deallocate()
{
    CudaCtxPush        lock(ctx);
    std::ostringstream oss;
    oss << "[" << __FUNCTION__ << " " << __LINE__ << "]:"
        << "cuMemFree:" << std::hex << gpuMem << ", size:" << GetRawMemSize() << std::dec;
    logger->debug("[{} {}]: {}", __FUNCTION__, __LINE__, oss.str());
    ThrowOnCudaError(cuMemFree(gpuMem), __LINE__);
    gpuMem = 0U;
}

static size_t CalculatePixelSize(Pixel_Format format, size_t width, size_t height, size_t batch)
{
    size_t numElems = 0;
    if (format == YUV420)
    {
        numElems = width * height * 3 / 2 * batch;
    }
    else if (format == RGB || format == RGB_PLANAR || format == BGR || format == BGR_PLANAR || format == RGB_32F ||
             format == RGB_32F_PLANAR)
    {
        numElems = width * height * 3 * batch;
    }
    return numElems;
}

// SurfaceCudaBuff
SurfaceCudaBuff::SurfaceCudaBuff() {}

SurfaceCudaBuff::SurfaceCudaBuff(const SurfaceCudaBuff& other)
    : batch(other.batch)
    , width(other.width)
    , height(other.height)
    , elemSize(other.elemSize)
    , numElems(other.numElems)
    , pix_format(other.pix_format)
    , buffer_flag(other.buffer_flag)
    , surfacecudabuffer(other.surfacecudabuffer)  // 直接复制 shared_ptr
{
}

SurfaceCudaBuff& SurfaceCudaBuff::operator=(const SurfaceCudaBuff& other)
{
    if (this != &other)
    {  // 防止自我赋值
        batch             = other.batch;
        width             = other.width;
        height            = other.height;
        elemSize          = other.elemSize;
        numElems          = other.numElems;
        pix_format        = other.pix_format;
        buffer_flag       = other.buffer_flag;
        surfacecudabuffer = other.surfacecudabuffer;  // 直接复制 shared_ptr
    }

    return *this;
}

SurfaceCudaBuff::SurfaceCudaBuff(size_t       width,
                                 size_t       height,
                                 Pixel_Format pix_format,
                                 CUcontext    context,
                                 size_t       batch,
                                 Buffer::Flag flag)
    : batch(batch)
    , width(width)
    , height(height)
    , pix_format(pix_format)
    , buffer_flag(flag)
{
    numElems = CalculatePixelSize(pix_format, width, height, batch);
    if (numElems <= 0)
    {
        std::ostringstream errorString;
        errorString << std::endl
                    << "Failed to create SurfaceCudaBuff." << std::endl
                    << "width :" << width << " height :" << height << " batch :" << batch << " numElems :" << numElems
                    << std::endl
                    << "Invalid buffer size or pix_format !!!" << std::endl;
        throw std::runtime_error(errorString.str());
    }
    elemSize          = (pix_format == RGB_32F_PLANAR || pix_format == RGB_32F) ? sizeof(float) : sizeof(uint8_t);
    surfacecudabuffer = std::shared_ptr<CudaBuffer>(
        CudaBuffer::Make(elemSize, numElems, context, buffer_flag));  // 使用 shared_ptr 管理缓冲区
}

SurfaceCudaBuff::~SurfaceCudaBuff() {}