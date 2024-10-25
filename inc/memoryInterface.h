#pragma once
#include "util.h"

namespace Buffer
{
enum class Flag
{

    UNDEFINED                = 0,
    VIDEO_FLAG               = 1,
    PPYOLOE_CROP_RESIZE_FLAG = 2,
    PPLCNET_CROP_RESIZE_FLAG = 3,
};
}

class CudaBuffer
{
public:
    CudaBuffer()                        = delete;
    CudaBuffer(const CudaBuffer& other) = delete;
    CudaBuffer& operator=(CudaBuffer& other) = delete;

    static CudaBuffer* Make(size_t elemSize, size_t numElems, CUcontext context, Buffer::Flag flag);
    static CudaBuffer* Make(size_t elemSize, size_t numElems, CUdeviceptr srcMem, CUcontext context, Buffer::Flag flag);
    static CudaBuffer* Make(const void*  ptr,
                            size_t       elemSize,
                            size_t       numElems,
                            CUcontext    context,
                            CUstream     str,
                            Buffer::Flag flag);
    CudaBuffer*        Clone();

    size_t      GetRawMemSize() const { return elem_size * num_elems; }
    size_t      GetNumElems() const { return num_elems; }
    size_t      GetElemSize() const { return elem_size; }
    CUdeviceptr GpuMem() { return gpuMem; }
    ~CudaBuffer();

private:
    CudaBuffer(size_t elemSize, size_t numElems, CUcontext context, Buffer::Flag flag);
    CudaBuffer(const void* ptr, size_t elemSize, size_t numElems, CUcontext context, CUstream str, Buffer::Flag flag);
    CudaBuffer(size_t elemSize, size_t numElems, CUdeviceptr srcMem, CUcontext context, Buffer::Flag flag);

    bool Allocate();
    void Deallocate();

    CUdeviceptr  gpuMem      = 0UL;
    CUcontext    ctx         = nullptr;
    size_t       elem_size   = 0U;
    size_t       num_elems   = 0U;
    Buffer::Flag buffer_flag = Buffer::Flag::UNDEFINED;
};

class SurfaceCudaBuff
{
public:
    ~SurfaceCudaBuff();

    SurfaceCudaBuff();
    SurfaceCudaBuff(const SurfaceCudaBuff& other);
    SurfaceCudaBuff& operator=(const SurfaceCudaBuff& other);

    // SurfaceCudaBuff(size_t width, size_t height, Pixel_Format pix_format, CUcontext context);
    SurfaceCudaBuff(size_t       width,
                    size_t       height,
                    Pixel_Format pix_format,
                    CUcontext    context,
                    size_t       batch = 1,
                    Buffer::Flag flag  = Buffer::Flag::UNDEFINED);
    // SurfaceCudaBuff(size_t width, size_t height, Pixel_Format pix_format, CUdeviceptr srcMem, CUcontext context);
    SurfaceCudaBuff(size_t       width,
                    size_t       height,
                    Pixel_Format pix_format,
                    CUdeviceptr  srcMem,
                    CUcontext    context,
                    size_t       batch = 1,
                    Buffer::Flag flag  = Buffer::Flag::UNDEFINED);

    size_t       Total() const { return elemSize * numElems; }
    size_t       GetBatch() const { return batch; }
    size_t       GetWidth() const { return width; }
    size_t       GetHeight() const { return height; }
    Pixel_Format GetPixelFormat() const { return pix_format; }

    CUdeviceptr GetGpuMem()
    {
        if (this->Empty())
        {
            std::ostringstream errorString;
            errorString << std::endl << __FUNCTION__ << " " << __LINE__ << ": input is Empty !!!" << std::endl;
            throw std::runtime_error(errorString.str());
        }
        return surfacecudabuffer->GpuMem();
    }

    bool Empty() { return nullptr == surfacecudabuffer; }

private:
    size_t       batch       = 0U;
    size_t       width       = 0U;
    size_t       height      = 0U;
    size_t       elemSize    = 0U;
    size_t       numElems    = 0U;
    Buffer::Flag buffer_flag = Buffer::Flag::UNDEFINED;
    Pixel_Format pix_format  = UNDEFINED;

    std::shared_ptr<CudaBuffer> surfacecudabuffer = nullptr;
};

class ViDecSurfaceCudaBuff : public SurfaceCudaBuff
{
public:
    ViDecSurfaceCudaBuff()
        : SurfaceCudaBuff()
    {
    }
    ViDecSurfaceCudaBuff(size_t             width,
                         size_t             height,
                         Pixel_Format       pix_format,
                         CUcontext          context,
                         size_t             batch     = 1,
                         const std::string& rtspInfo  = "",
                         const size_t       timeStamp = 0)
        : SurfaceCudaBuff(width, height, pix_format, context, batch, Buffer::Flag::VIDEO_FLAG)
        , rtspInfo(rtspInfo)
        , timeStamp(timeStamp)
    {
    }
    std::string GetRtspInfo() const { return rtspInfo; }
    size_t      GetTimeStamp() const { return timeStamp; }

private:
    std::string rtspInfo;
    size_t      timeStamp;
};
