//===----------------------------------------------------------------------===//
//
//   Copyright © 2022 Iluvatar CoreX. All rights reserved.
//   Copyright Declaration: This software, including all of its code and
//   documentation, except for the third-party software it contains, is a
//   copyrighted work of Shanghai Iluvatar CoreX Semiconductor Co., Ltd. and
//   its affiliates (“Iluvatar CoreX”) in accordance with the PRC Copyright Law
//   and relevant international treaties, and all rights contained therein are
//   enjoyed by Iluvatar CoreX. No user of this software shall have any right,
//   ownership or interest in this software and any use of this software shall
//   be in compliance with the terms and conditions of the End User License
//   Agreement.
//
//===----------------------------------------------------------------------===//

#include "IluvatarJpegCodec.h"
using namespace std;

static auto ThrowOnCudaError = [](CUresult res, int lineNum = -1) {
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

struct IxJpegImpl
{
    int  id             = 0;
    bool decoder_enable = true, encoder_enable = true;

    int    maxBatch       = 0;
    size_t maxBatchBuffer = 0;
    void*  BatchYuvBuff   = NULL;

    CUcontext    m_cuContext  = nullptr;
    cudaStream_t m_cudaStream = nullptr;

    nvjpegEncoderParams_t encode_params;
    nvjpegHandle_t        nvjpeg_handle = nullptr;
    nvjpegJpegState_t     jpeg_state_enc;
    nvjpegJpegState_t     jpeg_state_dec;
    nvjpegEncoderState_t  encoder_state;
};

int dev_malloc(void** p, size_t s)
{
    return (int)cudaMalloc(p, s);
}
int dev_free(void* p)
{
    return (int)cudaFree(p);
}

int IluvatarJpegCodec::GetEnableDecoder()
{
    return p_impl->decoder_enable;
};

int IluvatarJpegCodec::GetEnableEncoder()
{
    return p_impl->encoder_enable;
};

int IluvatarJpegCodec::ReadRawData(const std::string& img_name, std::vector<unsigned char>& raw_data)
{
    std::ifstream oInputStream(img_name.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
    if (!(oInputStream.is_open()))
    {
        logger->error("[{} {}]: Cannot open image: {}", __FUNCTION__, __LINE__, img_name);
        return -1;
    }
    std::streamsize nSize = oInputStream.tellg();
    oInputStream.seekg(0, std::ios::beg);

    if (!raw_data.empty())
    {
        raw_data.clear();
    }
    raw_data.resize(nSize);
    if (!oInputStream.read((char*)(raw_data.data()), nSize))
    {
        logger->error("[{} {}]: Read Image Fail", __FUNCTION__, __LINE__);
        return -2;
    }

    return 0;
}

static int SupportSubSampleing(nvjpegChromaSubsampling_t subsampling)
{
    switch (subsampling)
    {
        case NVJPEG_CSS_444:
            // std::cerr << "YUV 4:4:4 chroma subsampling" << std::endl;
            break;
        case NVJPEG_CSS_440:
            // std::cerr << "YUV 4:4:0 chroma subsampling" << std::endl;
            break;
        case NVJPEG_CSS_422:
            // std::cerr << "YUV 4:2:2 chroma subsampling" << std::endl;
            break;
        case NVJPEG_CSS_420:
            // std::cerr << "YUV 4:2:0 chroma subsampling" << std::endl;
            break;
        case NVJPEG_CSS_411:
            // std::cerr << "YUV 4:1:1 chroma subsampling" << std::endl;
            break;
        case NVJPEG_CSS_410:
            // std::cerr << "YUV 4:1:0 chroma subsampling" << std::endl;
            break;
        case NVJPEG_CSS_GRAY:
            // std::cerr << "Grayscale JPEG" << std::endl;
            break;
        case NVJPEG_CSS_UNKNOWN:
            logger->error("[{} {}]: Unknown chroma subsampling", __FUNCTION__, __LINE__);
            return -4;
    }
    return 0;
}

static bool is_interleaved(nvjpegOutputFormat_t format)
{
    if (format == NVJPEG_OUTPUT_RGBI || format == NVJPEG_OUTPUT_BGRI)
        return true;
    else
        return false;
}

static Pixel_Format SwitchPixelFormat(nvjpegOutputFormat_t output_format)
{
    Pixel_Format pix_format = UNDEFINED;
    switch (output_format)
    {
        case NVJPEG_OUTPUT_YUV: pix_format = YUV420; break;
        case NVJPEG_OUTPUT_RGB: pix_format = RGB_PLANAR; break;
        case NVJPEG_OUTPUT_BGR: pix_format = BGR_PLANAR; break;
        case NVJPEG_OUTPUT_RGBI: pix_format = RGB; break;
        case NVJPEG_OUTPUT_BGRI: pix_format = BGR; break;
        default: break;
    }
    return pix_format;
}

static nvjpegImage_t TransferData2NvjpegImages(unsigned char*             pBuffer,
                                               size_t                     width,
                                               size_t                     height,
                                               const nvjpegOutputFormat_t output_format)
{
    nvjpegImage_t imgdesc{};
    if (output_format == NVJPEG_OUTPUT_YUV)
    {
        imgdesc = {
            {pBuffer, pBuffer + width * height, pBuffer + width * height * 5 / 4, pBuffer + width * height * 3 / 2},
            {(unsigned int)(is_interleaved(output_format) ? width * 3 : width),
             (unsigned int)width / 2,
             (unsigned int)width / 2,
             (unsigned int)width / 2}};
    }
    else
    {
        imgdesc = {{pBuffer, pBuffer + width * height, pBuffer + width * height * 2, pBuffer + width * height * 3},
                   {(unsigned int)(is_interleaved(output_format) ? width * 3 : width),
                    (unsigned int)width,
                    (unsigned int)width,
                    (unsigned int)width}};
    }
    return imgdesc;
}

int IluvatarJpegCodec::DecodeSurface(std::vector<unsigned char>& raw_data,
                                     SurfaceCudaBuff&            output,
                                     const nvjpegOutputFormat_t  output_format) noexcept
{
    try
    {
        CudaCtxPush ctxPush(p_impl->m_cuContext);
        if (!p_impl->decoder_enable)
            throw runtime_error("JPEG handle not enable decoder !!!");

        unsigned char* dpImage = (unsigned char*)(raw_data.data());
        size_t         nSize   = raw_data.size();

        // Retrieve the componenet and size info.
        int                       nComponent = 0;
        nvjpegChromaSubsampling_t subsampling;
        int                       widths[NVJPEG_MAX_COMPONENT];
        int                       heights[NVJPEG_MAX_COMPONENT];
        if (NVJPEG_STATUS_SUCCESS !=
            nvjpegGetImageInfo(p_impl->nvjpeg_handle, dpImage, nSize, &nComponent, &subsampling, widths, heights))
        {
            std::ostringstream errorString;
            errorString << std::endl << "Error decoding JPEG header" << endl;
            throw runtime_error(errorString.str());
        }

        if (0 != SupportSubSampleing(subsampling))
        {
            throw runtime_error("Unknown chroma subsampling");
        }

        output = SurfaceCudaBuff(widths[0], heights[0], SwitchPixelFormat(output_format), p_impl->m_cuContext);
        unsigned char* pBuffer = (unsigned char*)(output.GetGpuMem());
        nvjpegImage_t  imgdesc = TransferData2NvjpegImages(pBuffer, widths[0], heights[0], output_format);

        int nReturnCode =
            nvjpegDecode(p_impl->nvjpeg_handle, p_impl->jpeg_state_dec, dpImage, nSize, output_format, &imgdesc, NULL);
        if (nReturnCode != 0)
        {
            throw runtime_error("Error in nvjpegDecode.");
        }
        checkCudaErrors(cudaDeviceSynchronize());
    }
    catch (exception& e)
    {
        cerr << e.what();
    }
    return 0;
}

template <typename T>
int encodesurface(IxJpegImpl*                 p_impl,
                  T&                          input,
                  std::vector<unsigned char>& output,
                  const nvjpegOutputFormat_t  output_format)
{
    try
    {
        CudaCtxPush ctxPush(p_impl->m_cuContext);
        if (!p_impl->encoder_enable)
            throw runtime_error("JPEG handle not enable encoder !!!");

        if (input.Empty())
            throw runtime_error("Input surface is empty !!!");

        if (input.GetPixelFormat() != YUV420)
            throw runtime_error("Encode only support input format YUV420 !!!");

        size_t width  = input.GetWidth();
        size_t height = input.GetHeight();

        unsigned char* pBuffer = (unsigned char*)(input.GetGpuMem());
        nvjpegImage_t  imgdesc = TransferData2NvjpegImages(pBuffer, width, height, output_format);

        if (NVJPEG_OUTPUT_YUV == output_format)
        {
            checkCudaErrors(nvjpegEncodeYUV(p_impl->nvjpeg_handle,
                                            p_impl->encoder_state,
                                            p_impl->encode_params,
                                            &imgdesc,
                                            NVJPEG_CSS_420,
                                            width,
                                            height,
                                            p_impl->m_cudaStream));
        }
        else
            throw runtime_error("Encode only support output format YUV420 !!!");

        size_t length;
        checkCudaErrors(nvjpegEncodeRetrieveBitstream(
            p_impl->nvjpeg_handle, p_impl->encoder_state, NULL, &length, p_impl->m_cudaStream));

        output.resize(length);
        checkCudaErrors(nvjpegEncodeRetrieveBitstream(
            p_impl->nvjpeg_handle, p_impl->encoder_state, output.data(), &length, p_impl->m_cudaStream));
        if (p_impl->m_cudaStream != nullptr)
            checkCudaErrors(cudaStreamSynchronize(p_impl->m_cudaStream));
        else
            checkCudaErrors(cudaDeviceSynchronize());
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }

    return 0;
}

int IluvatarJpegCodec::EncodeSurface(SurfaceCudaBuff&            input,
                                     std::vector<unsigned char>& output,
                                     const nvjpegOutputFormat_t  output_format) noexcept
{
    return encodesurface(p_impl, input, output, output_format);
}

int IluvatarJpegCodec::EncodeSurface(ViDecSurfaceCudaBuff&       input,
                                     std::vector<unsigned char>& output,
                                     const nvjpegOutputFormat_t  output_format) noexcept
{
    return encodesurface(p_impl, input, output, output_format);
}

template <typename T>
static bool GetEncodeParams(std::vector<T>&            input,
                            void**                     devBuff,
                            size_t                     maxBatchBufSize,
                            std::vector<jpegEncParam>& cfgs,
                            cudaStream_t               stream)
{
    int batches = input.size();
    cfgs.resize(batches);

    size_t total_size = 0;
    for (size_t i = 0; i < batches; i++)
    {
        if (input[i].Empty())
        {
            logger->error("[{} {}]: input is empty", __FUNCTION__, __LINE__);
            return false;
        }
        if (input[i].GetPixelFormat() != YUV420)
        {
            logger->error("[{} {}]: input is not yuv420", __FUNCTION__, __LINE__);
            return false;
        }

        total_size += input[i].Total();
    }

    if (total_size > maxBatchBufSize)
    {
        logger->error(
            "[{} {}]: nput size buffer {} > jpeg pre buffer {}", __FUNCTION__, __LINE__, total_size, maxBatchBufSize);
        return false;
    }
    void* devYuvBuff = *devBuff;

    for (unsigned int i = 0; i < batches; i++)
    {
        size_t      yuv_size = input[i].Total();
        CUdeviceptr yuvBuff  = input[i].GetGpuMem();

        cfgs[i].width        = input[i].GetWidth();
        cfgs[i].height       = input[i].GetHeight();
        cfgs[i].frame_rate   = 30;
        cfgs[i].frame_format = 0;
        cfgs[i].image_format = NVJPEG_CSS_420;
        cfgs[i].buffer_size  = yuv_size;

        checkCudaErrors(cudaMemcpyAsync(
            (void*)devYuvBuff, reinterpret_cast<void*>(yuvBuff), yuv_size, cudaMemcpyDeviceToDevice, stream));
        checkCudaErrors(cudaStreamSynchronize(stream));
        devYuvBuff += yuv_size;
        // printf("width:%d \n", cfgs[i].width);
        // printf("height:%d \n", cfgs[i].height);
        // printf("buffer_size:%d \n", cfgs[i].buffer_size);
        // printf("frame_format:%d \n", cfgs[i].frame_format);
        // printf("image_format:%d \n", cfgs[i].image_format);
    }

    return true;
}

template <typename T>
int encodesurface_batch(IxJpegImpl*                 p_impl,
                        std::vector<T>&             inputs,
                        std::vector<unsigned char>& outputs,
                        std::vector<size_t>&        lengthes,
                        const nvjpegOutputFormat_t  output_format)
{
    try
    {
        CudaCtxPush ctxPush(p_impl->m_cuContext);
        if (!p_impl->encoder_enable)
            throw runtime_error("JPEG handle not enable encoder !!!");

        if (inputs.size() <= 0)
            throw runtime_error("Input surface is empty !!!");

        for (size_t i = 0; i < inputs.size(); i++)
        {
            if (inputs[i].Empty())
            {
                ostringstream errorString;
                errorString << endl << __FUNCTION__ << " " << __LINE__ << ": input [" << i << "]: is Empty !!!" << endl;
                throw runtime_error(errorString.str());
            }
        }

        int batch = inputs.size();
        if (batch > p_impl->maxBatch)
            throw runtime_error("Input surface batch > jpeg MaxBatch !!!");

        std::vector<jpegEncParam> cfgs;
        if (!GetEncodeParams(inputs, &(p_impl->BatchYuvBuff), (p_impl->maxBatchBuffer), cfgs, p_impl->m_cudaStream))
        {
            throw runtime_error("Encode Params prepare fail !!!");
        }

        checkCudaErrors((CUresult)cujpegEncodeYUVBatched(p_impl->nvjpeg_handle,
                                                         p_impl->encoder_state,
                                                         (const unsigned char*)p_impl->BatchYuvBuff,
                                                         (const jpegEncParam*)(cfgs.data()),
                                                         batch,
                                                         (CUjpegstream)p_impl->m_cudaStream));

        lengthes.resize(batch);
        checkCudaErrors((CUresult)cujpegEncodeRetrieveBitstreamBatched(p_impl->nvjpeg_handle,
                                                                       p_impl->encoder_state,
                                                                       NULL,
                                                                       lengthes.data(),
                                                                       batch,
                                                                       (CUjpegstream)p_impl->m_cudaStream));

        size_t totalJpegSize = 0;
        for (unsigned int k = 0; k < batch; k++)
        {
            // printf("length[%d]:0x%lx\n",k,length[k]);
            totalJpegSize += lengthes[k];
        }

        outputs.resize(totalJpegSize);
        checkCudaErrors((CUresult)cujpegEncodeRetrieveBitstreamBatched(p_impl->nvjpeg_handle,
                                                                       p_impl->encoder_state,
                                                                       outputs.data(),
                                                                       NULL,
                                                                       batch,
                                                                       (CUjpegstream)p_impl->m_cudaStream));
        if (p_impl->m_cudaStream != nullptr)
            checkCudaErrors(cudaStreamSynchronize(p_impl->m_cudaStream));
        else
            checkCudaErrors(cudaDeviceSynchronize());
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }

    return 0;
}

int IluvatarJpegCodec::EncodeSurfaceBatch(std::vector<SurfaceCudaBuff>& input,
                                          std::vector<unsigned char>&   output,
                                          std::vector<size_t>&          lengthes,
                                          const nvjpegOutputFormat_t    output_format) noexcept
{
    return encodesurface_batch(p_impl, input, output, lengthes, output_format);
}

int IluvatarJpegCodec::EncodeSurfaceBatch(std::vector<ViDecSurfaceCudaBuff>& input,
                                          std::vector<unsigned char>&        output,
                                          std::vector<size_t>&               lengthes,
                                          const nvjpegOutputFormat_t         output_format) noexcept
{
    return encodesurface_batch(p_impl, input, output, lengthes, output_format);
}

int IluvatarJpegCodec::SaveRawData(const std::string& img_name, std::vector<unsigned char>& raw_data)
{
    if (FolderExist(img_name) < 0)
        return -1;

    std::ofstream outputFile(img_name.c_str(), std::ios::out | std::ios::binary);
    outputFile.write(reinterpret_cast<const char*>(raw_data.data()), static_cast<int>(raw_data.size()));
    outputFile.close();
    return 0;
}

IluvatarJpegCodec::IluvatarJpegCodec(int dev, int maxBatch, bool decoder_enable, bool encoder_enable)
{
    p_impl           = new IxJpegImpl();
    p_impl->id       = dev;
    p_impl->maxBatch = maxBatch;

    p_impl->decoder_enable = decoder_enable;
    p_impl->encoder_enable = encoder_enable;

    cudaDeviceProp props;
    checkCudaErrors(cudaGetDeviceProperties(&props, dev));
    logger->info("[{} {}]: Using GPU {} ({}, {} SMs, {} th/SM max, CC {}.{}, ECC {}",
                 __FUNCTION__,
                 __LINE__,
                 dev,
                 props.name,
                 props.multiProcessorCount,
                 props.maxThreadsPerMultiProcessor,
                 props.major,
                 props.minor,
                 props.ECCEnabled ? "on" : "off");

    // checkCudaErrors(cudaSetDevice(dev));
    checkCudaErrors(cuCtxGetCurrent(&(p_impl->m_cuContext)));
    checkCudaErrors(cudaStreamCreate(&(p_impl->m_cudaStream)));

    nvjpegDevAllocator_t dev_allocator = {&dev_malloc, &dev_free};

    checkCudaErrors(nvjpegCreate(NVJPEG_BACKEND_DEFAULT, &dev_allocator, &(p_impl->nvjpeg_handle)));
    if (decoder_enable)
        checkCudaErrors(nvjpegJpegStateCreate(p_impl->nvjpeg_handle, &(p_impl->jpeg_state_dec)));

    if (encoder_enable)
    {   //3840×2160
        p_impl->maxBatchBuffer = 3840 * 2160 * 3 / 2 * sizeof(uint8_t) * maxBatch;
        checkCudaErrors(cudaMalloc((void**)&(p_impl->BatchYuvBuff), p_impl->maxBatchBuffer));

        // checkCudaErrors(nvjpegJpegStateCreate(p_impl->nvjpeg_handle, &(p_impl->jpeg_state_enc)));
        checkCudaErrors(nvjpegEncoderStateCreate(p_impl->nvjpeg_handle, &(p_impl->encoder_state), NULL));
        checkCudaErrors(nvjpegEncoderParamsCreate(p_impl->nvjpeg_handle, &(p_impl->encode_params), NULL));

        // sample input parameters
        int                       quality         = 70;
        int                       huf             = 0;
        nvjpegChromaSubsampling_t SamplingFactors = NVJPEG_CSS_420;
        checkCudaErrors(nvjpegEncoderParamsSetQuality(p_impl->encode_params, quality, NULL));
        checkCudaErrors(nvjpegEncoderParamsSetOptimizedHuffman(p_impl->encode_params, huf, NULL));
        checkCudaErrors(nvjpegEncoderParamsSetSamplingFactors(p_impl->encode_params, SamplingFactors, NULL));
    }
}

IluvatarJpegCodec::~IluvatarJpegCodec()
{
    CudaCtxPush ctxPush(p_impl->m_cuContext);
    if (p_impl->encoder_enable)
    {
        checkCudaErrors(nvjpegEncoderParamsDestroy(p_impl->encode_params));
        checkCudaErrors(nvjpegEncoderStateDestroy(p_impl->encoder_state));
        // checkCudaErrors(nvjpegJpegStateDestroy(p_impl->jpeg_state_enc));
    }

    if (p_impl->decoder_enable)
        checkCudaErrors(nvjpegJpegStateDestroy(p_impl->jpeg_state_dec));

    checkCudaErrors(nvjpegDestroy(p_impl->nvjpeg_handle));
    if (p_impl->m_cudaStream != nullptr)
        checkCudaErrors(cudaStreamDestroy(p_impl->m_cudaStream));

    if (p_impl->BatchYuvBuff)
    {
        checkCudaErrors(cudaFree(p_impl->BatchYuvBuff));
        p_impl->BatchYuvBuff = NULL;
    }

    delete p_impl;
    p_impl = nullptr;
}