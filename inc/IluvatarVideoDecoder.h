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
#pragma once
#include "FFmpegDemuxer.h"
#include "memoryInterface.h"
#include "util.h"

typedef enum
{
    BS_MODE_INTERRUPT,
    BS_MODE_RESERVED,
    BS_MODE_PIC_END,
} BitStreamMode;

struct DecodedFrameContext
{
    CUdeviceptr mem;
    uint64_t    pts;
    uint64_t    bsl;
    size_t      size;
    //   PacketData out_pdata;

    // Set up this flag to feed decoder with empty input without setting up EOS
    // flag;
    bool no_eos;

    DecodedFrameContext(CUdeviceptr new_ptr, uint64_t new_pts, size_t size)
        : mem(new_ptr)
        , pts(new_pts)
        , size(size)
        , no_eos(false)
    {
    }

    DecodedFrameContext(CUdeviceptr new_ptr, uint64_t new_pts, size_t size, bool new_no_eos)
        : mem(new_ptr)
        , pts(new_pts)
        , size(size)
        , no_eos(new_no_eos)
    {
    }

    DecodedFrameContext()
        : mem(0U)
        , pts(0U)
        , size(0U)
        , no_eos(false)
    {
    }
};

class decoder_error : public std::runtime_error
{
public:
    decoder_error(const char* str)
        : std::runtime_error(str)
    {
    }
};

class cuvid_parser_error : public std::runtime_error
{
public:
    cuvid_parser_error(const char* str)
        : std::runtime_error(str)
    {
    }
};

namespace IX
{
class IluvatarVideoDecoder
{
public:
    IluvatarVideoDecoder()                                  = delete;
    IluvatarVideoDecoder(const IluvatarVideoDecoder& other) = delete;
    IluvatarVideoDecoder& operator=(const IluvatarVideoDecoder& other) = delete;

    IluvatarVideoDecoder(CUcontext cuContext, int eCodec, CUstream cuStream = nullptr, int id = -1);

    ~IluvatarVideoDecoder();

    int GetWidth();

    int GetHeight();

    size_t GetFrameSize();

    size_t GetDecoderTotalFrames();

    size_t GetDecoderReceiveFrames();

    int DecodeSatus();

    void Init(FFmpegDemuxer* demuxer, ProcessQueue<ViDecSurfaceCudaBuff>* m_DecFramesCtxQueue, int rate = -1)
    {
        HandleVideoSource(demuxer, m_DecFramesCtxQueue, rate);
    }

    int GetCodec() const;

private:
    /* All the functions with Handle* prefix doesn't
     * throw as they are called from different thread;
     */

    int HandleVideoSource(FFmpegDemuxer*                      demuxer,
                          ProcessQueue<ViDecSurfaceCudaBuff>* m_DecFramesCtxQueue,
                          int                                 rate) noexcept;

    int StopVideoSource() noexcept;

    void SendBitStream(FFmpegDemuxer* demuxer) noexcept;

    void GetFrame2Queue(ProcessQueue<ViDecSurfaceCudaBuff>* m_DecFramesCtxQueue) noexcept;

    void sendEOF();

    //   int ReconfigureDecoder(CUVIDEOFORMAT* pVideoFormat);

    struct IxDecoderImpl* p_impl;
};
}  // namespace IX
