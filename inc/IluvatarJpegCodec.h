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
#include "ixjpeg.h"
#include "memoryInterface.h"
#include "util.h"

class IluvatarJpegCodec
{
public:
    IluvatarJpegCodec()                               = delete;
    IluvatarJpegCodec(const IluvatarJpegCodec& other) = delete;
    IluvatarJpegCodec& operator=(const IluvatarJpegCodec& other) = delete;

    IluvatarJpegCodec(int dev, int maxBatch = 1, bool decoder_enable = true, bool encoder_enable = true);

    ~IluvatarJpegCodec();

    int GetEnableDecoder();

    int GetEnableEncoder();

    int ReadRawData(const std::string& img_name, std::vector<unsigned char>& raw_data);

    int DecodeSurface(std::vector<unsigned char>& raw_data,
                      SurfaceCudaBuff&            output,
                      const nvjpegOutputFormat_t  output_format = NVJPEG_OUTPUT_BGRI) noexcept;

    int EncodeSurface(SurfaceCudaBuff&            input,
                      std::vector<unsigned char>& output,
                      const nvjpegOutputFormat_t  output_format = NVJPEG_OUTPUT_YUV) noexcept;

    int EncodeSurface(ViDecSurfaceCudaBuff&       input,
                      std::vector<unsigned char>& output,
                      const nvjpegOutputFormat_t  output_format = NVJPEG_OUTPUT_YUV) noexcept;

    int EncodeSurfaceBatch(std::vector<SurfaceCudaBuff>& input,
                           std::vector<unsigned char>&   output,
                           std::vector<size_t>&          lengthes,
                           const nvjpegOutputFormat_t    output_format = NVJPEG_OUTPUT_YUV) noexcept;

    int EncodeSurfaceBatch(std::vector<ViDecSurfaceCudaBuff>& input,
                           std::vector<unsigned char>&        output,
                           std::vector<size_t>&               lengthes,
                           const nvjpegOutputFormat_t         output_format = NVJPEG_OUTPUT_YUV) noexcept;

    int SaveRawData(const std::string& img_name, std::vector<unsigned char>& raw_data);

private:
    /* All the functions with Handle* prefix doesn't
     * throw as they are called from different thread;
     */

    struct IxJpegImpl* p_impl;
};
