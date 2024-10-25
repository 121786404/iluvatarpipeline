#include <cuda_runtime.h>

#define NORMALIZE_SCALE_IS_STDDEV (1 << 0)
namespace iluvatar
{
namespace cropResize
{
/*
YUV420 planar:
    YU12, I420, IYUV : YYYYYYYYUUVV
    YV12             : YYYYYYYYVVUU
    NV12             ：YYYYYYYYUVUV
    NV21             ：YYYYYYYYVUVU

YUV packed:
    YUV 444 Packed   : YUVYUVYUV
*/
enum CropResizeColorType
{
    CROP_RESIZE_IYUV = 0,
};

enum CropResizeInterType
{
    INTERP_LINEAR = 0,
};

typedef struct
{
    unsigned x;
    unsigned y;
    unsigned width;
    unsigned height;
} Rect;

typedef struct
{
    unsigned imageIdx;
    Rect     rect;
} RectA;

void CropResize(uint8_t**    input,
                uint8_t*     output,
                int2*        in_shape,
                unsigned     out_width,
                unsigned     out_height,
                RectA*       cropRect,
                unsigned     rectNum,
                unsigned     crop_imageFormat,
                unsigned     resize_interType,
                cudaStream_t stream);
}  // namespace cropResize

namespace CvtNormReformat
{
enum CvtColorReformatType
{
    CVTCOLOR_REFORMAT_IYUV2RGBf32p = 0,
    CVTCOLOR_REFORMAT_IYUV2BGRf32p = 1,
};

void CvtNormReformat(uint8_t*     input,
                     float*       output,
                     unsigned     cvtcolor_reformat_type,
                     unsigned     batches,
                     unsigned     width,
                     unsigned     height,
                     float        alpha,
                     float        beta,
                     bool         norm_flag,
                     float*       base_data,
                     int3         base_size,
                     float*       scale_data,
                     int3         scale_size,
                     float        global_scale,
                     float        shift,
                     float        epsilon,
                     uint32_t     flags,
                     cudaStream_t stream);
}

}  // namespace iluvatar
