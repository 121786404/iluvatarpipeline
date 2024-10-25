#include <cmath>
#include <cstdio>
#include <typeinfo>
#include <assert.h>
#include <algorithm>
#include "preprocess.h"

#define CV_DESCALE(x, n) (((x) + (1 << ((n)-1))) >> (n))

inline int div_up(int a, int b)
{
    assert(b > 0);
    return ceil((float)a / b);
};

namespace iluvatar
{
namespace cropResize
{
__global__ void crop_resize_var_IYUV_linear_uchar3(uint8_t** src,
                                                   uint8_t*  dst,
                                                   int2*     in_shape,
                                                   unsigned  dstWidth,
                                                   unsigned  dstHeight,
                                                   RectA*    cropRect)
{
    const unsigned dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned dst_y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((dst_x >= dstWidth) || (dst_y >= dstHeight))
        return;

    const unsigned batch_idx = blockIdx.z; 
    unsigned imageIdx = cropRect[batch_idx].imageIdx;
    iluvatar::cropResize::Rect rect = cropRect[batch_idx].rect;

    const unsigned srcImageWidth = in_shape[imageIdx].x;
    const unsigned srcImageHeight = in_shape[imageIdx].y;

    unsigned cropX = rect.x;
    unsigned cropY = rect.y;
    unsigned cropWidth = rect.width;
    unsigned cropHeight = rect.height; 
    
    float scale_x = ((float)cropWidth) / ((float)dstWidth);
    float scale_y = ((float)cropHeight) / ((float)dstHeight);

    float fy = (float)(((float)dst_y + 0.5f) * scale_y - 0.5f);
    int sy = __float2int_rd(fy);
    fy -= sy;
    sy = max(0, min(sy, (int)cropHeight - 2));

    uint8_t* srcImage_base = src[imageIdx];
    unsigned srcImage_plane_y_step = srcImageWidth * srcImageHeight;
    unsigned srcImage_plane_uv_step = srcImage_plane_y_step >> 2;

    unsigned offset_a_y = (cropY + sy) * srcImageWidth + cropX;
    uint8_t* aPtr_y = srcImage_base + offset_a_y;

    unsigned offset_b_y = offset_a_y + srcImageWidth;
    uint8_t* bPtr_y = srcImage_base + offset_b_y;

    float fx = (float)(((float)dst_x + 0.5f) * scale_x - 0.5f);
    int sx = __float2int_rd(fx);
    fx -= sx;
    fx *= ((sx >= 0) && (sx < (int)cropWidth - 1));
    sx = max(0, min(sx, (int)cropWidth - 2));

    unsigned dst_plane_y_step = dstWidth * dstHeight;
    unsigned dst_plane_uv_step = dst_plane_y_step >> 2;
    unsigned dstStride = (batch_idx * dst_plane_y_step * 3) >> 1;
    unsigned offset_dst_y = dstStride + dst_y * dstWidth + dst_x;

    int re_y = (int)(((1.0f - fx) * (aPtr_y[sx] * (1.0f - fy) + bPtr_y[sx] * fy) + fx * (aPtr_y[sx + 1] * (1.0f - fy) + bPtr_y[sx + 1] * fy)) + 0.5f);
    re_y = re_y > 255 ? 255 : (re_y < 0 ? 0 : re_y);
    *(dst + offset_dst_y) = (uint8_t)re_y;

    if ((dst_x & 1) == 0 && (dst_y & 1) == 0)
    {
        int image_width_uv = srcImageWidth >> 1;

        unsigned offset_aPtr_y_sx_u = srcImage_plane_y_step + (unsigned)(((cropY + sy) >> 1) * image_width_uv) + (unsigned)(cropX >> 1) + (unsigned)(sx >> 1);
        unsigned offset_aPtr_y_sx_v = offset_aPtr_y_sx_u + srcImage_plane_uv_step;
        uint8_t aPtr_u_sx = *(srcImage_base + offset_aPtr_y_sx_u);
        uint8_t aPtr_v_sx = *(srcImage_base + offset_aPtr_y_sx_v);

        unsigned offset_bPtr_y_sx_u = srcImage_plane_y_step + (unsigned)(((cropY + sy + 1) >> 1) * image_width_uv) + (unsigned)(cropX >> 1) + (unsigned)(sx >> 1);
        unsigned offset_bPtr_y_sx_v = offset_bPtr_y_sx_u + srcImage_plane_uv_step;
        uint8_t bPtr_u_sx = *(srcImage_base + offset_bPtr_y_sx_u);
        uint8_t bPtr_v_sx = *(srcImage_base + offset_bPtr_y_sx_v);

        unsigned offset_aPtr_y_sx1_u = srcImage_plane_y_step + (unsigned)(((cropY + sy) >> 1) * image_width_uv) + (unsigned)(cropX >> 1) + (unsigned)((sx + 1) >> 1);
        unsigned offset_aPtr_y_sx1_v = offset_aPtr_y_sx1_u + srcImage_plane_uv_step;
        uint8_t aPtr_u_sx1 = *(srcImage_base + offset_aPtr_y_sx1_u);
        uint8_t aPtr_v_sx1 = *(srcImage_base + offset_aPtr_y_sx1_v);

        unsigned offset_bPtr_y_sx1_u = srcImage_plane_y_step + (unsigned)(((cropY + sy + 1) >> 1) * image_width_uv) + (unsigned)(cropX >> 1) + (unsigned)((sx + 1) >> 1);
        unsigned offset_bPtr_y_sx1_v = offset_bPtr_y_sx1_u + srcImage_plane_uv_step;
        uint8_t bPtr_u_sx1 = *(srcImage_base + offset_bPtr_y_sx1_u);
        uint8_t bPtr_v_sx1 = *(srcImage_base + offset_bPtr_y_sx1_v);

        int re_u = (int)(((1.0f - fx) * (aPtr_u_sx * (1.0f - fy) + bPtr_u_sx * fy) + fx * (aPtr_u_sx1 * (1.0f - fy) + bPtr_u_sx1 * fy)) + 0.5f);
        int re_v = (int)(((1.0f - fx) * (aPtr_v_sx * (1.0f - fy) + bPtr_v_sx * fy) + fx * (aPtr_v_sx1 * (1.0f - fy) + bPtr_v_sx1 * fy)) + 0.5f);
        re_u = re_u > 255 ? 255 : (re_u < 0 ? 0 : re_u);
        re_v = re_v > 255 ? 255 : (re_v < 0 ? 0 : re_v);

        unsigned out_width_uv = dstWidth >> 1;
        unsigned offset_dst_u = dstStride + dst_plane_y_step + (dst_y >> 1) * out_width_uv + (dst_x >> 1);
        unsigned offset_dst_v = offset_dst_u + dst_plane_uv_step;
        *(dst + offset_dst_u) = (uint8_t)re_u;
        *(dst + offset_dst_v) = (uint8_t)re_v;
    }
}

template <typename T>
void CropResize_var_kernel_launch(T**          input,
                                  T*           output,
                                  int2*        in_shape,
                                  unsigned     out_width,
                                  unsigned     out_height,
                                  RectA*       cropRect,
                                  unsigned     rectNum,
                                  unsigned     crop_imageFormat,
                                  unsigned     resize_interType,
                                  cudaStream_t stream)
{
    const dim3 blockSize(64, 32, 1);
    const dim3 gridSize(div_up(out_width, blockSize.x), div_up(out_height, blockSize.y), rectNum);

    switch (crop_imageFormat)
    {
        case CROP_RESIZE_IYUV:
        {
            if ((out_width % 2 != 0) || (out_height % 2 != 0))
                break;

            if ((resize_interType == INTERP_LINEAR) && (typeid(T) == typeid(uint8_t)))
            {
                crop_resize_var_IYUV_linear_uchar3<<<gridSize, blockSize, 0, stream>>>(
                    input, output, in_shape, out_width, out_height, cropRect);
            }
        }
        break;

        default: break;
    }
}

void CropResize(uint8_t**    input,
                uint8_t*     output,
                int2*        in_shape,
                unsigned     out_width,
                unsigned     out_height,
                RectA*       cropRect,
                unsigned     rectNum,
                unsigned     crop_imageFormat,
                unsigned     resize_interType,
                cudaStream_t stream)
{
    CropResize_var_kernel_launch<uint8_t>(
        input, output, in_shape, out_width, out_height, cropRect, rectNum, crop_imageFormat, resize_interType, stream);
}

}  // namespace cropResize

namespace CvtNormReformat
{
constexpr int ITUR_BT_601_CY    = 1220542;
constexpr int ITUR_BT_601_CUB   = 2116026;
constexpr int ITUR_BT_601_CUG   = -409993;
constexpr int ITUR_BT_601_CVG   = -852492;
constexpr int ITUR_BT_601_CVR   = 1673527;
constexpr int ITUR_BT_601_SHIFT = 20;

__device__ __forceinline__ void yuv42xxp_to_bgr_kernel(const uint8_t& Y,
                                                       const uint8_t& U,
                                                       const uint8_t& V,
                                                       uint8_t&       r,
                                                       uint8_t&       g,
                                                       uint8_t&       b)
{
    const int C0 = ITUR_BT_601_CY, C1 = ITUR_BT_601_CVR, C2 = ITUR_BT_601_CVG, C3 = ITUR_BT_601_CUG,
              C4           = ITUR_BT_601_CUB;
    const int yuv4xx_shift = ITUR_BT_601_SHIFT;

    int yy = std::max(0, (int)Y - 16) * C0;
    int uu = (int)U - 128;
    int vv = (int)V - 128;

    int rr = CV_DESCALE((yy + C1 * vv), yuv4xx_shift);
    int gg = CV_DESCALE((yy + C2 * vv + C3 * uu), yuv4xx_shift);
    int bb = CV_DESCALE((yy + C4 * uu), yuv4xx_shift);

    r = rr > 255 ? 255 : (rr < 0 ? 0 : rr);
    g = gg > 255 ? 255 : (gg < 0 ? 0 : gg);
    b = bb > 255 ? 255 : (bb < 0 ? 0 : bb);
}

__global__ void normalize_yuv420p2rgbf32p(uint8_t*    src,
                                          float*      dst,
                                          unsigned    width,
                                          unsigned    height,
                                          int         bidx,
                                          const float alpha,
                                          const float beta,
                                          bool        norm_flag,
                                          float*      base_data,
                                          int3        base_size,
                                          float*      scale_data,
                                          int3        scale_size,
                                          float       global_scale,
                                          float       shift,
                                          float       epsilon,
                                          bool        stddev)
{
    const unsigned tid_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned tid_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned batch_idx = blockIdx.z;

    if ((tid_x >= width) || (tid_y >= height))
        return;

    unsigned plane_y_step    = width * height;
    unsigned plane_uv_step   = plane_y_step >> 2;
    unsigned offset_src_base = batch_idx * (plane_y_step + 2 * plane_uv_step);

    unsigned offset_src_y = offset_src_base + tid_y * width + tid_x;
    uint8_t  Y            = *(src + offset_src_y);

    int      width_uv     = width >> 1;
    unsigned offset_dst_u = offset_src_base + plane_y_step + (tid_y >> 1) * width_uv + (tid_x >> 1);
    uint8_t  U            = *(src + offset_dst_u);

    unsigned offset_dst_v = offset_dst_u + plane_uv_step;
    uint8_t  V            = *(src + offset_dst_v);

    uint8_t Ri8{0}, Gi8{0}, Bi8{0};
    yuv42xxp_to_bgr_kernel(Y, U, V, Ri8, Gi8, Bi8);

    float Rf32 = alpha * static_cast<float>(Ri8) + beta;
    float Gf32 = alpha * static_cast<float>(Gi8) + beta;
    float Bf32 = alpha * static_cast<float>(Bi8) + beta;

    unsigned offset_dst = batch_idx * 3 * plane_y_step;
    unsigned offset_s   = tid_y * width + tid_x;

    if (norm_flag)
    {
        const unsigned base_x          = base_size.x == 1 ? 0 : tid_x;
        const unsigned base_y          = base_size.y == 1 ? 0 : tid_y;
        const unsigned base_batch_idx  = base_size.z == 1 ? 0 : batch_idx;
        const unsigned scale_x         = scale_size.x == 1 ? 0 : tid_x;
        const unsigned scale_y         = scale_size.y == 1 ? 0 : tid_y;
        const unsigned scale_batch_idx = scale_size.z == 1 ? 0 : batch_idx;

        unsigned offset_scale = (scale_batch_idx * scale_size.y * scale_size.x + scale_y * scale_size.x + scale_x) * 3;
        float3*  scalePtr     = reinterpret_cast<float3*>(scale_data + offset_scale);

        unsigned offset_base = (base_batch_idx * base_size.y * base_size.x + base_y * base_size.x + base_x) * 3;
        float3*  basePtr     = reinterpret_cast<float3*>(base_data + offset_base);

        float3 mul = *scalePtr;
        if (stddev)
        {
            mul.x = 1.0f / sqrt(mul.x * mul.x + epsilon);
            mul.y = 1.0f / sqrt(mul.y * mul.y + epsilon);
            mul.z = 1.0f / sqrt(mul.z * mul.z + epsilon);
        }

        float R_normal = (Rf32 - basePtr->x) * mul.x * global_scale + shift;
        float G_normal = (Gf32 - basePtr->y) * mul.y * global_scale + shift;
        float B_normal = (Bf32 - basePtr->z) * mul.z * global_scale + shift;

        
        *(dst + offset_dst + bidx * plane_y_step + offset_s)       = B_normal;
        *(dst + offset_dst + plane_y_step + offset_s)              = G_normal;
        *(dst + offset_dst + (bidx ^ 2) * plane_y_step + offset_s) = R_normal;
    }
    else
    {
        *(dst + offset_dst + bidx * plane_y_step + offset_s)       = Bf32;
        *(dst + offset_dst + plane_y_step + offset_s)              = Gf32;
        *(dst + offset_dst + (bidx ^ 2) * plane_y_step + offset_s) = Rf32;
    }
}

template <typename T, typename C>
void CvtcolorConvertoNormalizeReformat_kernel_launch(T*           input,
                                                     C*           output,
                                                     unsigned     cvtcolor_reformat_type,
                                                     unsigned     batches,
                                                     unsigned     width,
                                                     unsigned     height,
                                                     float        alpha,
                                                     float        beta,
                                                     bool         norm_flag,
                                                     C*           base_data,
                                                     int3         base_size,
                                                     C*           scale_data,
                                                     int3         scale_size,
                                                     float        global_scale,
                                                     float        shift,
                                                     float        epsilon,
                                                     uint32_t     flags,
                                                     cudaStream_t stream)
{
    switch (cvtcolor_reformat_type)
    {
        case CVTCOLOR_REFORMAT_IYUV2RGBf32p:
        case CVTCOLOR_REFORMAT_IYUV2BGRf32p:
        {
            if ((width % 2 != 0) || (height % 2 != 0))
                break;

            bool stddev = flags & NORMALIZE_SCALE_IS_STDDEV;
            int  bidx   = (cvtcolor_reformat_type == CVTCOLOR_REFORMAT_IYUV2BGRf32p) ? 0 : 2;

            dim3 blockSize(32, 32, 1);
            dim3 gridSize(div_up(width, blockSize.x), div_up(height, blockSize.y), batches);
            normalize_yuv420p2rgbf32p<<<gridSize, blockSize, 0, stream>>>(input,
                                                                          output,
                                                                          width,
                                                                          height,
                                                                          bidx,
                                                                          alpha,
                                                                          beta,
                                                                          norm_flag,
                                                                          base_data,
                                                                          base_size,
                                                                          scale_data,
                                                                          scale_size,
                                                                          global_scale,
                                                                          shift,
                                                                          epsilon,
                                                                          stddev);
        }
        break;

        default: break;
    }
}

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
                     cudaStream_t stream)
{
    CvtcolorConvertoNormalizeReformat_kernel_launch<uint8_t, float>(input,
                                                                    output,
                                                                    cvtcolor_reformat_type,
                                                                    batches,
                                                                    width,
                                                                    height,
                                                                    alpha,
                                                                    beta,
                                                                    norm_flag,
                                                                    base_data,
                                                                    base_size,
                                                                    scale_data,
                                                                    scale_size,
                                                                    global_scale,
                                                                    shift,
                                                                    epsilon,
                                                                    flags,
                                                                    stream);
}

}  // namespace CvtNormReformat

}  // namespace iluvatar
