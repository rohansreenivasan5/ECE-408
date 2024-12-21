#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include "cuda_fp16.h"
#include <stdio.h>
#define TILE_W1 16

__constant__ __half deviceKernel[16 * 4 * 7 * 7];

__global__ void convert_to_floating_point(const float *input_vals, const int size, __half *output_half)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        output_half[i] = __float2half(input_vals[i]);
    }
}
__global__ void conv_forward_kernel(float *output, const __half *input, const float *mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{

    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;
    const int W_grid_dim = ceil((float)W_out / TILE_W1);

#define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define mask_4d(i3, i2, i1, i0) deviceKernel[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int m = blockIdx.x;
    int h = (blockIdx.y / W_grid_dim) * TILE_W1 + threadIdx.y;
    int w = (blockIdx.y % W_grid_dim) * TILE_W1 + threadIdx.x;
    int b = blockIdx.z;
    if (w >= 0 && w < W_out && h >= 0 && h < H_out)
    {
        __half acc = __float2half(0.0);

        for (int c = 0; c < C; ++c)
        {
            for (int p = 0; p < K; ++p)
            {
                for (int q = 0; q < K; ++q)
                {
                    acc = __hfma(in_4d(b, c, (h * S) + p, (w * S) + q), mask_4d(m, c, p, q), acc);
                }
            }
        }
        out_4d(b, m, h, w) = __half2float(acc);
    }

#undef out_4d
#undef in_4d
#undef mask_4d
}
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{

    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;
    int input_size = B * C * H * W;
    int output_size = B * M * H_out * W_out;
    int mask_size = M * C * K * K;

    cudaMalloc((void **)device_output_ptr, output_size * sizeof(float));
    cudaMalloc((void **)device_input_ptr, input_size * sizeof(float));
    cudaMemcpy(*device_input_ptr, host_input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    __half *half_values;
    half_values = (__half *)malloc(mask_size * sizeof(__half));
    for (int i = 0; i < mask_size; i++)
    {
        float f = host_mask[i];
        half_values[i] = __float2half(f);
    }
    cudaMemcpyToSymbol(deviceKernel, half_values, mask_size * sizeof(__half));
}

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{

    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;
    const int W_grid = ceil(1.0 * W_out / TILE_W1);
    const int H_grid = ceil(1.0 * H_out / TILE_W1);
    int Z = H_grid * W_grid;
    dim3 dimGrid(M, Z, B);
    dim3 dimBlock(TILE_W1, TILE_W1, 1);
    int input_size = B * C * H * W;
    __half *fixed_point_input;
    cudaMalloc((void **)&fixed_point_input, B * C * H * W * sizeof(__half));
    int div_size = 1024;
    convert_to_floating_point<<<(ceil((float)input_size / div_size)), div_size>>>(device_input, input_size, fixed_point_input);

    conv_forward_kernel<<<dimGrid, dimBlock>>>(device_output, fixed_point_input, device_mask, B, M, C, H, W, K, S);

    cudaFree(fixed_point_input);
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{

    const int Height_out = (H - K) / S + 1;
    const int Width_out = (W - K) / S + 1;
    int output_size = B * M * Height_out * Width_out;

    cudaMemcpy(host_output, device_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);
}

__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout << "Device " << dev << " name: " << deviceProp.name << std::endl;
        std::cout << "Computational capabilities: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "Max Global memory size: " << deviceProp.totalGlobalMem << std::endl;
        std::cout << "Max Constant memory size: " << deviceProp.totalConstMem << std::endl;
        std::cout << "Max Shared memory size per block: " << deviceProp.sharedMemPerBlock << std::endl;
        std::cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "Max block dimensions: " << deviceProp.maxThreadsDim[0] << " x, " << deviceProp.maxThreadsDim[1] << " y, " << deviceProp.maxThreadsDim[2] << " z" << std::endl;
        std::cout << "Max grid dimensions: " << deviceProp.maxGridSize[0] << " x, " << deviceProp.maxGridSize[1] << " y, " << deviceProp.maxGridSize[2] << " z" << std::endl;
        std::cout << "Warp Size: " << deviceProp.warpSize << std::endl;
    }
}