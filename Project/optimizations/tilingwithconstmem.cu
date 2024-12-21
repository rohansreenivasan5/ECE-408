#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#define TILE_W1 16

__global__ void tiled_kernel(float *output, const float *input, const float *mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    S - stride step length
    */

    extern __shared__ float sharedTile[];

    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;
    const int W_grid_dim = ceil(1.0 * W / TILE_W1);

#define tile_3d(chan, y, x) sharedTile[(chan) * (TILE_W1 * TILE_W1) + (y) * (TILE_W1) + (x)]

#define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int m = blockIdx.y;
    int h = (blockIdx.z / W_grid_dim) * TILE_W1 + threadIdx.y;
    int w = (blockIdx.z % W_grid_dim) * TILE_W1 + threadIdx.x;
    int b = blockIdx.x;
    if (w < W && h < H && b < B && m < M)
    {
        for (int i = 0; i < C; i++)
        {
            tile_3d(i, threadIdx.y, threadIdx.x) = in_4d(b, i, h, w);
        }
    }
    __syncthreads();

    if (h % S == 0 && h + K <= H && w % S == 0 && w + K <= W && b < B && m < M)
    {
        float result = 0.0;
        for (int c = 0; c < C; ++c)
        {
            for (int p = 0; p < K; ++p) // p = y dim in mask, q = x dim in mask
            {
                for (int q = 0; q < K; ++q)
                {
                    if (threadIdx.x + q < TILE_W1 && threadIdx.y + p < TILE_W1)
                    {
                        result += mask_4d(m, c, p, q) * tile_3d(c, threadIdx.y + p, threadIdx.x + q);
                    }
                    else
                    {
                        result += mask_4d(m, c, p, q) * in_4d(b, c, h + p, w + q);
                    }
                }
            }
        }
        out_4d(b, m, h / S, w / S) = result;
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
    cudaMalloc((void **)device_mask_ptr, mask_size * sizeof(float));

    cudaMalloc((void **)device_input_ptr, input_size * sizeof(float));
    cudaMemcpy(*device_input_ptr, host_input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, mask_size * sizeof(float), cudaMemcpyHostToDevice);
}

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Set the kernel dimensions and call the kernel

    const int H_out = (H - K) / S + 1;
    const int Width_out = (W - K) / S + 1;

    int W_grid_dim = ceil(W / (float)TILE_W1);
    int H_grid = ceil(H / (float)TILE_W1);
    int Y = W_grid_dim * H_grid;
    dim3 dimBlock(TILE_W1, TILE_W1, 1);
    dim3 dimGrid(B, M, Y);
    tiled_kernel<<<dimGrid, dimBlock, C * TILE_W1 * TILE_W1 * sizeof(float)>>>(device_output, device_input, device_mask, B, M, C, H, W, K, S);
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Copy the output back to host

    // Free device memory
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