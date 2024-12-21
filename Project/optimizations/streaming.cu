#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#define TILE_W1 16
#define TILE_WIDTH 16
__global__ void conv_forward_kernel(int batch_num, float *output, const float *input, const float *mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
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

    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;
    const int W_grid_dim = ceil(1.0 * W_out / TILE_W1); // blocks in grid requried to cover the length of the output
    // (void)H_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)W_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

#define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
#define in_3d(i2, i1, i0) input[(i2) * (H * W) + (i1) * (W) + i0]
    // Insert your GPU convolution kernel code here
    int m = blockIdx.y;
    int h = (blockIdx.z / W_grid_dim) * TILE_W1 + threadIdx.y;
    int w = (blockIdx.z % W_grid_dim) * TILE_W1 + threadIdx.x;
    int b = blockIdx.x;
    float result = 0.0f;
    if (w >= 0 && w < W_out && h >= 0 && h < H_out)
    {
        for (int c = 0; c < C; ++c)
        {
            for (int p = 0; p < K; ++p)
            {
                for (int q = 0; q < K; ++q)
                {
                    result += in_3d(c, (h * S) + p, (w * S) + q) * mask_4d(m, c, p, q);
                }
            }
        }
        out_4d(batch_num, m, h, w) = result;
    }

#undef out_4d
#undef in_4d
#undef mask_4d
}
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Allocate memory and copy over the relevant data structures to the GPU

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
    const int W_out = (W - K) / S + 1;

    int W_grid_dim = ceil(W_out / (float)TILE_W1);
    int H_grid = ceil(H_out / (float)TILE_W1);
    int Y = W_grid_dim * H_grid;
    cudaStream_t stream[B];
    int STREAM_SIZE = B;
    int batch_dim = (C * H * W) * sizeof(float);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid(1, M, Y);

    for (int i = 0; i < STREAM_SIZE; i++)
    {
        cudaStreamCreate(&stream[i]);
    }

    float *input[STREAM_SIZE];

    for (int i = 0; i < STREAM_SIZE; i++)
    {
        cudaMalloc((void **)&input[i], batch_dim);
        cudaMemcpyAsync(input[i], ((void *)device_input) + (i * batch_dim), batch_dim, cudaMemcpyDeviceToDevice, stream[i]);
        conv_forward_kernel<<<dimGrid, dimBlock, 0, stream[i]>>>(i, device_output, input[i], device_mask, B, M, C, H, W, K, S);
    }

    // for (int i = 0; i < STREAM_SIZE; i++)
    // {
    //     conv_forward_kernel<<<dimGrid, dimBlock, 0, stream[i]>>>(i, device_output, input[i], device_mask, B, M, C, H, W, K, S);
    // }
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