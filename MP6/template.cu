// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                \
  do                                                                 \
  {                                                                  \
    cudaError_t err = stmt;                                          \
    if (err != cudaSuccess)                                          \
    {                                                                \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                    \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err)); \
      return -1;                                                     \
    }                                                                \
  } while (0)

__global__ void scan(float *input, float *output, int len)
{

  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  // each thread loads two values into shared mem
  __shared__ float T[2 * BLOCK_SIZE];
  unsigned int t = threadIdx.x;
  unsigned int start = 2 * blockIdx.x * blockDim.x;

  T[t] = (start + t < len) ? input[start + t] : 0.0;
  T[blockDim.x + t] = (start + blockDim.x + t < len) ? input[start + blockDim.x + t] : 0.0;

  int stride = 1;                 // creates the upsweep phase or prescan phase.
  while (stride < 2 * BLOCK_SIZE) // we are processes 2* blocksize elements anyway
  {
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 - 1; // when stride = 1, index will be 1,3,5,7, and sum into values T[1], T[3], T[5]
    // when stride = 2 we will now have index be 3, 7, 11
    if (index < 2 * BLOCK_SIZE && (index - stride) >= 0)
      T[index] += T[index - stride]; // when S = 1, we have T[1] += T[0], T[3] += T[2]....
    // WHEN S = 2 we have T[3] += T[1], T[7] += T[5]
    stride = stride * 2;
  }

  // POST SCAN STEP
  stride = BLOCK_SIZE / 2;
  while (stride > 0)
  {
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 - 1; // Index will now get us values we didnt hit before, we will sum values starting from BLOCKSIZE / 2

    if ((index + stride) < 2 * BLOCK_SIZE)
      T[index + stride] += T[index]; // now add T[index + stride] += T[index], S = 4, T[7] += T[3]
    stride = stride / 2;
  }
  __syncthreads();

  // write to output
  if (start + t < len)
  {
    output[start + t] = T[t];
  }
  if (start + t + blockDim.x < len)
  {
    output[start + t + blockDim.x] = T[t + blockDim.x];
  }
}

__global__ void finalAdd(float *input, float *auxArrayPrefixSum, int len)
{                                                                // this kernel will sum out auxArray with all the block prefix sums starting with block idx 1
  int index = 2 * ((blockIdx.x + 1) * blockDim.x) + threadIdx.x; // global addressing of threads in blocks [1 to n]
  // do the first add
  if (index < len)
  {
    input[index] += auxArrayPrefixSum[blockDim.x];
  }
  // second add, one blockDim Away in global mem, since each thread [o to block dim] does 2 elements
  if (index + blockDim.x < len)
  {
    input[index + blockDim.x] += auxArrayPrefixSum[blockDim.x];
  }
}

int main(int argc, char **argv)
{
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;

  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  int num_blocks_in_grid = ceil(1.0 * numElements / (2 * BLOCK_SIZE));
  dim3 gridDim(num_blocks_in_grid, 1, 1); // One block for each output element. then we can have host add the rest
  dim3 blockDim(BLOCK_SIZE, 1, 1);        // threads / block

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce

  /*
  Intuition: we want to to call parallel scan on the first array where each block creates a prefiex sum. Each block assumes it
  is starting at the beggining of the array. since we want a prefix sum for the whole array, we will make a new array for with the
  last element of each block in an array. call prefix sum on this array. Then take each element of this array and add to each block sum satrting
  from block idx 1.
  */

  // run scan so we now have the auxArray in the output array
  scan<<<gridDim, blockDim>>>(deviceInput, deviceOutput, numElements);
  cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
             cudaMemcpyDeviceToHost);

  // host output now has the intermediateoutput
  float *auxArray = (float *)malloc(num_blocks_in_grid * sizeof(float)); // aux array is not the copy of output, its the last elem from each block
  for (int i = 0; i < num_blocks_in_grid; i++)
  {
    auxArray[i] = hostOutput[((i + 1) * BLOCK_SIZE * 2) - 1];
  }

  // auxArray now has last elem from each block
  // now we want to call scan again on aux Array. we need to copy it to a device array and get it back from a device output array

  float *deviceInputAuxArray;
  float *deviceOutputAuxArray;
  cudaMalloc((void **)&deviceInputAuxArray, num_blocks_in_grid * sizeof(float));
  cudaMalloc((void **)&deviceOutputAuxArray, num_blocks_in_grid * sizeof(float));
  cudaMemcpy(deviceInputAuxArray, auxArray, num_blocks_in_grid * sizeof(float),
             cudaMemcpyHostToDevice);

  dim3 gridDimHelper(1, 1, 1);                             // we need a new kernel launch for the second scan, only need one block
  dim3 blockDimHelper(ceil(num_blocks_in_grid / 2), 1, 1); // each thread does 2 elements so div 2
  scan<<<gridDimHelper, blockDimHelper>>>(deviceInputAuxArray, deviceOutputAuxArray, num_blocks_in_grid);
  finalAdd<<<gridDim, blockDim>>>(deviceOutput, deviceOutputAuxArray, numElements);
  // no need to copy back to host, we can just get do our final add

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
