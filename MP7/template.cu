// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define TILE_WIDTH 16

//@@ insert code here
__global__ void float_to_unsigned_char(float *input, unsigned char *output, int width, int height, int numChannels)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int channel = threadIdx.z;
  if (x < width && y < height)
  {
    int idx = (((y * width) + x) * numChannels) + channel;
    output[idx] = (unsigned char)(255 * input[idx]);
  }
}

__global__ void rgb_to_grayscale(unsigned char *input, unsigned char *output, int width, int height, int numChannels)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  if (x < width && y < height)
  {
    int grayIdx = ((y * width) + x);
    int rgbIdx = grayIdx * numChannels;
    unsigned char r = input[rgbIdx];
    unsigned char g = input[rgbIdx + 1];
    unsigned char b = input[rgbIdx + 2];
    output[grayIdx] = (unsigned char)(0.21 * r + 0.71 * g + 0.07 * b);
  }
}

__global__ void histogram(unsigned char *input, int *histo, int width, int height)
{
  __shared__ int private_histo[HISTOGRAM_LENGTH];
  int idxInBlock = blockDim.x * threadIdx.y + threadIdx.x; // index of element within this block only, which is why we don't consider blockDim.y
  if (idxInBlock < HISTOGRAM_LENGTH)
  {
    private_histo[idxInBlock] = 0;
  }

  __syncthreads();

  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  if (x < width && y < height)
  { // done by every thread launched?
    int idx = (y * width) + x;
    atomicAdd(&(private_histo[input[idx]]), 1);
  }

  __syncthreads();

  if (idxInBlock < HISTOGRAM_LENGTH)
  {                                                             // only done by 256 threads-- one for every histogram element?
    atomicAdd(&(histo[idxInBlock]), private_histo[idxInBlock]); // add the block-level private result to the global histogram
  }
}

__global__ void histogram_cdf(int *histo_input, float *output, int width, int height)
{
  float numElements = 1.0 * width * height;
  // COPIED OVER MY OWN SCAN CODE FROM MP6 BELOW -- do we need finalAdd()?
  __shared__ float T[HISTOGRAM_LENGTH]; // updated to histogram size as that is what we are running scan over

  int bx = blockIdx.x;
  int tx = threadIdx.x;
  int start = 2 * (bx * blockDim.x);

  // each thread loads two elems
  if (start + tx < HISTOGRAM_LENGTH)
  {
    T[tx] = histo_input[start + tx] / (1.0 * numElements);
  }
  else
  {
    T[tx] = 0.0;
  }
  if (start + tx + blockDim.x < HISTOGRAM_LENGTH)
  {
    T[tx + blockDim.x] = histo_input[start + tx + blockDim.x] / (1.0 * numElements);
  }
  else
  {
    T[tx + blockDim.x] = 0.0;
  }

  // reduction
  int stride = 1;
  while (stride < HISTOGRAM_LENGTH)
  {
    __syncthreads();
    int index = (tx + 1) * stride * 2 - 1;
    if (index < HISTOGRAM_LENGTH && (index - stride) >= 0)
    {
      T[index] += T[index - stride];
    }
    stride *= 2;
  }

  // post scan
  int post_stride = blockDim.x / 2;
  while (post_stride > 0)
  {
    __syncthreads();
    int post_index = (tx + 1) * post_stride * 2 - 1;
    if ((post_index + post_stride) < HISTOGRAM_LENGTH)
    {
      T[post_index + post_stride] += T[post_index];
    }
    post_stride /= 2;
  }

  __syncthreads();

  // write both calculated elements to output
  if (start + tx < HISTOGRAM_LENGTH)
  {
    output[start + tx] = T[tx] + ((histo_input[tx] * 1.0) / numElements); // apply p(x) probability function defined in spec
  }
  if (start + tx + blockDim.x < HISTOGRAM_LENGTH)
  {
    output[start + tx + blockDim.x] = T[tx + blockDim.x] + ((histo_input[tx + blockDim.x] * 1.0) / numElements); // apply p(x) probability function defined in spec
  }
}

// the equalized value of each pixel in the input image only depends on the corrected_color calculation of itself, so we can modify the passed input image directly and do not have to write to a separate output
__global__ void histogram_equalization(float *histo_cdf, unsigned char *input, int width, int height, int numChannels)
{
  // create a shared variable cdf_min so it can be used by all threads during this part of the calculation
  __shared__ float cdf_min;
  if (threadIdx.x == 0 && threadIdx.y == 0)
  { // cdf_min should only be modified/updated by one thread, so the thread with indices 0,0 will modify it to contain the min value
    cdf_min = histo_cdf[0];
  }
  __syncthreads(); // syncthreads after updating min so every thread has it for calculations below

  // combines correct_color, clamp, and histogram_equalization to avoid passing around params to multiple functions
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int channel = threadIdx.z;

  if (x < width && y < height)
  {
    int idx = (((y * width) + x) * numChannels) + channel;
    input[idx] = min(max(255 * (histo_cdf[input[idx]] - cdf_min) / (1.0 - cdf_min), 0.0), 255.0); // apply correct_color(val) and clamp(x, start, end) inline
  }
}

__global__ void unsigned_char_to_float(unsigned char *input, float *output, int width, int height, int numChannels)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int channel = threadIdx.z;
  if (x < width && y < height)
  {
    int idx = (((y * width) + x) * numChannels) + channel;
    output[idx] = (float)((input[idx] * 1.0) / 255);
  }
}

int main(int argc, char **argv)
{
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  // initialize variables for all steps of the process
  float *deviceInputImageFloat;
  unsigned char *deviceInputImageUnsigned;
  unsigned char *deviceInputImageGrayscale;
  int *deviceHistogram;
  float *deviceHistogramCDF;
  float *deviceOutputImageFloat;

  // allocate memory for these variables on the device
  cudaMalloc((void **)&deviceInputImageFloat, (imageWidth * imageHeight * imageChannels) * sizeof(float));
  cudaMalloc((void **)&deviceInputImageUnsigned, (imageWidth * imageHeight * imageChannels) * sizeof(unsigned char));
  cudaMalloc((void **)&deviceInputImageGrayscale, (imageWidth * imageHeight) * sizeof(unsigned char));
  cudaMalloc((void **)&deviceHistogram, (HISTOGRAM_LENGTH) * sizeof(int));
  cudaMalloc((void **)&deviceHistogramCDF, (HISTOGRAM_LENGTH) * sizeof(float));
  cudaMalloc((void **)&deviceOutputImageFloat, (imageWidth * imageHeight * imageChannels) * sizeof(float));

  // copy input image from host to device
  cudaMemcpy(deviceInputImageFloat, hostInputImageData, (imageWidth * imageHeight * imageChannels) * sizeof(float), cudaMemcpyHostToDevice);

  // STEP 1: CONVERT INPUT IMAGE FROM FLOAT TO UNSIGNED CHAR
  // create dimensions for float to unsigned char conversion
  dim3 DimGridForConversion(ceil((1.0 * imageWidth) / TILE_WIDTH), ceil((1.0 * imageHeight) / TILE_WIDTH), 1); // (X, Y, 1) for 2D image
  dim3 DimBlockMultipleChannels(TILE_WIDTH, TILE_WIDTH, imageChannels);                                        // by adding 3rd dim for channels, we can parallelize a bit more
  // launch kernel
  float_to_unsigned_char<<<DimGridForConversion, DimBlockMultipleChannels>>>(deviceInputImageFloat, deviceInputImageUnsigned, imageWidth, imageHeight, imageChannels);

  // STEP 2: CONVERT INPUT IMAGE FROM RGB TO GRAYSCALE
  // create new block dimensions for rgb to grayscale
  dim3 DimBlockSingleChannel(TILE_WIDTH, TILE_WIDTH, 1); // once the image has been converted from rgb to grayscale, there is only one channel, so the conversion process can only have one channel
  // launch kernel
  rgb_to_grayscale<<<DimGridForConversion, DimBlockSingleChannel>>>(deviceInputImageUnsigned, deviceInputImageGrayscale, imageWidth, imageHeight, imageChannels);

  // STEP 3: COMPUTE HISTOGRAM FROM GRAYSCALE IMAGE
  // launch kernel
  histogram<<<DimGridForConversion, DimBlockSingleChannel>>>(deviceInputImageGrayscale, deviceHistogram, imageWidth, imageHeight);

  // STEP 4: COMPUTE CDF OF HISTOGRAM
  // create new block and grid dimensions for cdf scan of histogram
  dim3 DimGridHistogramCDF(1, 1, 1);                           // we only need one block because the histogram has 256 elements and we can process that in one block. to understand the logic for calculating launch code for scans, look back at MP6 code
  dim3 DimBlockHistogramCDF(ceil(HISTOGRAM_LENGTH / 2), 1, 1); // shouldn't need the ceil bc HISTOGRAM_LENGTH is 256, but better safe than sorry
  // launch kernel
  histogram_cdf<<<DimGridHistogramCDF, DimBlockHistogramCDF>>>(deviceHistogram, deviceHistogramCDF, imageWidth, imageHeight);

  // STEP 5: EQUALIZE THE HISTOGRAM
  // launch kernel
  histogram_equalization<<<DimGridForConversion, DimBlockMultipleChannels>>>(deviceHistogramCDF, deviceInputImageUnsigned, imageWidth, imageHeight, imageChannels);

  // STEP 6: CONVERT OUTPUT IMAGE FROM UNSIGNED CHAR BACK TO FLOAT
  // launch kernel
  unsigned_char_to_float<<<DimGridForConversion, DimBlockMultipleChannels>>>(deviceInputImageUnsigned, deviceOutputImageFloat, imageWidth, imageHeight, imageChannels);

  // synchronize to make sure computation is complete
  cudaDeviceSynchronize();

  // copy output image from device back to host
  cudaMemcpy(hostOutputImageData, deviceOutputImageFloat, (imageWidth * imageHeight * imageChannels) * sizeof(float), cudaMemcpyDeviceToHost);

  wbSolution(args, outputImage);

  //@@ insert code here
  cudaFree(deviceInputImageFloat);
  cudaFree(deviceInputImageUnsigned);
  cudaFree(deviceInputImageGrayscale);
  cudaFree(deviceHistogram);
  cudaFree(deviceHistogramCDF);
  cudaFree(deviceOutputImageFloat);

  return 0;
}
