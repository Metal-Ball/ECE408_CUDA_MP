// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256

//@@ insert code here
# define BLOCK_X 32
# define BLOCK_Y 32
# define BLOCK_SIZE 1024
# define HIST_LEN   HISTOGRAM_LENGTH
# define HALF_HIST_LEN HIST_LEN/2


__global__
void imageCast(float *src, unsigned char *dst, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < size) dst[idx] = (unsigned char)(255 * src[idx]);
}

__global__
void imageCastReverse(unsigned char *src, float *dst, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < size) dst[idx] = src[idx] / 255.0;
}

__global__
void grayScale(unsigned char *rgb, unsigned char *gray, int length)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length){
    float r = (float)rgb[3*idx];
    float g = (float)rgb[3*idx + 1];
    float b = (float)rgb[3*idx + 2];
    gray[idx] = (unsigned char) (0.21*r + 0.71*g + 0.07*b);
  }
}

__global__
void histo(unsigned char *img, unsigned int *histogram, int length){
  __shared__ unsigned int buffer[HIST_LEN];
  int tx = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // clear histogram junk data
  if(tx < HIST_LEN)  buffer[tx] = 0;
  __syncthreads();

  // build shared histogram buffer
  int stride = blockDim.x * gridDim.x;
  while(idx < length){
    int gray_idx = (int)img[idx];
    atomicAdd(&(buffer[gray_idx]), 1);
    idx += stride;
  } 
  __syncthreads();

  // contribute to global histogram 
  if(tx < HIST_LEN) atomicAdd(&(histogram[tx]), buffer[tx]);
}

__global__ 
void scan_cdf(unsigned int *histogram, float *cdf, int numPixels) 
{
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  
  // step 1: each thread load two input elements into shared memory
  __shared__ float buffer[HIST_LEN];
  int start = HIST_LEN  * blockIdx.x;
  int tx = threadIdx.x;
  // each thread loads two elements
  int first = start+tx;
  int second = first + HALF_HIST_LEN;
  if(first  < HIST_LEN) buffer[tx] = ((float)histogram[first])/numPixels;
  else buffer[tx] = 0.0;
  if(second < HIST_LEN) buffer[tx + HALF_HIST_LEN] = ((float)histogram[second])/numPixels;
  else buffer[tx + HALF_HIST_LEN] = 0.0;
 
  // step 2: stride 1 to HALF_HIST_LEN
  int stride = 1;
  while(stride < HIST_LEN){
    // finish reading for first iteration
    // or finish writing to buffer
    __syncthreads();
    // threadIdx => buffer index
    int idx = (tx+1)*stride*2-1;
    if(idx < HIST_LEN && idx>=stride)
      // reduction
      buffer[idx] += buffer[idx-stride];
    // update stride
    stride *= 2;
  }
  
  // step 3: stide HALF_HIST_LEN to 1
  stride = HALF_HIST_LEN;
  while(stride > 0){
    // wait for all threads to finish writing
    __syncthreads();
    // threadIdx => buffer index
    int idx = (tx+1)*stride*2-1;
    if(idx + stride < HIST_LEN)
      //reduction
      buffer[idx+stride] += buffer[idx];
    stride /= 2;
  }
  
  // step 4: copy shared memory to cdf
  __syncthreads();
  if(first  < HIST_LEN) cdf[first]  = buffer[tx];
  if(second < HIST_LEN) cdf[second] = buffer[tx + HALF_HIST_LEN];
}
//some helper functions
__device__
unsigned char clamp(float x, float start, float end)
{
  float max = x>start ? x:start;
  return (unsigned char)(max<end ? max:end);
}

__device__
unsigned char correct_color(float cdf_val, float cdf_min) 
{
  return clamp(255*(cdf_val - cdf_min)/(1.0 - cdf_min), 0.0, 255.0);
}

__global__
void equalization(unsigned char *img, float *cdf, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int cdf_min = cdf[0];
  if(idx < size) img[idx] = correct_color(cdf[img[idx]], cdf_min);
}

int main(int argc, char **argv) {
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
  float *deviceInputImageData;  // input image in rgb of floats on GPU
  float *deviceOutputImageData; // output image in rgb of floats on GPU
  unsigned char *deviceImageChar; // input image in rgb of bytes on GPU
  unsigned char *deviceGrayImage;      // input image in grayscale of bytes on GPU
  unsigned int *histogram;      // 256-bin histogram on GPU
  float *cdf;   // 256-bin CDF of histogram

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

  hostInputImageData   = wbImage_getData(inputImage);
  hostOutputImageData  = wbImage_getData(inputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  // allocate memory on GPU and data transfer
  int imageSize = imageWidth*imageHeight*imageChannels;
  int imageLen  = imageWidth*imageHeight;
  cudaMalloc((void**)&deviceInputImageData,  imageSize*sizeof(float));
  cudaMalloc((void**)&deviceOutputImageData, imageSize*sizeof(float));
  cudaMalloc((void**)&deviceImageChar,  imageSize*sizeof(unsigned char));
  cudaMalloc((void**)&deviceGrayImage,       imageLen*sizeof(unsigned char));
  cudaMalloc((void**)&histogram,             HIST_LEN*sizeof(unsigned int));
  cudaMalloc((void**)&cdf,                   HIST_LEN*sizeof(float));
  cudaMemset(histogram, 0, HIST_LEN*sizeof(unsigned int));
  cudaMemset(cdf, 0,  HIST_LEN*sizeof(float));
  cudaMemcpy(deviceInputImageData, hostInputImageData, imageSize*sizeof(float), cudaMemcpyHostToDevice);

  //@@ insert code here
  // specify kernel dimension
  dim3 blockDimCast(BLOCK_SIZE, 1, 1);
  dim3 gridDimCast(ceil(imageSize/(1.0*BLOCK_SIZE)), 1, 1);
  dim3 blockDimGray(BLOCK_SIZE, 1, 1);
  dim3 gridDimGray(ceil(imageLen/(1.0*BLOCK_SIZE)), 1, 1);
  dim3 blockDimHisto(HIST_LEN, 1, 1);
  dim3 gridDImHisto(ceil(imageLen/(1.0*HIST_LEN)), 1, 1);
  dim3 blockDimCDF(HALF_HIST_LEN, 1, 1);
  dim3 gridDimCDF(1, 1, 1);
  dim3 blockDimEq(BLOCK_SIZE, 1, 1);
  dim3 gridDimEq(ceil(imageSize/(1.0*BLOCK_SIZE)), 1, 1);

  // launch kernels
  imageCast<<<gridDimCast, blockDimCast>>>(deviceInputImageData, deviceImageChar, imageSize);
  cudaDeviceSynchronize();
  grayScale<<<gridDimGray, blockDimGray>>>(deviceImageChar, deviceGrayImage, imageLen);
  cudaDeviceSynchronize();
  histo<<<gridDImHisto, blockDimHisto>>>(deviceGrayImage, histogram, imageLen);
  cudaDeviceSynchronize(); 
  scan_cdf<<<gridDimCDF, blockDimCDF>>>(histogram, cdf, imageLen);
  cudaDeviceSynchronize();
  equalization<<<gridDimEq, blockDimEq>>>(deviceImageChar, cdf, imageSize);
  cudaDeviceSynchronize();
  imageCastReverse<<<gridDimCast, blockDimCast>>>(deviceImageChar, deviceOutputImageData, imageSize);
  cudaDeviceSynchronize();
  
  // transfer data back to CPU
  unsigned char *hostImageChar =(unsigned char*)malloc(sizeof(char)*imageSize);
  cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageSize*sizeof(float), cudaMemcpyDeviceToHost);

  // free GPU memory
  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);
  cudaFree(deviceImageChar);
  cudaFree(deviceGrayImage);
  cudaFree(histogram);
  cudaFree(cdf);
  
  outputImage->data = hostOutputImageData;
  wbSolution(args, outputImage);

  return 0;
}