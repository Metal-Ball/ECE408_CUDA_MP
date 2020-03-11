#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)


//@@ Define any useful program-wide constants here
# define TILE_WIDTH 6
# define MASK_WIDTH 3

//@@ Define constant memory for device kernel here

__constant__ float mask[MASK_WIDTH*MASK_WIDTH*MASK_WIDTH];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  __shared__ float tile [TILE_WIDTH+MASK_WIDTH-1][TILE_WIDTH+MASK_WIDTH-1][TILE_WIDTH+MASK_WIDTH-1];

  int x_o = blockIdx.x * TILE_WIDTH + threadIdx.x;
  int y_o = blockIdx.y * TILE_WIDTH + threadIdx.y;
  int z_o = blockIdx.z * TILE_WIDTH + threadIdx.z;

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;

  int x_i = x_o - MASK_WIDTH/2;
  int y_i = y_o - MASK_WIDTH/2;
  int z_i = z_o - MASK_WIDTH/2;

  // all threads load one element
  if (z_i >= 0 && z_i < z_size
      && y_i >= 0 && y_i < y_size
      && x_i >= 0 && x_i < x_size)
  {
      tile[tz][ty][tx] = input[z_i*y_size*x_size + y_i*x_size + x_i];
  }
  else
  {
      tile[tz][ty][tx] = 0.0f;
  }

   __syncthreads();
   // some thread computes one element
  if(tz<TILE_WIDTH && ty<TILE_WIDTH && tx<TILE_WIDTH){
    float acc = 0.0f;
    for(int i =0; i<MASK_WIDTH; i++)
    {
      for(int j =0; j<MASK_WIDTH; j++)
      {
        for(int k =0; k<MASK_WIDTH; k++)
        {
          acc += tile[i+tz][j+ty][k+tx] * mask[i*MASK_WIDTH*MASK_WIDTH+j*MASK_WIDTH+k];
        }
      }
    }
    if(x_o < x_size && y_o < y_size && z_o < z_size)
       output[z_o*y_size*x_size + y_o*x_size + x_o] = acc;
   }
}



void printResult(float *result, int length){
  for(int i =0; i<length;i++){
    printf("%lf",result[i]);
    printf(", ");
  }

}


int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  int size = z_size*y_size*x_size;
  cudaMalloc((void**) &deviceInput,  size*sizeof(float));
  cudaMalloc((void**) &deviceOutput, size*sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(deviceInput, hostInput+3, size*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(mask, hostKernel, kernelLength * sizeof(float));
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 dimBlock(TILE_WIDTH+MASK_WIDTH-1, TILE_WIDTH+MASK_WIDTH-1, TILE_WIDTH+MASK_WIDTH-1);
  dim3 dimGrid(ceil(x_size/(float)(TILE_WIDTH)), ceil(y_size/(float)(TILE_WIDTH)), ceil(z_size/(float)(TILE_WIDTH)));
  //@@ Launch the GPU kernel here
  conv3d<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");

  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(hostOutput+3, deviceOutput, size*sizeof(float),cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;

  //printResult(hostOutput, inputLength);
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
