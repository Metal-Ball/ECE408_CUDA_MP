// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// scan over each element in a block
__global__ void scan(float *input, float *output, float *blockSum, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  
  // step 1: each thread load two input elements into shared memory
  __shared__ float buffer[2*BLOCK_SIZE];
  int start = 2 * BLOCK_SIZE * blockIdx.x;
  int tx = threadIdx.x;
  // each thread loads two elements
  int first = start+tx;
  int second = first + BLOCK_SIZE;
  if(first  < len) buffer[tx] = input[first];
  else buffer[tx] = 0;
  if(second < len) buffer[tx + BLOCK_SIZE] = input[second];
  else buffer[tx + BLOCK_SIZE] = 0;

  // step 2: stride 1 to BLOCK_SIZE
  int stride = 1;
  while(stride < 2 * BLOCK_SIZE){
    // finish reading for first iteration
    // or finish writing to buffer
    __syncthreads();
    // threadIdx => buffer index
    int idx = (tx+1)*stride*2-1;
    if(idx < 2*BLOCK_SIZE && idx>=stride)
      // reduction
      buffer[idx] += buffer[idx-stride];
    // update stride
    stride *= 2;
  }
  
  // step 3: stide BLOCK_SIZE to 1
  stride = BLOCK_SIZE/2;
  while(stride > 0){
    // wait for all threads to finish writing
    __syncthreads();
    // threadIdx => buffer index
    int idx = (tx+1)*stride*2-1;
    if(idx + stride < 2*BLOCK_SIZE)
      //reduction
      buffer[idx+stride] += buffer[idx];
    stride /= 2;
  }
  
  // step 4: write sum of block into blockSum
  __syncthreads();
  int bx = blockIdx.x;
  if(!tx) blockSum[bx] = buffer[2*BLOCK_SIZE - 1];
  
  // step 5: copy shared memory back to output(intermediate result)
  if(first  < len) output[first]  = buffer[tx];
  if(second < len) output[second] = buffer[tx + BLOCK_SIZE];
}

// scan over the sum of each block
__global__ void scan_block(float *blockSum, int len) {
  // step 1: each thread load two input elements into shared memory
  __shared__ float buffer[2*BLOCK_SIZE];
  int start = 2 * BLOCK_SIZE * blockIdx.x;
  int tx = threadIdx.x;
  // each thread loads two elements
  int first = start+tx;
  int second = first + BLOCK_SIZE;
  if(first  < len) buffer[tx] = blockSum[first];
  else buffer[tx] = 0;
  if(second < len) buffer[tx + BLOCK_SIZE] = blockSum[second];
  else buffer[tx + BLOCK_SIZE] = 0;

  // step 2: stride 1 to BLOCK_SIZE
  int stride = 1;
  while(stride < 2 * BLOCK_SIZE){
    // finish reading for first iteration
    // or finish writing to buffer
    __syncthreads();
    // threadIdx => buffer index
    int idx = (tx+1)*stride*2-1;
    if(idx < 2*BLOCK_SIZE && idx>=stride)
      // reduction
      buffer[idx] += buffer[idx-stride];
    // update stride
    stride *= 2;
  }
  
  // step 3: stide BLOCK_SIZE to 1
  stride = BLOCK_SIZE/2;
  while(stride > 0){
    // wait for all threads to finish writing
    __syncthreads();
    // threadIdx => buffer index
    int idx = (tx+1)*stride*2-1;
    if(idx + stride < 2*BLOCK_SIZE)
      //reduction
      buffer[idx+stride] += buffer[idx];
    stride /= 2;
  }
  
  // step 4: copy shared memory back to blockSum
  if(first  < len) blockSum[first]  = buffer[tx];
  if(second < len) blockSum[second] = buffer[tx + BLOCK_SIZE];
}

// add offset to blocks other than blockIdx.x==0
__global__ void add(float *input, float *blockScan, int len) {
  int bx = blockIdx.x;
  int tx = threadIdx.x;
  int first  = bx*BLOCK_SIZE*2 + tx;
  int second = first + BLOCK_SIZE;
  int sum_offset = (bx>0 ? blockScan[bx-1]:0);
  if(first < len)
    input[first]  += sum_offset;
  if(second < len)
    input[second] += sum_offset;
}



int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;    // The input 1D list
  float *hostOutput;   // The output list
  float *deviceBlockSum; // list of sum of all elements of each block
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
  wbCheck(cudaMalloc((void **)&deviceInput,  numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  
  // allocate GPU memory for numBlockSum
  int numBlockSum = numElements / (BLOCK_SIZE << 1); 
  if (numElements % (BLOCK_SIZE << 1)) {
    numBlockSum++;
  }
  wbCheck(cudaMalloc((void **)&deviceBlockSum, numBlockSum * sizeof(float)));

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  int gridSize = ceil(numElements/float(BLOCK_SIZE * 2));
  dim3 scan_dimBlock(BLOCK_SIZE, 1, 1);
  dim3 scan_dimGrid(gridSize, 1, 1);
  dim3 scan_block_dimBlock(BLOCK_SIZE, 1, 1);
  dim3 scan_block_dimGrid(ceil(gridSize/float(BLOCK_SIZE * 2)), 1, 1);
  dim3 add_dimBlock(BLOCK_SIZE, 1, 1);
  dim3 add_dimGrid(gridSize, 1, 1);
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<scan_dimGrid, scan_dimBlock>>>(deviceInput, deviceOutput, deviceBlockSum, numElements);
  cudaDeviceSynchronize();
  scan_block<<<scan_block_dimGrid, scan_block_dimBlock>>>(deviceBlockSum,gridSize);
  cudaDeviceSynchronize();
  add<<<add_dimGrid, add_dimBlock>>>(deviceOutput, deviceBlockSum,numElements);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(deviceBlockSum);
  wbTime_stop(GPU, "Freeing GPU Memory");
  
  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);
  return 0;
}

