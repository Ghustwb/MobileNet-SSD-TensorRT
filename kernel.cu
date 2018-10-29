/*
 * http://github.com/dusty-nv/jetson-inference
 */

#include "cuda/cudaUtility.h"
#include <iostream>


// gpuPreImageNet
__global__ void gpuPreImageNet( float2 scale, float4* input, int iWidth, float* output, int oWidth, int oHeight )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int n = oWidth * oHeight;
	
	if( x >= oWidth || y >= oHeight )
		return;

	const int dx = ((float)x * scale.x);
	const int dy = ((float)y * scale.y);

	const float4 px  = input[ dy * iWidth + dx ];
	const float3 bgr = make_float3(px.z, px.y, px.x);
	
	output[n * 0 + y * oWidth + x] = bgr.x;
	output[n * 1 + y * oWidth + x] = bgr.y;
	output[n * 2 + y * oWidth + x] = bgr.z;
}

// cudaPreImageNet
cudaError_t cudaPreImageNet( float4* input, size_t inputWidth, size_t inputHeight,
				         float* output, size_t outputWidth, size_t outputHeight )
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( inputWidth == 0 || outputWidth == 0 || inputHeight == 0 || outputHeight == 0 )
		return cudaErrorInvalidValue;

	const float2 scale = make_float2( float(inputWidth) / float(outputWidth),
							    float(inputHeight) / float(outputHeight) );

	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(outputWidth,blockDim.x), iDivUp(outputHeight,blockDim.y));

	gpuPreImageNet<<<gridDim, blockDim>>>(scale, input, inputWidth, output, outputWidth, outputHeight);

	return CUDA(cudaGetLastError());
}

// gpuPreImageNetMean
__global__ void gpuPreImageNetMean( float2 scale, float3* input, int iWidth, float* output, int oWidth, int oHeight, float3 mean_value )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int n = oWidth * oHeight;
	
	if( x >= oWidth || y >= oHeight )
		return;

	const int dx = ((float)x * scale.x);
	const int dy = ((float)y * scale.y);

	const float3 px  = input[ dy * iWidth + dx ];
	const float3 bgr = make_float3(px.z - mean_value.x, px.y - mean_value.y, px.x - mean_value.z);
	
	output[n * 0 + y * oWidth + x] = bgr.x;
	output[n * 1 + y * oWidth + x] = bgr.y;
	output[n * 2 + y * oWidth + x] = bgr.z;
}

// cudaPreImageNetMean
cudaError_t cudaPreImageNetMean( float3* input, size_t inputWidth, size_t inputHeight,
				             float* output, size_t outputWidth, size_t outputHeight, const float3& mean_value )

{
	if( !input || !output ){
        std::cout << "error here. "<< std::endl;
        return cudaErrorInvalidDevicePointer;
    }

	if( inputWidth == 0 || outputWidth == 0 || inputHeight == 0 || outputHeight == 0 ){
        std::cout << "Or here. " << std::endl;
        return cudaErrorInvalidValue;
    }


	const float2 scale = make_float2( float(inputWidth) / float(outputWidth),
							    float(inputHeight) / float(outputHeight) );


	// launch kernel

	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(outputWidth,blockDim.x), iDivUp(outputHeight,blockDim.y));

	gpuPreImageNetMean<<<gridDim, blockDim>>>(scale, input, inputWidth, output, outputWidth, outputHeight, mean_value);

	return CUDA(cudaGetLastError());

}

__global__ void kernel_extract_roi(float* input, float* output, char* mean,
    const int input_w, const int output_w, const int output_h,
    const int in_plane_r, const int in_plane_g, const int in_plane_b,
    const int out_plane_r, const int out_plane_g, const int out_plane_b,
    const int bbox_x, const int bbox_y, const int bbox_w, const int bbox_h)
{
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if( x < output_w && y < output_h)
    {
        float r[2] = { float(x) * bbox_w / output_w + bbox_x,
                       float(y) * bbox_h / output_h + bbox_y };

        int   pos[4][2] = { { int(floor(r[0])), int(floor(r[1])) },
                            { int( ceil(r[0])), int(floor(r[1])) },
                            { int(floor(r[0])),  int(ceil(r[1])) },
                            { int( ceil(r[0])),  int(ceil(r[1])) } };

        float u = r[0]-floor(r[0]);
        float v = r[1]-floor(r[1]);

        float s[4] = { (1-u)*(1-v), u*(1-v), (1-u)*v, u*v };

        int map[4] = { pos[0][1]*input_w + pos[0][0], pos[1][1]*input_w + pos[1][0],
                       pos[2][1]*input_w + pos[2][0], pos[3][1]*input_w + pos[3][0]};

        int idx = y * output_w + x;
        output[idx+out_plane_r] = round( s[0]*input[map[0]+in_plane_r]
                                       + s[1]*input[map[1]+in_plane_r]
                                       + s[2]*input[map[2]+in_plane_r]
                                       + s[3]*input[map[3]+in_plane_r] );// float(mean[idx+out_plane_r]));
        output[idx+out_plane_g] = round( s[0]*input[map[0]+in_plane_g]
                                       + s[1]*input[map[1]+in_plane_g]
                                       + s[2]*input[map[2]+in_plane_g]
                                       + s[3]*input[map[3]+in_plane_g] );//float(mean[idx+out_plane_g]));
        output[idx+out_plane_b] = round( s[0]*input[map[0]+in_plane_b]
                                       + s[1]*input[map[1]+in_plane_b]
                                       + s[2]*input[map[2]+in_plane_b]
                                       + s[3]*input[map[3]+in_plane_b] );//float(mean[idx+out_plane_b]));
    }
}

void convertROI(float* input, float* output, char* mean, const int* srcSize, const int* dstSize, const int* roi, cudaStream_t stream)
{
    int in_plane_r = 0;
    int in_plane_g = srcSize[1] * srcSize[2];
    int in_plane_b = srcSize[1] * srcSize[2] * 2;

    int out_plane_r = 0;
    int out_plane_g = dstSize[1] * dstSize[2];
    int out_plane_b = dstSize[1] * dstSize[2] * 2;

    int bbox_x = min(max(roi[0], 0), srcSize[2]-1);
    int bbox_y = min(max(roi[1], 0), srcSize[1]-1);
    int bbox_w = min(max(roi[2]-roi[0], 0), srcSize[2]-bbox_x-1 );
    int bbox_h = min(max(roi[3]-roi[1], 0), srcSize[1]-bbox_y-1 );

    dim3 dimBlock(32,32);
    dim3 dimGrid(dstSize[2]/dimBlock.x+1, dstSize[1]/dimBlock.y+1);

    std::cout << "ROI: " << bbox_x << " " << bbox_y << " " << bbox_w << " " << bbox_h << std::endl;

    kernel_extract_roi <<< dimGrid, dimBlock, 0, stream >>> (input, output, mean,
                       srcSize[2], dstSize[2], dstSize[1],
                       in_plane_r,   in_plane_g,  in_plane_b,
                       out_plane_r, out_plane_g, out_plane_b,
                       bbox_x, bbox_y, bbox_w, bbox_h);
}


__global__  void kernelSoftmax( float* x, int channels, float* y)
{

	extern __shared__ float mem[];
    __shared__ float sum_value;
	sum_value=0;
	float number = *(x + blockDim.x*blockIdx.x + threadIdx.x);
	float number_exp = __expf(number);

//    sum_value += number_exp ;
    /* *
     * @TODO: Can do with the help of atomicAdd.
     * */
    atomicAdd(&sum_value, number_exp);
    __syncthreads();

//	mem[threadIdx.x] = number_exp;

    /* *
     * @TODO: Can do with the help of a for loop. Try different methods and find the time taken.
     * */
    //	float sum = 0.0f;
    //	for (int i=0;i<channels;i++)
    //	{
    //		sum += mem[i];
    //	}

	y[blockDim.x*blockIdx.x + threadIdx.x] = __fdiv_rd(number_exp, sum_value);

}

void cudaSoftmax(int n, int channels,  float* x, float*y)
{
	kernelSoftmax<<< (n/channels), channels, channels*sizeof(float)>>>( x, channels, y);
	cudaDeviceSynchronize();
}

