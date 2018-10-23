/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
#include <iostream>
#include "loadImage.h"
#include "../util/cuda/cudaMappedMemory.h"

#include <QImage>
#include <opencv2/core/mat.hpp>


// loadImageRGBA
//bool loadImageRGBA( const char* filename, float4** cpu, float4** gpu, int* width, int* height )
//{
//	if( !filename || !cpu || !gpu || !width || !height )
//	{
//		printf("loadImageRGBA - invalid parameter\n");
//		return false;
//	}
//
//	// load original image
//	QImage qImg;
//
//	if( !qImg.load(filename) )
//	{
//		printf("failed to load image %s\n", filename);
//		return false;
//	}
//
//	if( *width != 0 && *height != 0 )
//		qImg = qImg.scaled(*width, *height, Qt::IgnoreAspectRatio);
//
//	const uint32_t imgWidth  = qImg.width();
//	const uint32_t imgHeight = qImg.height();
//	const uint32_t imgPixels = imgWidth * imgHeight;
//	const size_t   imgSize   = imgWidth * imgHeight * sizeof(float) * 4;
//
//	printf("loaded image  %s  (%u x %u)  %zu bytes\n", filename, imgWidth, imgHeight, imgSize);
//
//	// allocate buffer for the image
//	if( !cudaAllocMapped((void**)cpu, (void**)gpu, imgSize) )
//	{
//		printf(LOG_CUDA "failed to allocated %zu bytes for image %s\n", imgSize, filename);
//		return false;
//	}
//
//	float4* cpuPtr = *cpu;
//
//	for( uint32_t y=0; y < imgHeight; y++ )
//	{
//		for( uint32_t x=0; x < imgWidth; x++ )
//		{
//			const QRgb rgb  = qImg.pixel(x,y);
//			const float4 px = make_float4(float(qRed(rgb)),
//										  float(qGreen(rgb)),
//										  float(qBlue(rgb)),
//										  float(qAlpha(rgb)));
//
//			cpuPtr[y*imgWidth+x] = px;
//		}
//	}
//
//	*width  = imgWidth;
//	*height = imgHeight;
//	return true;
//}
//
//

// loadImageRGB
bool loadImageRGB( const char* filename, float3** cpu, float3** gpu, int* width, int* height, const float3& mean )
{
    if( !filename || !cpu || !gpu || !width || !height )
    {
        printf("loadImageRGB - invalid parameter\n");
        return false;
    }

    // load original image
    QImage qImg;

    if( !qImg.load(filename) )
    {
        printf("failed to load image %s\n", filename);
        return false;
    }

    if( *width != 0 && *height != 0 )
        qImg = qImg.scaled(*width, *height, Qt::IgnoreAspectRatio);

    const uint32_t imgWidth  = qImg.width();
    const uint32_t imgHeight = qImg.height();
    const uint32_t imgPixels = imgWidth * imgHeight;
    const size_t   imgSize   = imgWidth * imgHeight * sizeof(float) * 3;

    printf("loaded image  %s  (%u x %u)  %zu bytes\n", filename, imgWidth, imgHeight, imgSize);

    // allocate buffer for the image
    if( !cudaAllocMapped((void**)cpu, (void**)gpu, imgSize) )
    {
        printf(LOG_CUDA "failed to allocated %zu bytes for image %s\n", imgSize, filename);
        return false;
    }

    float* cpuPtr = (float*)*cpu;

    for( uint32_t y=0; y < imgHeight; y++ )
    {
        for( uint32_t x=0; x < imgWidth; x++ )
        {
            const QRgb rgb  = qImg.pixel(x,y);
            const float mul = 0.007843f; 	//1.0f / 255.0f;
            const float3 px = make_float3((float(qRed(rgb))   - mean.x) * mul,
                                          (float(qGreen(rgb)) - mean.y) * mul,
                                          (float(qBlue(rgb))  - mean.z) * mul );

            // note:  caffe/GIE is band-sequential (as opposed to the typical Band Interleaved by Pixel)
            cpuPtr[imgPixels * 0 + y * imgWidth + x] = px.x;
            cpuPtr[imgPixels * 1 + y * imgWidth + x] = px.y;
            cpuPtr[imgPixels * 2 + y * imgWidth + x] = px.z;
        }
    }

    *width  = imgWidth;
    *height = imgHeight;
    return true;
}

bool    loadImageBGR( cv::Mat frame, float3** cpu, float3** gpu, int* width, int* height, const float3& mean )
{
	const uint32_t imgWidth  = 300;
	const uint32_t imgHeight = 300;
	const uint32_t imgPixels = imgWidth * imgHeight;
	const size_t   imgSize   = imgWidth * imgHeight * sizeof(float) * 3;

	// allocate buffer for the image
	if( !cudaAllocMapped((void**)cpu, (void**)gpu, imgSize) )
	{
		printf(LOG_CUDA "failed to allocated bytes for image");
		return false;
	}

	float* cpuPtr = (float*)*cpu;

	for( uint32_t y=0; y < imgHeight; y++ )
	{
		for( uint32_t x=0; x < imgWidth; x++ )
		{
      cv::Vec3b intensity = frame.at<cv::Vec3b>(y,x);
			cpuPtr[imgPixels * 0 + y * imgWidth + x] = (float)intensity.val[0];
			cpuPtr[imgPixels * 1 + y * imgWidth + x] = (float)intensity.val[1];
			cpuPtr[imgPixels * 2 + y * imgWidth + x] = (float)intensity.val[2];
		}
	}
	return true;
}
