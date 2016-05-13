import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import pycuda.gpuarray as gpuarray

from PIL import Image
import scipy as sp
import numpy as np
import string
import time

from pycuda.compiler import SourceModule

# -- Python interface
def convolve_gpu(sourceImage, convFilter, convType):
    """
    convType is the same as in:
    http://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve.html#scipy.signal.convolve
    """
    
    # Cuda C code
    template = """

    #define FILTER_W $FILTER_W
    #define FILTER_H $FILTER_H
    
    #include <stdio.h>
    
    __device__ __constant__ float d_Kernel_filter[FILTER_H*FILTER_W];

    __global__ void ConvolutionKernel(
                float* img, int imgW, int imgH,
                float* out
                )
    {       
        const int nThreads = blockDim.x * gridDim.x;            
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;      
        
        const int outW = imgW - FILTER_W + 1;
        const int outH = imgH - FILTER_H + 1;
        
        const int nPixels = outW * outH;            

        for(int curPixel = idx; curPixel < nPixels; curPixel += nThreads) 
        {
            int x = curPixel % outW;
            int y = curPixel / outW;                    
            float sum = 0;
            for (int filtY = 0; filtY < FILTER_H; filtY++)
                for (int filtX = 0; filtX < FILTER_W; filtX++)
                    {
                    int sx = x + filtX;
                    int sy = y + filtY;
                    sum+= img[sy*imgW + sx] * d_Kernel_filter[filtY*FILTER_W + filtX];
                    }               
            out[y * outW + x] = sum;
        }   
    }
    """
    convFilter = np.flipud(np.fliplr(convFilter))
    (DATA_H,  DATA_W) = sourceImage.shape
    (outH, outW) = (0, 0)
    
    # -- Add zero paddings
    (padWl, padWr, padHt, padHb) = (0, 0, 0, 0) 
    (filtH, filtW) = (convFilter.shape[0], convFilter.shape[1])
    if convType == 'full':
        padWl = filtW-1
        padWr = filtW-1
        padHt = filtH-1
        padHb = filtH-1
        (outH, outW) = (DATA_H+filtH-1, DATA_W+filtW-1)
    elif convType == 'same':
        padWl = filtW/2
        padWr = filtW/2 - (1-filtW%2)
        padHt = filtH/2
        padHb = filtH/2 - (1-filtH%2)        
        (outH, outW) = (DATA_H, DATA_W)
    elif convType == 'valid':
        (outH, outW) = (sourceImage.shape[0]-convFilter.shape[0]+1, sourceImage.shape[1]-convFilter.shape[1]+1)
    
    # -- zero padding
    tmpImg = np.zeros((padHt+DATA_H+padHb, padWl+DATA_W+padWr))
    tmpImg[padHt:padHt+DATA_H, padWl:padWl+DATA_W] = sourceImage
    sourceImage = tmpImg
    (DATA_H,  DATA_W) = sourceImage.shape

    destImage = np.float32(np.zeros((outH, outW)))
    #assert sourceImage.dtype == 'float32',  'source image must be float32'
    #assert convFilter.dtype == 'float32',  'convFilter must be float32'
    
    # -- interface stuff to Cuda C
    template = string.Template(template)
    code = template.substitute(FILTER_H = convFilter.shape[0], FILTER_W = convFilter.shape[1])
    module = SourceModule(code)
    
    # -- change the numpy arrays to row vectors of float32
    sourceImage = np.float32(sourceImage.reshape(sourceImage.size))
    convFilter = np.float32(convFilter.reshape(convFilter.size))
    
    convolutionGPU = module.get_function('ConvolutionKernel')
    d_Kernel_filter = module.get_global('d_Kernel_filter')[0]

    # -- Prepare device arrays
    destImage_gpu = cuda.mem_alloc_like(destImage)
    sourceImage_gpu = cuda.mem_alloc_like(sourceImage)
    cuda.memcpy_htod(sourceImage_gpu, sourceImage)
    
    cuda.memcpy_htod(d_Kernel_filter,  convFilter) # The kernel goes into constant memory via a symbol defined in the kernel

    convolutionGPU(sourceImage_gpu,  np.int32(DATA_W),  np.int32(DATA_H), destImage_gpu,  block=(400,1,1), grid=(1,1))

    # Pull the data back from the GPU.
    cuda.memcpy_dtoh(destImage,  destImage_gpu)
    return destImage
    
def test():
    img=Image.open('lena.jpg').convert('L')
    original = sp.misc.fromimage(img, flatten = False).real
    #Image.fromarray(original).show()

    #filt = np.array([[-1,-1,-1], [-1, 8, -1], [-1, -1, -1]])
    filt = np.random.rand(10,10)/25
    #filt = np.ones((5,5))/25
    tic = time.clock()
    
    print "Hiiiiiii - here is pycuda_convolution!!!"
    res = convolve_gpu(original, filt, 'full')
    toc = time.clock()
    print toc - tic

    print original.shape
    print res.shape

    Image.fromarray(res).show()

    
if __name__ == "__main__":
    test()
