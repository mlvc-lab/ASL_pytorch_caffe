#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>


template <typename scalar_t>
__inline__ __device__ void myAtomicAdd(scalar_t *buf, scalar_t val);

template <>
__inline__ __device__ void myAtomicAdd<float>(float *buf, float val)
{
    atomicAdd(buf, val);
}

template <>
__inline__ __device__ void myAtomicAdd<double>(double *buf, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)buf;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val +
                    __longlong_as_double(assumed)));

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    //return __longlong_as_double(old);
}

template <typename scalar_t>
__global__ void asl_cuda_forward_kernel(
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ shift_param,
        scalar_t* __restrict__ output,
        const int numB, 
        const int numC, 
        const int numW) {

    const int ch_dim = numW*numW;
    const int batch_dim = numC*ch_dim;
    
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    //const int index = blockIdx.y * blockDim.x * ((batch_dim + blockDim.x - 1) / blockDim.x) + column;
    const int index = blockIdx.y * batch_dim + column;
    if (column < batch_dim) {

        const int batch = index / batch_dim;
        const int ch = (index % batch_dim) / (ch_dim);
        const int h = ((index % batch_dim) % (ch_dim)) / numW;
        const int w = ((index % batch_dim) % (ch_dim)) % numW;

//        int batch = blockIdx.y;
//        int ch = column / ch_dim;
//        int h = (column % ch_dim) / numW;
//        int w = (column % ch_dim) % numW;

        const scalar_t a = shift_param[ch * 2];
        const scalar_t b = shift_param[ch * 2 + 1];

        float a1_f = floorf(a);
        float b1_f = floorf(b);
        float da = a - a1_f;
        float db = b - b1_f;

        int a1 = float2int(a1_f);
        int b1 = float2int(b1_f);
        
        scalar_t z00 = 0.0;
        scalar_t z01 = 0.0;
        scalar_t z10 = 0.0;
        scalar_t z11 = 0.0;
        if (!(h+a1 < 0 || w+b1 < 0 || h+a1 >= numW || w+b1 >= numW)) {
            z00 = input[batch * batch_dim + ch * ch_dim + (h+a1) * numW + (w+b1)];
        } 
        if (!(h+a1 < 0 || w+b1+1 < 0 || h+a1 >= numW || w+b1+1 >= numW)) {
            z01 = input[batch * batch_dim + ch * ch_dim + (h+a1) * numW + (w+b1+1)];
        }
        if (!(h+a1+1 < 0 || w+b1 < 0 || h+a1+1 >= numW || w+b1 >= numW)) {
            z10 = input[batch * batch_dim + ch * ch_dim + (h+a1+1) * numW + (w+b1)];
        }
        if (!(h+a1+1 < 0 || w+b1+1 < 0 || h+a1+1 >= numW || w+b1+1 >= numW)) {
            z11 = input[batch * batch_dim + ch * ch_dim + (h+a1+1) * numW + (w+b1+1)];
        }

        output[index] = \
            z00 * (1-da) * (1-db) + z10 * da * (1-db) + \
            z01 * (1-da) * db + z11 * da * db;

    } // endif

    /*
    const int ch_dim = numW*numW;
    const int batch_dim = numC*ch_dim;
    
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    while (index < numB * batch_dim) {

        int batch = index / batch_dim ;
        int ch = (index % batch_dim) / (ch_dim);
        int h = ((index % batch_dim) % (ch_dim)) / numW;
        int w = ((index % batch_dim) % (ch_dim)) % numW;

        const float a = shift_param[ch * 2];
        const float b = shift_param[ch * 2 + 1];

        float a1_f = floorf(a);
        float b1_f = floorf(b);
        float da = a - a1_f;
        float db = b - b1_f;

        int a1 = float2int(a1_f);
        int b1 = float2int(b1_f);
        
        scalar_t z00 = 0.0;
        scalar_t z01 = 0.0;
        scalar_t z10 = 0.0;
        scalar_t z11 = 0.0;
        if (!(h+a1 < 0 || w+b1 < 0 || h+a1 >= numW || w+b1 >= numW)) {
            z00 = input[batch * batch_dim + ch * ch_dim + (h+a1) * numW + (w+b1)];
        } 
        if (!(h+a1 < 0 || w+b1+1 < 0 || h+a1 >= numW || w+b1+1 >= numW)) {
            z01 = input[batch * batch_dim + ch * ch_dim + (h+a1) * numW + (w+b1+1)];
        }
        if (!(h+a1+1 < 0 || w+b1 < 0 || h+a1+1 >= numW || w+b1 >= numW)) {
            z10 = input[batch * batch_dim + ch * ch_dim + (h+a1+1) * numW + (w+b1)];
        }
        if (!(h+a1+1 < 0 || w+b1+1 < 0 || h+a1+1 >= numW || w+b1+1 >= numW)) {
            z11 = input[batch * batch_dim + ch * ch_dim + (h+a1+1) * numW + (w+b1+1)];
        }

        output[batch * batch_dim + ch * ch_dim + h * numW + w] = \
            z00 * (1-da) * (1-db) + z10 * da * (1-db) + \
            z01 * (1-da) * db + z11 * da * db;

        index += blockDim.x * gridDim.x;

    } // while
    */
}

template <typename scalar_t>
__global__ void asl_cuda_backward_input_kernel(
        const scalar_t* __restrict__ grad_output,
        const scalar_t* __restrict__ shift_param,
        scalar_t* __restrict__ grad_input,
        const int numB, 
        const int numC, 
        const int numW) {

    const int ch_dim = numW*numW;
    const int batch_dim = numC*ch_dim;

    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int index = blockIdx.y * batch_dim + column;
    
    if (column < batch_dim) {
        const int batch = index / batch_dim;
        const int ch = (index % batch_dim) / (ch_dim);
        const int h = ((index % batch_dim) % (ch_dim)) / numW;
        const int w = ((index % batch_dim) % (ch_dim)) % numW;

        const scalar_t a = shift_param[ch * 2];
        const scalar_t b = shift_param[ch * 2 + 1];

        const float a1_f = floorf(a);
        const float b1_f = floorf(b);
        const float da = a - a1_f;
        const float db = b - b1_f;

        const int a1 = float2int(a1_f);
        const int b1 = float2int(b1_f);
        
        scalar_t z00 = 0.0;
        scalar_t z01 = 0.0;
        scalar_t z10 = 0.0;
        scalar_t z11 = 0.0;

        if (!(h-a1 < 0 || w-b1 < 0 || h-a1 >= numW || w-b1 >= numW)) {
            z00 = grad_output[batch * batch_dim + ch * ch_dim + (h-a1) * numW + (w-b1)];
        } 
        if (!(h-a1 < 0 || w-b1-1 < 0 || h-a1 >= numW || w-b1-1 >= numW)) {
            z01 = grad_output[batch * batch_dim + ch * ch_dim + (h-a1) * numW + (w-b1-1)];
        }
        if (!(h-a1-1 < 0 || w-b1 < 0 || h-a1-1 >= numW || w-b1 >= numW)) {
            z10 = grad_output[batch * batch_dim + ch * ch_dim + (h-a1-1) * numW + (w-b1)];
        }
        if (!(h-a1-1 < 0 || w-b1-1 < 0 || h-a1-1 >= numW || w-b1-1 >= numW)) {
            z11 = grad_output[batch * batch_dim + ch * ch_dim + (h-a1-1) * numW + (w-b1-1)];
        }

        grad_input[index] = \
            z00 * (1-da) * (1-db) + z10 * da * (1-db) + \
            z01 * (1-da) * db + z11 * da * db;
    } // if

    /*
    const int ch_dim = numW*numW;
    const int batch_dim = numC*ch_dim;
    
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    while (index < numB * batch_dim) {

        int batch = index / batch_dim ;
        int ch = (index % batch_dim) / (ch_dim);
        int h = ((index % batch_dim) % (ch_dim)) / numW;
        int w = ((index % batch_dim) % (ch_dim)) % numW;

        const float a = shift_param[ch * 2];
        const float b = shift_param[ch * 2 + 1];

        float a1_f = floorf(a);
        float b1_f = floorf(b);
        float da = a - a1_f;
        float db = b - b1_f;

        int a1 = float2int(a1_f);
        int b1 = float2int(b1_f);
        
        scalar_t z00 = 0.0;
        scalar_t z01 = 0.0;
        scalar_t z10 = 0.0;
        scalar_t z11 = 0.0;

        if (!(h-a1 < 0 || w-b1 < 0 || h-a1 >= numW || w-b1 >= numW)) {
            z00 = grad_output[batch * batch_dim + ch * ch_dim + (h-a1) * numW + (w-b1)];
        } 
        if (!(h-a1 < 0 || w-b1-1 < 0 || h-a1 >= numW || w-b1-1 >= numW)) {
            z01 = grad_output[batch * batch_dim + ch * ch_dim + (h-a1) * numW + (w-b1-1)];
        }
        if (!(h-a1-1 < 0 || w-b1 < 0 || h-a1-1 >= numW || w-b1 >= numW)) {
            z10 = grad_output[batch * batch_dim + ch * ch_dim + (h-a1-1) * numW + (w-b1)];
        }
        if (!(h-a1-1 < 0 || w-b1-1 < 0 || h-a1-1 >= numW || w-b1-1 >= numW)) {
            z11 = grad_output[batch * batch_dim + ch * ch_dim + (h-a1-1) * numW + (w-b1-1)];
        }

        grad_input[batch * batch_dim + ch * ch_dim + h * numW + w] = \
            z00 * (1-da) * (1-db) + z10 * da * (1-db) + \
            z01 * (1-da) * db + z11 * da * db;

        index += blockDim.x * gridDim.x;
    } // while
    */
}
        
template <typename scalar_t>
__global__ void asl_cuda_backward_weight_kernel(
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ grad_output,
        const scalar_t* __restrict__ shift_param,
        scalar_t* __restrict__ grad_weight,
        const int numB, 
        const int numC, 
        const int numW) {

    const int ch_dim = numW*numW;
    const int batch_dim = numC*ch_dim;

    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int index = blockIdx.y * batch_dim + column;
    
    if (column < batch_dim) {

        const int batch = index / batch_dim;
        const int ch = (index % batch_dim) / (ch_dim);
        const int h = ((index % batch_dim) % (ch_dim)) / numW;
        const int w = ((index % batch_dim) % (ch_dim)) % numW;

        const scalar_t a = shift_param[ch * 2];
        const scalar_t b = shift_param[ch * 2 + 1];

        const float a1_f = floorf(a);
        const float b1_f = floorf(b);
        const float da = a - a1_f;
        const float db = b - b1_f;

        const int a1 = float2int(a1_f);
        const int b1 = float2int(b1_f);
        
        scalar_t z00 = 0.0;
        scalar_t z01 = 0.0;
        scalar_t z10 = 0.0;
        scalar_t z11 = 0.0;

        if (!(h+a1 < 0 || w+b1 < 0 || h+a1 >= numW || w+b1 >= numW)) {
            z00 = input[batch * batch_dim + ch * ch_dim + (h+a1) * numW + (w+b1)];
        } 
        if (!(h+a1 < 0 || w+b1+1 < 0 || h+a1 >= numW || w+b1+1 >= numW)) {
            z01 = input[batch * batch_dim + ch * ch_dim + (h+a1) * numW + (w+b1+1)];
        }
        if (!(h+a1+1 < 0 || w+b1 < 0 || h+a1+1 >= numW || w+b1 >= numW)) {
            z10 = input[batch * batch_dim + ch * ch_dim + (h+a1+1) * numW + (w+b1)];
        }
        if (!(h+a1+1 < 0 || w+b1+1 < 0 || h+a1+1 >= numW || w+b1+1 >= numW)) {
            z11 = input[batch * batch_dim + ch * ch_dim + (h+a1+1) * numW + (w+b1+1)];
        }
        
        const scalar_t diff = grad_output[index];
        myAtomicAdd<scalar_t>(grad_weight + ch * 2, ((1 - db) * (z10 - z00) + db * (z11 - z01)) * diff);
        myAtomicAdd<scalar_t>(grad_weight + ch * 2 + 1, ((1 - da) * (z01 - z00) + da * (z11 - z10)) * diff);

    } // endif

    /*
    const int ch_dim = numW*numW;
    const int batch_dim = numC*ch_dim;
    
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    while (index < numB * batch_dim) {

        int batch = index / batch_dim ;
        int ch = (index % batch_dim) / (ch_dim);
        int h = ((index % batch_dim) % (ch_dim)) / numW;
        int w = ((index % batch_dim) % (ch_dim)) % numW;

        const float a = shift_param[ch * 2];
        const float b = shift_param[ch * 2 + 1];

        float a1_f = floorf(a);
        float b1_f = floorf(b);
        float da = a - a1_f;
        float db = b - b1_f;

        int a1 = float2int(a1_f);
        int b1 = float2int(b1_f);
        
        scalar_t z00 = 0.0;
        scalar_t z01 = 0.0;
        scalar_t z10 = 0.0;
        scalar_t z11 = 0.0;

        if (!(h+a1 < 0 || w+b1 < 0 || h+a1 >= numW || w+b1 >= numW)) {
            z00 = input[batch * batch_dim + ch * ch_dim + (h+a1) * numW + (w+b1)];
        } 
        if (!(h+a1 < 0 || w+b1+1 < 0 || h+a1 >= numW || w+b1+1 >= numW)) {
            z01 = input[batch * batch_dim + ch * ch_dim + (h+a1) * numW + (w+b1+1)];
        }
        if (!(h+a1+1 < 0 || w+b1 < 0 || h+a1+1 >= numW || w+b1 >= numW)) {
            z10 = input[batch * batch_dim + ch * ch_dim + (h+a1+1) * numW + (w+b1)];
        }
        if (!(h+a1+1 < 0 || w+b1+1 < 0 || h+a1+1 >= numW || w+b1+1 >= numW)) {
            z11 = input[batch * batch_dim + ch * ch_dim + (h+a1+1) * numW + (w+b1+1)];
        }
        
        auto diff = grad_output[batch * batch_dim + ch * ch_dim + h * numW + w];
        myAtomicAdd<scalar_t>(grad_weight + ch * 2, ((1 - db) * (z10 - z00) + db * (z11 - z01)) * diff);
        myAtomicAdd<scalar_t>(grad_weight + ch * 2 + 1, ((1 - da) * (z01 - z00) + da * (z11 - z10)) * diff);

        index += blockDim.x * gridDim.x;
    } // while
    */
}

template <typename scalar_t>
__global__ void asl_cuda_backward_weight_normalize_kernel(
        scalar_t* __restrict__ grad_weight,
        const int numB,
        const int numC,
        const int numW ) {

    const int ch_dim = numW*numW;
    const int batch_dim = numC*ch_dim;

    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int index = blockIdx.y * batch_dim + column;

    const int batch = index / batch_dim;
    if (column < batch_dim && batch == 0) {
        const int ch = (index % batch_dim) / (ch_dim);
        
        const scalar_t da = grad_weight[ch * 2];
        const scalar_t db = grad_weight[ch * 2 + 1];
        const scalar_t dr = sqrt(da*da + db*db);

        if (dr != 0) {
            grad_weight[ch * 2] = da / dr / numB;
            grad_weight[ch * 2 + 1] = db / dr / numB;
        } else {
            grad_weight[ch * 2] = da / numB;
            grad_weight[ch * 2 + 1] = db / numB;
        }
    } 

    /*
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    while (index < numC * 2) {
        
        const scalar_t da = grad_weight[index * 2];
        const scalar_t db = grad_weight[index * 2 + 1];
        const scalar_t dr = sqrt(da*da + db*db);

        if (dr != 0) {
            grad_weight[index * 2] = da / dr;
            grad_weight[index * 2 + 1] = db / dr;
        }

        index += blockDim.x * gridDim.x;
    } // while
    */
}

std::vector<torch::Tensor> asl_cuda_forward(
        torch::Tensor input,
        torch::Tensor shift_param) {

    // input: (numB, numC, numH, numW)
    const auto numB = input.size(0);
    const auto numC = input.size(1);
    const auto numH = input.size(2);
    const auto numW = input.size(3);

    auto output = torch::zeros_like(input);

    // Write_data is output.
    const int threads = 1024;
    //const int blocks = (2100000000 + threads - 1) / threads;
    const int batch_dim = numC * numW * numW;
    const dim3 blocks = make_uint3((batch_dim + threads - 1) / threads, numB, 1);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "asl_forward_cuda", ([&] {
        asl_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            shift_param.data<scalar_t>(),
            output.data<scalar_t>(),
            numB,
            numC,
            numW
        );
    }));

    return {output};
}


std::vector<torch::Tensor> asl_cuda_backward(
    torch::Tensor input,
    torch::Tensor grad_output,
    torch::Tensor shift_param) {

    // grad_output: (numB, numC, numH, numW)
    const auto numB = grad_output.size(0);
    const auto numC = grad_output.size(1);
    const auto numH = grad_output.size(2);
    const auto numW = grad_output.size(3);
    const int batch_dim = numC * numW * numW;

    auto grad_input = torch::zeros_like(grad_output);
    auto grad_weight = torch::zeros_like(shift_param);

    // Write_data = grad_input and index = grad_input
    const int threads = 1024;
    //const int blocks = (2100000000 + threads - 1) / threads;
    const dim3 blocks = make_uint3((batch_dim + threads - 1) / threads, numB, 1);

    AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "asl_backward_cuda_input", ([&] {
        asl_cuda_backward_input_kernel<scalar_t><<<blocks, threads>>>(
            grad_output.data<scalar_t>(),
            shift_param.data<scalar_t>(),
            grad_input.data<scalar_t>(),
            numB,
            numC,
            numW
        );
    }));

    // Write_data = grad_weight and index = grad_output
    const int threads_2 = 1024;
    const dim3 blocks_2 = make_uint3((batch_dim + threads_2 - 1) / threads_2, numB, 1);

    AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "asl_backward_cuda_weight", ([&] {
        asl_cuda_backward_weight_kernel<scalar_t><<<blocks_2, threads_2>>>(
            input.data<scalar_t>(),
            grad_output.data<scalar_t>(),
            shift_param.data<scalar_t>(),
            grad_weight.data<scalar_t>(),
            numB,
            numC,
            numW
        );
    }));

    // Vector Normalize (v / |v|)
    // Write_data is grad_weight.
    //const int threads_3 = 1024;
    //const int blocks_3 = (numC * 2 + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "asl_backward_cuda_weight_normalize", ([&] {
        asl_cuda_backward_weight_normalize_kernel<scalar_t><<<blocks_2, threads_2>>>(
            grad_weight.data<scalar_t>(),
            numB,
            numC,
            numW
        );
    }));

    return {grad_input, grad_weight};
}
