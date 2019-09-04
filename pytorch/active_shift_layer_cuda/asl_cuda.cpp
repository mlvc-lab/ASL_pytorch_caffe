#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> asl_cuda_forward(
    torch::Tensor input,
    torch::Tensor shift_param);

std::vector<torch::Tensor> asl_cuda_backward(
    torch::Tensor input,
    torch::Tensor grad_output,
    torch::Tensor shift_param);

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> asl_forward(
        torch::Tensor input,
        torch::Tensor shift_param) {
    CHECK_INPUT(input);
    CHECK_INPUT(shift_param);

    return asl_cuda_forward(input, shift_param);
}

std::vector<torch::Tensor> asl_backward(
        torch::Tensor input,
        torch::Tensor grad_output,
        torch::Tensor shift_param) {
    CHECK_INPUT(input);
    CHECK_INPUT(grad_output);
    CHECK_INPUT(shift_param);

    return asl_cuda_backward(input, grad_output, shift_param);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &asl_forward, "ASL forward (CUDA)");
    m.def("backward", &asl_backward, "ASL backward (CUDA)"); 
}
