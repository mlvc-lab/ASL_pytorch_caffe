from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='asl_cuda',
    ext_modules=[
        CUDAExtension('asl_cuda', [
            'asl_cuda.cpp',
            'asl_cuda_op.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
