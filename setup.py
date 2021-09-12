from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='torchex',
    packages=find_packages(),
    version='0.1.0',
    author='Lve Fan',
    ext_modules=[
        CUDAExtension(
            'sparse_roi_voxelization', 
            ['./torchex/sparse_roi_voxelization/sparse_roiaware_pool3d.cpp',
             './torchex/sparse_roi_voxelization/sparse_roiaware_pool3d_kernel.cu',]
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)