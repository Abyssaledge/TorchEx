from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='torchex',
    packages=find_packages(),
    version='0.1.0',
    author='Lve Fan',
    ext_modules=[
        CUDAExtension(
            'weighted_nms', 
            ['./torchex/src/weighted_nms/wnms.cpp',
             './torchex/src/weighted_nms/wnms_kernel.cu',]
        ),
        CUDAExtension(
            'sparse_roi_voxelization', 
            ['./torchex/src/sparse_roi_voxelization/sparse_roiaware_pool3d.cpp',
             './torchex/src/sparse_roi_voxelization/sparse_roiaware_pool3d_kernel.cu',]
        ),
        CUDAExtension(
            'enlarged_roi_voxelization', 
            ['./torchex/src/enlarged_roi_voxelization/enlarged_roiaware_pool3d.cpp',
             './torchex/src/enlarged_roi_voxelization/enlarged_roiaware_pool3d_kernel.cu',]
        ),
        CUDAExtension(
            'roi_point_voxelization', 
            ['./torchex/src/roi_point_voxelization/roi_point_voxelization.cpp',
             './torchex/src/roi_point_voxelization/roi_point_voxelization_kernel.cu',]
        ),
        CUDAExtension(
            'roi_point_voxelization_v2', 
            ['./torchex/src/roi_point_voxelization_v2/roi_point_voxelization_v2.cpp',
             './torchex/src/roi_point_voxelization_v2/roi_point_voxelization_kernel_v2.cu',]
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)