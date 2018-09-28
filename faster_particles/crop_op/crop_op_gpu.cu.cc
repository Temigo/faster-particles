#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "crop_op.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

// Define the CUDA kernel.
// Input image is size NxNxNxC
template <typename T>
__global__ void CropCudaKernel(
  const T* image_ptr,
  const int* crop_centers_ptr,
  const int image_size,
  const int channels,
  int crop_size,
  const int num_crops,
  T* crops_ptr
) {
  const int crop_id = blockIdx.x;
  const int center_x = crop_centers_ptr[crop_id*3];
  const int center_y = crop_centers_ptr[1 + 3*crop_id];
  const int center_z = crop_centers_ptr[2 + 3*crop_id];

  for (int id = threadIdx.x; id < crop_size*crop_size*crop_size*channels; id += blockDim.x) {
    // Coordinates inside the crop (0 <= coords < crop_size)
    int id_temp = id;
    const int c = id_temp % channels;
    id_temp /= channels;
    const int z = id_temp % crop_size;
    id_temp /= crop_size;
    const int y = id_temp % crop_size;
    const int x = id_temp / crop_size;

    // Corresponding coordinates in original image
    int image_x = x + (center_x - crop_size / 2);
    int image_y = y + (center_y - crop_size / 2);
    int image_z = z + (center_z - crop_size / 2);
    int img_idx = c + channels * (image_z + image_size * (image_y + image_size * image_x ));
    //int img_idx = image_x + image_size * (image_y + image_size * (image_z + image_size * c));

    if ((img_idx >= image_size * image_size * image_size * channels) || (img_idx < 0)) continue;
    //printf("Image: %d %d %d %d %f", image_x, image_y, image_z, img_idx, image_ptr[img_idx]);
    int crop_idx = c + channels * (z + crop_size * (y + crop_size * (x + crop_size * crop_id)));
    //int crop_idx = crop_id + crop_size * (x + crop_size * (y + crop_size * (z + crop_size * c)));
    crops_ptr[crop_idx] = image_ptr[img_idx];
    //printf("Crop: %d %d %d %d %f", x, y, z, c, crops_ptr[crop_idx]);
  }
}


template <typename T>
__global__ void CropCudaKernel2(
  const T* image_ptr,
  const int* crop_centers_ptr,
  const int image_size,
  const int channels,
  int crop_size,
  const int num_crops,
  T* crops_ptr
) {
  const int crop_id = blockIdx.x/crop_size;
  const int center_x = crop_centers_ptr[crop_id*3];
  const int center_y = crop_centers_ptr[1 + crop_id*3];
  const int center_z = crop_centers_ptr[2 + crop_id*3];
  int offset = (blockIdx.x % crop_size) * crop_size*crop_size*channels;

  // if(threadIdx.x == 0)printf("%d %d %d\n", blockIdx.x, crop_id, offset);
  for (int id = threadIdx.x; id < crop_size*crop_size*channels; id += blockDim.x) {
    // Coordinates inside the crop (0 <= coords < crop_size)
    int id_temp = offset + id;
    const int c = id_temp % channels;
    id_temp /= channels;
    const int z = id_temp % crop_size;
    id_temp /= crop_size;
    const int y = id_temp % crop_size;
    const int x = id_temp / crop_size;

    // Corresponding coordinates in original image
    int image_x = x + (center_x - crop_size / 2);
    int image_y = y + (center_y - crop_size / 2);
    int image_z = z + (center_z - crop_size / 2);
    int img_idx = c + channels * (image_z + image_size * (image_y + image_size * image_x ));

    if ((img_idx >= image_size * image_size * image_size * channels) || (img_idx < 0)) continue;

    int crop_idx = c + channels * (z + crop_size * (y + crop_size * (x + crop_size * crop_id)));
    crops_ptr[crop_idx] = image_ptr[img_idx];
  }
}


// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
void CropFunctor<GPUDevice, T>::operator()(
    const GPUDevice& d,
    /*typename TTypes<T, 4>::ConstTensor image_ptr,
    typename TTypes<int, 2>::ConstTensor crop_centers_ptr,
    int crop_size,
    typename TTypes<T, 5>::Tensor crops_ptr*/
    const T* image_ptr,
    const int* crop_centers_ptr,
    int crop_size,
    int image_size,
    int channels,
    int num_crops,
    T* crops_ptr
  ) {
  // Launch the cuda kernel.
  //
  // See core/util/cuda_kernel_helper.h for example of computing
  // block count and thread_per_block count.
  //int block_count = num_crops;
  // int thread_per_block = 128;
  int block_count = num_crops * crop_size;
  int thread_per_block = 1024;
  CropCudaKernel2<T>
      <<<block_count, thread_per_block, 0, d.stream()>>>(
        image_ptr,
        crop_centers_ptr,
        image_size,
        channels,
        crop_size,
        num_crops,
        crops_ptr
      );
    cudaDeviceSynchronize();
  //return d.ok();
}

// Explicitly instantiate functors for the types of OpKernels registered.
template struct CropFunctor<GPUDevice, float>;
template struct CropFunctor<GPUDevice, int32>;

#endif // GOOGLE_CUDA
