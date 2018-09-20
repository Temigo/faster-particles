#include "crop_op.h"
#include <iostream>

using namespace tensorflow;

// Register TF operation
REGISTER_OP("Crop")
    .Attr("T: {float, int32} = DT_FLOAT")
    .Input("image: float32")
    .Input("crop_centers: int32")
    .Input("crop_size: int32")
    .Output("crops: float32");
    /*.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(1));
      return Status::OK();
    });*/

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

// CPU specialization of actual computation.
template <typename T>
struct CropFunctor<CPUDevice, T> {
  void operator()(
    const CPUDevice& d,
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
    /*for (int crop_idx = 0; crop_idx < num_crops; ++crop_idx) {
      const int center_x = crop_centers_ptr[crop_idx*3];
    }*/
  }
};

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class CropOp : public OpKernel {
 public:
  explicit CropOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensors
    const Tensor& image = context->input(0);
    const Tensor& crop_centers = context->input(1);
    const Tensor& crop_size_tensor = context->input(2);
    // FIXME
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(crop_size_tensor.shape()), errors::InvalidArgument("crop_size must be scalar, has shape ", crop_size_tensor.shape().DebugString()));
    //const int crop_size = crop_size_tensor.scalar<int32>()();
    const int crop_size = 64;

    // Basic shape checks on input image and crop centers
    // OP_REQUIRES(context, image.dims() == 4, errors::InvalidArgument("Input image must be 4-D", image.shape().DebugString()));
    // //OP_REQUIRES(context, image.dim_size(0) == image.dim_size(1) == image.dim_size(2), errors::InvalidArgument("Expected square input tensor."));
    // OP_REQUIRES(context, crop_centers.dims() == 2, errors::InvalidArgument("Expected shape of size 2 for crop centers."));
    // OP_REQUIRES(context, crop_size <= image.dim_size(0), errors::InvalidArgument("Crops size must be less or equal than image size."));
    // OP_REQUIRES(context, crop_centers.dim_size(1) == image.dims() - 1, errors::InvalidArgument("Crop centers dimensions must match image dimensions."));

    // Get shapes of input tensors
    const TensorShape& image_shape = image.shape();
    const TensorShape& crop_centers_shape = crop_centers.shape();
    int image_size = image_shape.dim_size(1);
    int channels = image_shape.dim_size(3);
    int num_crops = crop_centers_shape.dim_size(0);
    int dim = crop_centers_shape.dim_size(1);

    // Create an output tensor
    Tensor* crops = NULL;
    // create output shape
    TensorShape crops_shape;
    crops_shape.AddDim(num_crops);
    crops_shape.AddDim(crop_size);
    crops_shape.AddDim(crop_size);
    crops_shape.AddDim(crop_size);
    crops_shape.AddDim(channels);
    OP_REQUIRES_OK(context, context->allocate_output(0, crops_shape,
                                                     &crops));

    // Do the computation.
    // OP_REQUIRES(context, image.NumElements() <= tensorflow::kint32max,
    //             errors::InvalidArgument("Too many elements in input tensor"));
    // OP_REQUIRES(context, crop_centers.NumElements() <= tensorflow::kint32max,
    //             errors::InvalidArgument("Too many elements in crop centers tensor"));

    CropFunctor<Device, T>()(
        context->eigen_device<Device>(),
        image.flat<T>().data(),
        crop_centers.flat<int>().data(),
        crop_size,
        image_size,
        channels,
        num_crops,
        crops->flat<T>().data()
      );

  }
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Crop") \
      .Device(DEVICE_CPU) \
      .TypeConstraint<T>("T"), \
    CropOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(int32);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T) \
  extern template struct CropFunctor<GPUDevice, T>; \
  REGISTER_KERNEL_BUILDER( \
      Name("Crop")      \
      .Device(DEVICE_GPU)   \
      .TypeConstraint<T>("T"),  \
    CropOp<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(int32);
#endif // GOOGLE_CUDA
