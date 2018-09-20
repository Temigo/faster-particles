#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

// Register TF operation
REGISTER_OP("Crop")
    .Input("image: float32")
    .Input("crop_centers: int32")
    .Input("crop_size: int32")
    .Output("crops: float32");
    /*.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(1));
      return Status::OK();
    });*/
