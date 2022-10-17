#include <veda/tensorflow/api.h>
#include <tensorflow/core/framework/tensor_util.h>

namespace tensorflow {
//------------------------------------------------------------------------------
template<typename T, int __unused__>
class BroadcastToOp : public OpKernel {
public:
	explicit BroadcastToOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

	void Compute(OpKernelContext* ctx) override {
		using namespace veda::tensorflow; 

		const Tensor& input_tensor = ctx->input(0);
		const TensorShape& input_shape = input_tensor.shape();

		const Tensor& shape_tensor = ctx->input(1);

		TensorShape output_shape;
		OP_REQUIRES_OK(ctx, tensor::MakeShape(shape_tensor, &output_shape));

		// Handle copy.
		if(output_shape == input_shape) {
			ctx->set_output(0, input_tensor);
			return;
		}

		OP_REQUIRES(ctx, input_shape.dims() <= output_shape.dims(), errors::InvalidArgument("Rank of input (", input_shape.dims(), ") must be no greater than rank of output shape (",output_shape.dims(), ")."));

		Tensor* output_tensor = nullptr;
		OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output_tensor));

		// Handle broadcast from Scalar.
		if(input_shape.dims() == 0 || input_tensor.NumElements() == 1) {
			auto d_out		= tf2veda<T>(output_tensor);
			auto d_input	= tf2veda<T>(input_tensor);
			CVEDA(veda_tensors_copy(handle(ctx), &d_out, &d_input));
			return;
		}

		// Check whether the broadcast is valid.
		BCast bcast(BCast::FromShape(input_shape), BCast::FromShape(output_shape), /*fewer_dims_optimization=*/true);
		OP_REQUIRES(ctx, bcast.IsValid(), errors::InvalidArgument("Incompatible shapes: ", input_shape.DebugString(), " vs. ", output_shape.DebugString()));
		OP_REQUIRES(ctx, BCast::ToShape(bcast.output_shape()) == output_shape, errors::InvalidArgument("Unable to broadcast tensor of shape ", input_shape, " to tensor of shape ", output_shape));

		// Handle empty case.
		if(output_shape.num_elements() == 0) {
			return;
		}

		TODO();
		/*functor::BroadcastTo<Device, T>()(device, ctx, *output_tensor, output_shape,
										  input_tensor, input_shape, bcast);*/
	}
};

//------------------------------------------------------------------------------
}

#include "__ns.h"
//------------------------------------------------------------------------------
void init_broadcast_ops(void) {
	REG10_("BroadcastTo", "T", .HostMemory("shape"), ::tensorflow::BroadcastToOp, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, 0)
}

//------------------------------------------------------------------------------
#include "__ns.h"