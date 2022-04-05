#include <veda/tensorflow/api.h>

namespace tensorflow {
//------------------------------------------------------------------------------
static NodeDef StripTensorDataFromNodeDef(OpKernelConstruction* ctx) {
	const NodeDef& original = ctx->def();
	if(std::is_base_of<protobuf::Message, NodeDef>()) {
		DCHECK_EQ(reinterpret_cast<const protobuf::Message*>(&original)->GetDescriptor()->field_count(), 6)
			<< "The NodeDef format has changed, and the attr-stripping code may "
				"need to be updated.";
	}
	NodeDef ret;
	ret.set_name(original.name());
	ret.set_op(original.op());
	ret.set_device(original.device());
	AddNodeAttr("dtype", ctx->output_type(0), &ret);
	MergeDebugInfo(original, &ret);
	return ret;
}

//------------------------------------------------------------------------------
ConstantOp::ConstantOp(OpKernelConstruction* ctx) : OpKernel(ctx, StripTensorDataFromNodeDef(ctx), false), tensor_(ctx->output_type(0)) {
	const TensorProto* proto = nullptr;
	OP_REQUIRES_OK(ctx, ctx->GetAttr("value", &proto));
	OP_REQUIRES_OK(ctx, ctx->device()->MakeTensorFromProto(*proto, AllocatorAttributes(), &tensor_));
	OP_REQUIRES(ctx, ctx->output_type(0) == tensor_.dtype(),
    	errors::InvalidArgument("Type mismatch between value (",
                              DataTypeString(tensor_.dtype()), ") and dtype (",
                              DataTypeString(ctx->output_type(0)), ")"));
}

//------------------------------------------------------------------------------
void ConstantOp::Compute(OpKernelContext* ctx) {
	ctx->set_output(0, tensor_);
	if(TF_PREDICT_FALSE(ctx->track_allocations())) {
		ctx->record_persistent_memory_allocation(tensor_.AllocatedBytes());
	}
}

//------------------------------------------------------------------------------
ConstantOp::~ConstantOp() {}

//------------------------------------------------------------------------------
}

#include "__ns.h"
//------------------------------------------------------------------------------
void init_constant_op(void) {
	REGISTER_KERNEL_BUILDER(Name("Const").Device(DEVICE_VE).TypeConstraint("dtype", ::tensorflow::VE_TYPES), ::tensorflow::ConstantOp);
}

//------------------------------------------------------------------------------
#include "__ns.h"