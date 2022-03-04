#include <veda/tensorflow/api.h>

namespace tensorflow {
//------------------------------------------------------------------------------
ArgOp::ArgOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
	OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
	OP_REQUIRES_OK(ctx, ctx->GetAttr("index", &index_));
}

//------------------------------------------------------------------------------
void ArgOp::Compute(OpKernelContext* ctx) {
	auto frame = ctx->call_frame();
	OP_REQUIRES(ctx, frame != nullptr, errors::Internal("no call frame"));
	const Tensor* val;

	auto validate_type = [this](const Tensor& val) {
		if(val.dtype() == dtype_)	return Status::OK();
		else						return errors::InvalidArgument("Type mismatch: actual ", DataTypeString(val.dtype()), " vs. expect ", DataTypeString(dtype_));
	};

	if(frame->CanConsumeArg(index_)) {
		Tensor val;
		frame->ConsumeArg(index_, &val);
		OP_REQUIRES_OK(ctx, validate_type(val));
		ctx->set_output(0, std::move(val));
	} else {
		OP_REQUIRES_OK(ctx, frame->GetArg(index_, &val));
		OP_REQUIRES_OK(ctx, validate_type(*val));
		ctx->set_output(0, *val);
	}
}

//------------------------------------------------------------------------------
RetvalOp::RetvalOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
	OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
	OP_REQUIRES_OK(ctx, ctx->GetAttr("index", &index_));
}

//------------------------------------------------------------------------------
void RetvalOp::Compute(OpKernelContext* ctx) {
	const Tensor& val = ctx->input(0);
	OP_REQUIRES(ctx, val.dtype() == dtype_, errors::InvalidArgument("Type mismatch: actual ", DataTypeString(val.dtype()), " vs. expect ", DataTypeString(dtype_)));
	auto frame = ctx->call_frame();
	OP_REQUIRES(ctx, frame != nullptr, errors::Internal("no call frame"));
	OP_REQUIRES_OK(ctx, frame->SetRetval(index_, val));
}

//------------------------------------------------------------------------------
}

#include "__ns.h"
//------------------------------------------------------------------------------
void init_function_ops(void) {
	using namespace ::tensorflow;
	REGISTER_KERNEL_BUILDER(Name(kArgOp).Device(DEVICE_VE).TypeConstraint("T", VE_TYPES),													ArgOp);
	REGISTER_KERNEL_BUILDER(Name(kArgOp).Device(DEVICE_VE).HostMemory("output").TypeConstraint<ResourceHandle>("T"),						ArgOp);
	REGISTER_KERNEL_BUILDER(Name(kArgOp).Device(DEVICE_VE).TypeConstraint<Variant>("T"),													ArgOp);
	REGISTER_KERNEL_BUILDER(Name(kRetOp).Device(DEVICE_VE).TypeConstraint("T", VE_TYPES),													RetvalOp);
	REGISTER_KERNEL_BUILDER(Name(kRetOp).Device(DEVICE_VE).TypeConstraint<ResourceHandle>("T").HostMemory("input"),							RetvalOp);
	REGISTER_KERNEL_BUILDER(Name(kDeviceRetOp).Device(DEVICE_VE).TypeConstraint<int32>("T"),												RetvalOp);
	REGISTER_KERNEL_BUILDER(Name(FunctionLibraryDefinition::kArgOp).Device(DEVICE_VE).HostMemory("output").TypeConstraint<tstring>("T"),	ArgOp);
	REGISTER_KERNEL_BUILDER(Name(FunctionLibraryDefinition::kRetOp).Device(DEVICE_VE).TypeConstraint<tstring>("T").HostMemory("input"),		RetvalOp);
}

//------------------------------------------------------------------------------
#include "__ns.h"