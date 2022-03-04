#include <veda/tensorflow/api.h>

namespace tensorflow {
//------------------------------------------------------------------------------
template<typename Device, typename T>
class AssignVariableOp : public OpKernel {
public:
	explicit AssignVariableOp(OpKernelConstruction* c) : OpKernel(c) {
		OP_REQUIRES_OK(c, c->GetAttr("dtype", &dtype_));
		if (!c->GetAttr("_grappler_relax_allocator_constraints",
						&relax_constraints_)
				.ok()) {
		relax_constraints_ = false;
		}
	}

	void Compute(OpKernelContext* context) override {
		OP_REQUIRES(context, dtype_ == context->input(1).dtype(),
					errors::InvalidArgument(
						"Variable and value dtypes don't match; respectively, ",
						DataTypeString(dtype_), " and ",
						DataTypeString(context->input(1).dtype())));
		core::RefCountPtr<Var> variable;
		const Tensor& value = context->input(1);
		// Note: every resource-variable-manipulating op assumes copy-on-write
		// semantics, and creates a copy of the variable's Tensor if its refcount is
		// bigger than 1 when we try to modify it. This means we never need to copy
		// the original tensor for AssignVariableOp; even if there are other live
		// users of it we know none can modify it so this is always safe (even in
		// esoteric cases where the same tensor is used to initialize multiple
		// variables or the tensor is a constant this is safe, as future writes will
		// trigger copies).
		OP_REQUIRES_OK(context, LookupOrCreateResource<Var>(
									context, HandleFromInput(context, 0), &variable,
									[this, &value](Var** ptr) {
									*ptr = new Var(dtype_);
									*(*ptr)->tensor() = value;
									(*ptr)->is_initialized = true;
									return Status::OK();
									}));
		mutex_lock ml(*variable->mu());
		// (variable->tensor()->dtype() == DT_INVALID && !variable->is_initialized)
		// check below is to allow an XLA specific situation wherein update can
		// happen first by the AssignVariableOp,
		// in which case the variable is still uninitialized.
		// When using TF-XLA, this scenario is possible when the execution uses the
		// 'fallback' path (which essentially invokes Tensorflow ops via
		// partitioned_call).
		OP_REQUIRES(context,
					(variable->tensor()->dtype() == DT_INVALID &&
					!variable->is_initialized) ||
						variable->tensor()->dtype() == dtype_,
					errors::InvalidArgument(
						"Trying to assign variable with wrong dtype. Expected ",
						DataTypeString(variable->tensor()->dtype()), " got ",
						DataTypeString(dtype_)));
		if (variable->copy_on_read_mode.load()) {
			AllocatorAttributes attr;
			attr.set_gpu_compatible(true);
			attr.set_nic_compatible(true);
			OP_REQUIRES_OK(context,
							context->allocate_temp(value.dtype(), value.shape(),
													variable->tensor(), attr));
			functor::DenseUpdate<Device, T, ASSIGN> copy_functor;
			copy_functor(context->eigen_device<Device>(),
						variable->tensor()->flat<T>(), value.flat<T>());
		} else {
			*variable->tensor() = value;
		}
		variable->is_initialized = true;
	}

private:
	DataType dtype_;
	bool relax_constraints_;
};

//------------------------------------------------------------------------------
DestroyResourceOp::DestroyResourceOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
	OP_REQUIRES_OK(ctx, ctx->GetAttr("ignore_lookup_error", &ignore_lookup_error_));
}

//------------------------------------------------------------------------------
void DestroyResourceOp::Compute(OpKernelContext* ctx) {
	const ResourceHandle& p = HandleFromInput(ctx, 0);
	Status status = DeleteResource(ctx, p);
	if(ignore_lookup_error_ && errors::IsNotFound(status))
		return;
	OP_REQUIRES_OK(ctx, status);
}

//------------------------------------------------------------------------------
ReadVariableOp::ReadVariableOp(OpKernelConstruction* c) : OpKernel(c) {
	OP_REQUIRES_OK(c, c->GetAttr("dtype", &dtype_));
}

//------------------------------------------------------------------------------
static Status CopyVariable(int output_idx, OpKernelContext* ctx, const Tensor* t) {
	Tensor* output;
	Notification n;
	Status status;
	AllocatorAttributes attr;
	if (t->dtype() == DT_VARIANT) {
		attr.set_on_host(true);
	}
	TF_RETURN_IF_ERROR(
		ctx->allocate_output(output_idx, t->shape(), &output, attr));
	if (t->dtype() == DT_VARIANT) {
		output->flat<Variant>() = t->flat<Variant>();
	} else if (ctx->op_device_context() != nullptr) {
		// TODO(apassos): remove the down_cast by just returning Device* from
		// OpKernelContext
		Device* device = down_cast<Device*>(ctx->device());
		ctx->op_device_context()->CopyTensorInSameDevice(
			t, device, output, [&n, &status](const Status& s) {
			status = s;
			n.Notify();
			});
		n.WaitForNotification();
		return status;
	} else {
		switch (t->dtype()) {
		#define HANDLER(type)						\
		case DataTypeToEnum<type>::value:			\
			output->flat<type>() = t->flat<type>();	\
			break;
			TF_CALL_ALL_TYPES(HANDLER);
		#undef HANDLER
		default:
			return errors::Internal("Unsupported dtype", t->dtype());
		}
	}
	return Status::OK();
}

//------------------------------------------------------------------------------
void ReadVariableOp::Compute(OpKernelContext* ctx) {
	core::RefCountPtr<Var> variable;
	const ResourceHandle& handle = HandleFromInput(ctx, 0);
	const auto status = LookupResource(ctx, handle, &variable);
	OP_REQUIRES(ctx, status.ok(),
				errors::FailedPrecondition(
					"Could not find variable ", handle.name(), ". ",
					"This could mean that the variable has been deleted. ",
					"In TF1, it can also mean the variable is uninitialized. ",
					"Debug info: container=", handle.container(),
					", status=", status.ToString()));

	tf_shared_lock ml(*variable->mu());
	// We're acquiring a reference to the underlying buffer while
	// holding a shared lock to guarantee ordering of reads and
	// writes when in copy-on-write mode.
	const Tensor* t = variable->tensor();
	if (!variable->copy_on_read_mode.load()) {
	OP_REQUIRES(
		ctx, dtype_ == t->dtype(),
		errors::InvalidArgument(
			"Trying to read variable with wrong dtype. Expected ",
			DataTypeString(dtype_), " got ", DataTypeString(t->dtype())));
		ctx->set_output(0, *t);
	} else {
		OP_REQUIRES_OK(ctx, CopyVariable(0, ctx, t));
	}
}

//------------------------------------------------------------------------------
ReadVariablesOp::ReadVariablesOp(OpKernelConstruction* c) : OpKernel(c) {
	int n;
	OP_REQUIRES_OK(c, c->GetAttr("N", &n));
	OP_REQUIRES_OK(c, c->GetAttr("dtypes", &dtypes_));
	OP_REQUIRES(c, n == dtypes_.size(), errors::InvalidArgument("Mismatched number of arguments to ReadVariablesOp (", n, " vs. ", dtypes_.size(), ")"));
}

//------------------------------------------------------------------------------
void ReadVariablesOp::Compute(OpKernelContext* ctx) {
	std::vector<core::RefCountPtr<Var>> variables(dtypes_.size());
	std::vector<const ResourceHandle*> handles(dtypes_.size());
	for (size_t i = 0; i < dtypes_.size(); ++i) {
		handles[i] = &HandleFromInput(ctx, i);
	}

	OP_REQUIRES_OK(ctx, LookupResources(ctx, handles, &variables));

	std::vector<string> uninitialized_vars;
	for (int64 i = 0; i < variables.size(); i++) {
		if (variables[i] == nullptr) {
			uninitialized_vars.push_back(handles[i]->name());
		}
	}

	OP_REQUIRES(ctx, uninitialized_vars.empty(),
				errors::FailedPrecondition(
					"In ReadVariablesOp the following variables were "
					"found uninitialized: ",
					absl::StrJoin(uninitialized_vars, ", ")));

	for (size_t i = 0; i < dtypes_.size(); ++i) {
		// We're acquiring a reference to the underlying buffer while
		// holding a shared lock to guarantee ordering of reads and
		// writes.
		tf_shared_lock ml(*variables[i]->mu());
		OP_REQUIRES(ctx, dtypes_[i] == variables[i]->tensor()->dtype(),
					errors::InvalidArgument(
						"Trying to read variable ", handles[i]->name(),
						" from Container: ", handles[i]->container(),
						" with wrong dtype. Expected ", DataTypeString(dtypes_[i]),
						" got ", DataTypeString(variables[i]->tensor()->dtype())));
		if (variables[i]->copy_on_read_mode.load()) {
			OP_REQUIRES_OK(ctx, CopyVariable(i, ctx, variables[i]->tensor()));
		} else {
			const Tensor& t = *variables[i]->tensor();
			ctx->set_output(i, t);
		}
	}
}

//------------------------------------------------------------------------------
VarHandleOp::VarHandleOp(OpKernelConstruction* context) : OpKernel(context) {
	OP_REQUIRES_OK(context, context->GetAttr("container", &container_));
	OP_REQUIRES_OK(context, context->GetAttr("shared_name", &name_));

	OP_REQUIRES_OK(context, context->GetAttr("dtype", &dtype_and_shape_.dtype));
	OP_REQUIRES_OK(context, context->GetAttr("shape", &dtype_and_shape_.shape));

	is_anonymous_ = name_ == ResourceHandle::ANONYMOUS_NAME;

	if(!is_anonymous_) {
		AllocatorAttributes attr;
		attr.set_on_host(true);
		OP_REQUIRES_OK(context, context->allocate_temp(DT_RESOURCE, TensorShape({}),
													&resource_, attr));
		resource_.scalar<ResourceHandle>()() = MakeResourceHandle<Var>(
			context, container_, name_,
			std::vector<DtypeAndPartialTensorShape>{dtype_and_shape_});
	}
}

//------------------------------------------------------------------------------
void VarHandleOp::Compute(OpKernelContext* ctx) {
	if(is_anonymous_) {
		AllocatorAttributes attr;
		attr.set_on_host(true);
		Tensor handle;
		OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_RESOURCE, TensorShape({}), &handle, attr));
		handle.scalar<ResourceHandle>()() = MakeResourceHandle<Var>(
			ctx, container_, name_,
			std::vector<DtypeAndPartialTensorShape>{dtype_and_shape_},
			ctx->stack_trace());
		ctx->set_output(0, handle);
	} else {
		ctx->set_output(0, resource_);
	}
}

//------------------------------------------------------------------------------
}

#include "__ns.h"
//------------------------------------------------------------------------------
void init_resource_variable_ops(void) {
	using namespace ::tensorflow;

	#define REGISTER_TYPES(FUNC) FUNC(uint8_t) FUNC(uint16_t) FUNC(uint32_t) FUNC(uint64_t) FUNC(int8_t) FUNC(int16_t) FUNC(int32_t) FUNC(int64_t) FUNC(float) FUNC(double)
	#define REGISTER_AssignVariableOp(type)				REGISTER_KERNEL_BUILDER(Name("AssignVariableOp").Device(DEVICE_VE).TypeConstraint<type>("dtype"), AssignVariableOp<VEDevice, type>);

	REGISTER_TYPES(REGISTER_AssignVariableOp)

	REGISTER_KERNEL_BUILDER(Name("DestroyResourceOp")	.Device(DEVICE_VE).HostMemory("resource"),		DestroyResourceOp);
	REGISTER_KERNEL_BUILDER(Name("ReadVariableOp")		.Device(DEVICE_VE).HostMemory("resource"),		ReadVariableOp);
	REGISTER_KERNEL_BUILDER(Name("VarHandleOp")			.Device(DEVICE_VE).HostMemory("resource"),		VarHandleOp);
	REGISTER_KERNEL_BUILDER(Name("_ReadVariablesOp")	.Device(DEVICE_VE).HostMemory("resources"),		ReadVariablesOp);
}

//------------------------------------------------------------------------------
#include "__ns.h"