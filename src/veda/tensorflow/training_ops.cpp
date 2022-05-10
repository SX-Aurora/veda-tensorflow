#include <veda/tensorflow/api.h>

namespace tensorflow {
//------------------------------------------------------------------------------
inline void MaybeForwardRefInputToRefOutput(OpKernelContext* ctx, int input, int output) {
	if(ctx->input_dtype(input) != DT_RESOURCE) {
		ctx->forward_ref_input_to_ref_output(input, output);
	}
}

//------------------------------------------------------------------------------
template <typename Device, typename T>
class ApplyGradientDescentOp : public OpKernel {
public:
	explicit ApplyGradientDescentOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
		OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
	}

	void Compute(OpKernelContext* ctx) override {
		using namespace veda::tensorflow;

		const bool sparse = false;
		auto locks = MaybeLockVariableInputMutexesInOrder<Device, T>(ctx, use_exclusive_lock_, sparse, {0});
		Tensor var;
		OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(ctx, 0, use_exclusive_lock_, sparse, &var));
		OP_REQUIRES(ctx, var.IsInitialized(), errors::FailedPrecondition("Attempting to use uninitialized variables: ", requested_input(0)));
		const Tensor& alpha = ctx->input(1);
		OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(alpha.shape()), errors::InvalidArgument("alpha is not a scalar: ", alpha.shape().DebugString()));
		const Tensor& delta = ctx->input(2);
		OP_REQUIRES(ctx, var.shape().IsSameSize(delta.shape()), errors::InvalidArgument("var and delta do not have the same shape", var.shape().DebugString(), " ", delta.shape().DebugString()));

		auto d_var		= tf2veda<T>(var);
		auto d_alpha	= tf2scalar<T>(alpha);
		auto d_delta	= tf2veda<T>(delta);

		CVEDA(veda_tensors_unary_tts(handle(ctx), &d_var, &d_var, &d_delta, d_alpha, VEDA_TENSORS_UNARY_SUB));

		MaybeForwardRefInputToRefOutput(ctx, 0, 0);
	}

private:
	bool use_exclusive_lock_;
};

//------------------------------------------------------------------------------
template <typename Device, typename T>
class ApplyAdadeltaOp : public OpKernel {
public:
	explicit ApplyAdadeltaOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
		OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
	}

	void Compute(OpKernelContext* ctx) override {
		const bool sparse = false;
		auto locks = MaybeLockVariableInputMutexesInOrder<Device, T>(ctx, use_exclusive_lock_, sparse, {0, 1, 2});
		DoValidate(ctx);
		if (!ctx->status().ok()) return;
		DoCompute(ctx);
		MaybeForwardRefInputToRefOutput(ctx, 0, 0);
	}

private:
	bool use_exclusive_lock_;

	void DoValidate(OpKernelContext* ctx) {
		Tensor var;
		const bool sparse = false;
		OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(ctx, 0, use_exclusive_lock_, sparse, &var));
		Tensor accum;
		OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(ctx, 1, use_exclusive_lock_, sparse, &accum));
		Tensor accum_update;
		OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(ctx, 2, use_exclusive_lock_, sparse, &accum_update));

		OP_REQUIRES(ctx, var.IsInitialized(), errors::FailedPrecondition("Attempting to use uninitialized variables: ", requested_input(0)));
		OP_REQUIRES(ctx, accum.IsInitialized(), errors::FailedPrecondition("Attempting to use uninitialized variables: ", requested_input(1)));
		OP_REQUIRES(ctx, accum_update.IsInitialized(), errors::FailedPrecondition("Attempting to use uninitialized variables: ", requested_input(2)));

		const Tensor& lr = ctx->input(3);
		const Tensor& rho = ctx->input(4);
		const Tensor& epsilon = ctx->input(5);
		const Tensor& grad = ctx->input(6);

		OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()), errors::InvalidArgument("lr is not a scalar: ", lr.shape().DebugString()));
		OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(rho.shape()), errors::InvalidArgument("rho is not a scalar: ", rho.shape().DebugString()));
		OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()), errors::InvalidArgument("epsilon is not a scalar: ", epsilon.shape().DebugString()));

		OP_REQUIRES(ctx, var.shape().IsSameSize(accum.shape()), errors::InvalidArgument("var and accum do not have the same shape", var.shape().DebugString(), " ", accum.shape().DebugString()));
		OP_REQUIRES(ctx, var.shape().IsSameSize(grad.shape()), errors::InvalidArgument("var and grad do not have the same shape", var.shape().DebugString(), " ", grad.shape().DebugString()));
	}

	void DoCompute(OpKernelContext* ctx) {
		using namespace veda::tensorflow; 

		Tensor var;
		const bool sparse = false;
		OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(ctx, 0, use_exclusive_lock_, sparse, &var));
		Tensor accum;
		OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(ctx, 1, use_exclusive_lock_, sparse, &accum));
		Tensor accum_update;
		OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(ctx, 2, use_exclusive_lock_, sparse, &accum_update));

		const Tensor& lr		= ctx->input(3);
		const Tensor& rho		= ctx->input(4);
		const Tensor& epsilon	= ctx->input(5);
		const Tensor& grad		= ctx->input(6);

		auto d_var			= tf2veda<T>(var);
		auto d_accum		= tf2veda<T>(accum);
		auto d_grad			= tf2veda<T>(grad);
		auto d_accum_update = tf2veda<T>(accum_update);
		auto d_rho			= tf2scalar<T>(rho);
		auto d_epsilon		= tf2scalar<T>(epsilon);
		auto d_lr			= tf2scalar<T>(lr);

		CVEDA(veda_tensors_adadelta(handle(ctx), &d_var, &d_accum, &d_accum_update, &d_grad, d_rho, d_epsilon, d_lr));

		MaybeForwardRefInputToRefOutput(ctx, 0, 0);
	}
};

//------------------------------------------------------------------------------
template <typename Device, typename T>
class ApplyAdagradV2Op : public OpKernel {
public:
	explicit ApplyAdagradV2Op(OpKernelConstruction* ctx) : OpKernel(ctx) {
		OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
		OP_REQUIRES_OK(ctx, ctx->GetAttr("update_slots", &update_slots_));
	}

	void Compute(OpKernelContext* ctx) override {
		using namespace veda::tensorflow;

		const bool sparse = false;
		auto locks = MaybeLockVariableInputMutexesInOrder<Device, T>(ctx, use_exclusive_lock_, sparse, {0, 1});
		Tensor var;
		OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(ctx, 0, use_exclusive_lock_, sparse, &var));
		Tensor accum;
		OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(ctx, 1, use_exclusive_lock_, sparse, &accum));
		OP_REQUIRES(ctx, var.IsInitialized(), errors::FailedPrecondition("Attempting to use uninitialized variables: ", requested_input(0)));
		OP_REQUIRES(ctx, accum.IsInitialized(), errors::FailedPrecondition("Attempting to use uninitialized variables: ", requested_input(1)));
		const Tensor& lr = ctx->input(2);
		OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()), errors::InvalidArgument("lr is not a scalar: ", lr.shape().DebugString()));
		const Tensor& epsilon = ctx->input(3);
		OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()), errors::InvalidArgument("epsilon is not a scalar: ", epsilon.shape().DebugString()));
		const Tensor& grad = ctx->input(4);
		OP_REQUIRES(ctx, var.shape().IsSameSize(accum.shape()), errors::InvalidArgument("var and accum do not have the same shape", var.shape().DebugString(), " ", accum.shape().DebugString()));
		OP_REQUIRES(ctx, var.shape().IsSameSize(grad.shape()), errors::InvalidArgument("var and grad do not have the same shape", var.shape().DebugString(), " ", grad.shape().DebugString()));

		auto d_var		= tf2veda<T>(var);
		auto d_accum	= tf2veda<T>(accum);
		auto d_grad		= tf2veda<T>(grad);
		auto d_lr		= tf2scalar<T>(lr);
		auto d_epsilon	= tf2scalar<T>(epsilon);

		CVEDA(veda_tensors_adagrad(handle(ctx), &d_var, &d_accum, &d_grad, d_epsilon, d_lr, update_slots_));

		MaybeForwardRefInputToRefOutput(ctx, 0, 0);
	}

private:
	bool use_exclusive_lock_;
	bool update_slots_;
};

//------------------------------------------------------------------------------
template <typename Device, typename T>
class ApplyAdamOp : public OpKernel {
public:
	explicit ApplyAdamOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
		OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
		OP_REQUIRES_OK(ctx, ctx->GetAttr("use_nesterov", &use_nesterov_));
	}

	void Compute(OpKernelContext* ctx) override {
		using namespace veda::tensorflow;

		const bool sparse = false;
		auto locks = MaybeLockVariableInputMutexesInOrder<Device, T>(ctx, use_exclusive_lock_, sparse, {0, 1, 2});

		Tensor var;
		OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(ctx, 0, use_exclusive_lock_, sparse, &var));
		Tensor m;
		OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(ctx, 1, use_exclusive_lock_, sparse, &m));
		Tensor v;
		OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(ctx, 2, use_exclusive_lock_, sparse, &v));
		OP_REQUIRES(ctx, var.IsInitialized(), errors::FailedPrecondition("Attempting to use uninitialized variables: ", requested_input(0)));
		OP_REQUIRES(ctx, m.IsInitialized(), errors::FailedPrecondition("Attempting to use uninitialized variables: ", requested_input(1)));
		OP_REQUIRES(ctx, v.IsInitialized(), errors::FailedPrecondition("Attempting to use uninitialized variables: ", requested_input(2)));

		const Tensor& beta1_power = ctx->input(3);
		const Tensor& beta2_power = ctx->input(4);
		const Tensor& lr = ctx->input(5);
		const Tensor& beta1 = ctx->input(6);
		const Tensor& beta2 = ctx->input(7);
		const Tensor& epsilon = ctx->input(8);

		OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1_power.shape()), errors::InvalidArgument("beta1_power is not a scalar: ", beta1_power.shape().DebugString()));
		OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2_power.shape()), errors::InvalidArgument("beta2_power is not a scalar: ", beta2_power.shape().DebugString()));
		OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()), errors::InvalidArgument("lr is not a scalar : ", lr.shape().DebugString()));
		OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1.shape()), errors::InvalidArgument("beta1 is not a scalar: ", beta1.shape().DebugString()));
		OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2.shape()), errors::InvalidArgument("beta2 is not a scalar: ", beta2.shape().DebugString()));
		OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()), errors::InvalidArgument("epsilon is not a scalar: ", epsilon.shape().DebugString()));

		const Tensor& grad = ctx->input(9);
		OP_REQUIRES(ctx, var.shape().IsSameSize(m.shape()), errors::InvalidArgument("var and m do not have the same shape", var.shape().DebugString(), " ", m.shape().DebugString()));
		OP_REQUIRES(ctx, var.shape().IsSameSize(v.shape()), errors::InvalidArgument("var and v do not have the same shape", var.shape().DebugString(), " ", v.shape().DebugString()));
		OP_REQUIRES(ctx, var.shape().IsSameSize(grad.shape()), errors::InvalidArgument("var and grad do not have the same shape", var.shape().DebugString(), " ", grad.shape().DebugString()));

		auto d_var			= tf2veda<T>(var);
		auto d_m			= tf2veda<T>(m);
		auto d_v			= tf2veda<T>(v);
		auto d_grad			= tf2veda<T>(grad);
		auto d_beta1_power	= tf2scalar<T>(beta1_power);
		auto d_beta2_power	= tf2scalar<T>(beta2_power);
		auto d_lr			= tf2scalar<T>(lr);
		auto d_beta1		= tf2scalar<T>(beta1);
		auto d_beta2		= tf2scalar<T>(beta2);
		auto d_epsilon		= tf2scalar<T>(epsilon);

		CVEDA(veda_tensors_adam(handle(ctx), &d_var, &d_m, &d_v, &d_grad, d_beta1_power, d_beta2_power, d_lr, d_beta1, d_beta2, d_epsilon, use_nesterov_));

		MaybeForwardRefInputToRefOutput(ctx, 0, 0);
	}

private:
	bool use_exclusive_lock_;
	bool use_nesterov_;
};

//------------------------------------------------------------------------------
template <typename Device, typename T>
class ApplyAdaMaxOp : public OpKernel {
public:
	explicit ApplyAdaMaxOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
		OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
	}

	void Compute(OpKernelContext* ctx) override {
		using namespace veda::tensorflow;

		const bool sparse = false;
		auto locks = MaybeLockVariableInputMutexesInOrder<Device, T>(ctx, use_exclusive_lock_, sparse, {0, 1, 2});

		Tensor var;
		OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(ctx, 0, use_exclusive_lock_, sparse, &var));
		Tensor m;
		OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(ctx, 1, use_exclusive_lock_, sparse, &m));
		Tensor v;
		OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(ctx, 2, use_exclusive_lock_, sparse, &v));
		OP_REQUIRES(ctx, var.IsInitialized(), errors::FailedPrecondition("Attempting to use uninitialized variables: ", requested_input(0)));
		OP_REQUIRES(ctx, m.IsInitialized(), errors::FailedPrecondition("Attempting to use uninitialized variables: ", requested_input(1)));
		OP_REQUIRES(ctx, v.IsInitialized(), errors::FailedPrecondition("Attempting to use uninitialized variables: ", requested_input(2)));

		const Tensor& beta1_power = ctx->input(3);
		const Tensor& lr = ctx->input(4);
		const Tensor& beta1 = ctx->input(5);
		const Tensor& beta2 = ctx->input(6);
		const Tensor& epsilon = ctx->input(7);

		OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1_power.shape()), errors::InvalidArgument("beta1_power is not a scalar: ", beta1_power.shape().DebugString()));
		OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()), errors::InvalidArgument("lr is not a scalar : ", lr.shape().DebugString()));
		OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1.shape()), errors::InvalidArgument("beta1 is not a scalar: ", beta1.shape().DebugString()));
		OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2.shape()), errors::InvalidArgument("beta2 is not a scalar: ", beta2.shape().DebugString()));
		OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()), errors::InvalidArgument("epsilon is not a scalar: ", epsilon.shape().DebugString()));

		const Tensor& grad = ctx->input(8);
		OP_REQUIRES(ctx, var.shape().IsSameSize(m.shape()), errors::InvalidArgument("var and m do not have the same shape", var.shape().DebugString(), " ", m.shape().DebugString()));
		OP_REQUIRES(ctx, var.shape().IsSameSize(v.shape()), errors::InvalidArgument("var and v do not have the same shape", var.shape().DebugString(), " ", v.shape().DebugString()));
		OP_REQUIRES(ctx, var.shape().IsSameSize(grad.shape()), errors::InvalidArgument("var and grad do not have the same shape", var.shape().DebugString(), " ", grad.shape().DebugString()));

		auto d_var			= tf2veda<T>(var);
		auto d_m			= tf2veda<T>(m);
		auto d_v			= tf2veda<T>(v);
		auto d_grad			= tf2veda<T>(grad);
		auto d_beta1_power	= tf2scalar<T>(beta1_power);
		auto d_lr			= tf2scalar<T>(lr);
		auto d_beta1		= tf2scalar<T>(beta1);
		auto d_beta2		= tf2scalar<T>(beta2);
		auto d_epsilon		= tf2scalar<T>(epsilon);

		CVEDA(veda_tensors_adamax(handle(ctx), &d_var, &d_m, &d_v, &d_grad, d_beta1_power, d_lr, d_beta1, d_beta2, d_epsilon));

		MaybeForwardRefInputToRefOutput(ctx, 0, 0);
	}

private:
	bool use_exclusive_lock_;
};

//------------------------------------------------------------------------------
}

#include "__ns.h"
//------------------------------------------------------------------------------
void init_training_ops(void) {
	using namespace ::tensorflow;

	#define REGISTER_ApplyGradientDescent_(S, T) REGISTER_KERNEL_BUILDER(Name(S).Device(DEVICE_VE).HostMemory("alpha").TypeConstraint<T>("T"), ApplyGradientDescentOp<VEDevice, T>);
	#define REGISTER_ApplyGradientDescent(T)\
		REGISTER_ApplyGradientDescent_("ApplyGradientDescent", T)\
		REGISTER_ApplyGradientDescent_("ResourceApplyGradientDescent", T)

	#define REGISTER_ApplyAdadelta_(S, T) REGISTER_KERNEL_BUILDER(Name(S).Device(DEVICE_VE).HostMemory("lr").HostMemory("rho").HostMemory("epsilon").TypeConstraint<T>("T"), ApplyAdadeltaOp<VEDevice, T>);
	#define REGISTER_ApplyAdadelta(T)\
 		REGISTER_ApplyAdadelta_("ApplyAdadelta", T)\
  		REGISTER_ApplyAdadelta_("ResourceApplyAdadelta", T)

	#define REGISTER_ApplyAdagradV2_(S, T) REGISTER_KERNEL_BUILDER(Name(S).Device(DEVICE_VE).HostMemory("lr").HostMemory("epsilon").TypeConstraint<T>("T"), ApplyAdagradV2Op<VEDevice, T>);
	#define REGISTER_ApplyAdagradV2(T)\
		REGISTER_ApplyAdagradV2_("ApplyAdagradV2", T)\
  		REGISTER_ApplyAdagradV2_("ResourceApplyAdagradV2", T)

	#define REGISTER_ApplyAdam_(S, T) REGISTER_KERNEL_BUILDER(Name(S).Device(DEVICE_VE).HostMemory("beta1_power").HostMemory("beta2_power").HostMemory("lr").HostMemory("beta1").HostMemory("beta2").HostMemory("epsilon").TypeConstraint<T>("T"), ApplyAdamOp<VEDevice, T>);
	#define REGISTER_ApplyAdam(T)\
		REGISTER_ApplyAdam_("ApplyAdam", T)\
		REGISTER_ApplyAdam_("ResourceApplyAdam", T)

	#define REGISTER_ApplyAdamax_(S, T) REGISTER_KERNEL_BUILDER(Name(S).Device(DEVICE_VE).HostMemory("beta1_power").HostMemory("lr").HostMemory("beta1").HostMemory("beta2").HostMemory("epsilon").TypeConstraint<T>("T"), ApplyAdaMaxOp<VEDevice, T>);
	#define REGISTER_ApplyAdamax(T)\
		REGISTER_ApplyAdamax_("ApplyAdaMax", T)\
		REGISTER_ApplyAdamax_("ResourceApplyAdaMax", T)

	#define REGISTER_TYPES(FUNC) FUNC(float) FUNC(double)
	REGISTER_TYPES(REGISTER_ApplyGradientDescent)
	REGISTER_TYPES(REGISTER_ApplyAdadelta)
	REGISTER_TYPES(REGISTER_ApplyAdagradV2)
	REGISTER_TYPES(REGISTER_ApplyAdam)
	REGISTER_TYPES(REGISTER_ApplyAdamax)
}

//------------------------------------------------------------------------------
#include "__ns.h"