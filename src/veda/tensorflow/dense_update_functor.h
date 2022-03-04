namespace tensorflow {
	namespace functor {
//------------------------------------------------------------------------------
template<typename T>
struct DenseUpdate<VEDevice, T, ADD> {
	void operator()(const VEDevice& d, typename TTypes<T>::Flat params, typename TTypes<T>::ConstFlat update) {
		const_cast<VEDevice&>(d).unaryTT((VEDAdeviceptr)params.data(), (VEDAdeviceptr)params.data, (VEDAdeviceptr)update.data(), params.size(), update.size(), sol::runtime::UnaryOp::ADD, sol_dtype<T>());
	}
};

//------------------------------------------------------------------------------
template<typename T>
struct DenseUpdate<VEDevice, T, SUB> {
	void operator()(const VEDevice& d, typename TTypes<T>::Flat params, typename TTypes<T>::ConstFlat update) {
		const_cast<VEDevice&>(d).unaryTT((VEDAdeviceptr)params.data(), (VEDAdeviceptr)params.data, (VEDAdeviceptr)update.data(), params.size(), update.size(), sol::runtime::UnaryOp::SUB, sol_dtype<T>());
	}
};

//------------------------------------------------------------------------------
template<typename T>
struct DenseUpdate<VEDevice, T, ASSIGN> {
	void operator()(const VEDevice& d, typename TTypes<T>::Flat params, typename TTypes<T>::ConstFlat update) {
		const_cast<VEDevice&>(d).copy((VEDAdeviceptr)params.data(), (VEDAdeviceptr)update.data(), params.size(), update.size(), sol_dtype<T>());
	}
};

//------------------------------------------------------------------------------
	}
}