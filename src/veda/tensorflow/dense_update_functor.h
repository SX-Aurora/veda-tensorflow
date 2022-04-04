namespace tensorflow {
	namespace functor {
//------------------------------------------------------------------------------
template<typename T>
struct DenseUpdate<VEDATensors_handle_struct, T, ADD> {
	void operator()(const VEDATensors_handle_struct& d, typename TTypes<T>::Flat params, typename TTypes<T>::ConstFlat update) {
		CVEDA(veda_tensors_unary_ll_tt(&d, (VEDAdeviceptr)params.data, (VEDAdeviceptr)update.data(), params.size(), update.size(), VEDA_TENSORS_UNARY_ADD, veda::tensorflow::dtype<T>()));
	}
};

//------------------------------------------------------------------------------
template<typename T>
struct DenseUpdate<VEDATensors_handle_struct, T, SUB> {
	void operator()(const VEDATensors_handle_struct& d, typename TTypes<T>::Flat params, typename TTypes<T>::ConstFlat update) {
		CVEDA(veda_tensors_unary_ll_tt(&d, (VEDAdeviceptr)params.data, (VEDAdeviceptr)update.data(), params.size(), update.size(), VEDA_TENSORS_UNARY_SUB, veda::tensorflow::dtype<T>()));
	}
};

//------------------------------------------------------------------------------
template<typename T>
struct DenseUpdate<VEDATensors_handle_struct, T, ASSIGN> {
	void operator()(const VEDATensors_handle_struct& d, typename TTypes<T>::Flat params, typename TTypes<T>::ConstFlat update) {
		CVEDA(veda_tensors_ll_copy(&d, (VEDAdeviceptr)params.data(), (VEDAdeviceptr)update.data(), params.size(), update.size(), veda::tensorflow::dtype<T>()));
	}
};

//------------------------------------------------------------------------------
	}
}