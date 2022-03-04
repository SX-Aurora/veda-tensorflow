#include <veda/tensorflow/api.h>
#include <malloc.h>
#include <mutex>
#include <chrono>

struct SP_Event_st {
private:
	SE_EventStatus 			m_status;
	std::condition_variable	m_condition;
	std::mutex				m_mutex;

public:
	inline								SP_Event_st	(void) : m_status(SE_EVENT_COMPLETE) {}
	inline	SE_EventStatus				status		(void) const					{	return m_status;	}
	inline	std::condition_variable&	condition	(void)							{	return m_condition;	}
	inline	std::mutex&					mutex		(void)							{	return m_mutex;		}
	inline	void						setStatus	(const SE_EventStatus status)	{	m_status = status;	}
};

#define LOCK(event) std::lock_guard<std::mutex> __lock__ (event->mutex())

#include "__ns.h"
//------------------------------------------------------------------------------
static void create_event(const SP_Device* device, SP_Event* event, TF_Status* status) {
	//L_TRACE("create_event");
	*event = new SP_Event_st;
	TF_SetStatus(status, TF_OK,	"");
}

//------------------------------------------------------------------------------
static void destroy_event(const SP_Device* device, SP_Event event) {
	//L_TRACE("destroy_event");
	delete event;
}

//------------------------------------------------------------------------------
static SE_EventStatus get_event_status(const SP_Device* device, SP_Event event) {
	//L_TRACE("get_event_status");
	LOCK(event);
	return event->status();
}

//------------------------------------------------------------------------------
static void record_event_helper(VEDAstream stream, VEDAresult result, void* args) {
	auto event = (SP_Event)args;
	{
		LOCK(event);
		ASSERT(event->status() == SE_EVENT_PENDING);
		event->setStatus(SE_EVENT_COMPLETE);
	}
	event->condition().notify_all();
}

//------------------------------------------------------------------------------
static void record_event(const SP_Device* device, SP_Stream stream, SP_Event event, TF_Status* status) {
	//L_TRACE("record_event");
	TF_SetStatus(status, TF_OK,	"");

	GUARD(device);
	{
		LOCK(event);
		ASSERT(event->status() == SE_EVENT_COMPLETE);
		event->setStatus(SE_EVENT_PENDING);
	}
	CVEDA(vedaStreamAddCallback(stream->stream, record_event_helper, event, 0));
}

//------------------------------------------------------------------------------
static void wait_for_event(const SP_Device* const device, SP_Stream stream, SP_Event event, TF_Status* const status) {
	//L_TRACE("wait_for_event");
	TF_SetStatus(status, TF_OK,	"");

	while(true) {
		std::unique_lock<std::mutex> lock(event->mutex());
		event->condition().wait(lock);
		if(event->status() == SE_EVENT_COMPLETE)
			return;
	}
}

//------------------------------------------------------------------------------
static void allocate(const SP_Device* device, uint64_t size, int64_t memory_space, SP_DeviceMemoryBase* mem) {
	GUARD(device);
	VEDAdeviceptr ptr;
	CVEDA(vedaMemAllocAsync(&ptr, size, 0));
	mem->opaque		= ptr;
	mem->size		= size;
	mem->payload	= 0;
}

//------------------------------------------------------------------------------
static void deallocate(const SP_Device* device, SP_DeviceMemoryBase* memory) {
	GUARD(device);
	CVEDA(vedaMemFreeAsync((VEDAdeviceptr)memory->opaque, 0));
}

//------------------------------------------------------------------------------
static void* host_memory_allocate(const SP_Device* device, uint64_t size) {
	// [.../tensorflow/core/framework/tensor.h:870] Check failed: IsAligned() seems to require 32 bit alignment?
	auto ptr = memalign(size, 32);
	L_TRACE("[VE#%i] %p = malloc(%llu)", device->ordinal, ptr, size);
	return ptr;
}

//------------------------------------------------------------------------------
static void host_memory_deallocate(const SP_Device* device, void* ptr) {
	L_TRACE("[VE#%i] free(%p)", device->ordinal, ptr);
	free(ptr);
}

//------------------------------------------------------------------------------
static void* unified_memory_allocate(const SP_Device* device, uint64_t bytes) {
	GUARD(device);
	VEDAdeviceptr ptr = 0;
	CVEDA(vedaMemAllocAsync(&ptr, bytes, 0));
	return (void*)ptr;
}

//------------------------------------------------------------------------------
static void unified_memory_deallocate(const SP_Device* device, void* location) {
	GUARD(device);
	CVEDA(vedaMemFreeAsync((VEDAdeviceptr)location, 0));
}

//------------------------------------------------------------------------------
static TF_Bool get_allocator_stats(const SP_Device* device, SP_AllocatorStats* stats) {
	return 1; // TODO
}

//------------------------------------------------------------------------------
static TF_Bool device_memory_usage(const SP_Device* device, int64_t* free, int64_t* total) {
	GUARD(device);
	size_t _free, _total;
	CVEDA(vedaMemGetInfo(&_free, &_total));
	*free	= (int64_t)_free;
	*total	= (int64_t)_total;
	return 1;
}

//------------------------------------------------------------------------------
static void create_stream(const SP_Device* device, SP_Stream* stream, TF_Status* status) {
	*stream = new SP_Stream_st;
	(*stream)->stream = 0;
	TF_SetStatus(status, TF_OK,	"");
}

//------------------------------------------------------------------------------
static void destroy_stream(const SP_Device* device, SP_Stream stream) {
	delete stream;
}

//------------------------------------------------------------------------------
static void create_stream_dependency(const SP_Device* device, SP_Stream dependent, SP_Stream other, TF_Status* status) {
	ASSERT(dependent->stream == other->stream);
	TF_SetStatus(status, TF_OK,	"");
}

//------------------------------------------------------------------------------
static void get_stream_status(const SP_Device* device, SP_Stream stream, TF_Status* status) {
	GUARD(device);
	switch(vedaStreamQuery(stream->stream)) {
		case VEDA_ERROR_VEO_STATE_UNKNOWN:	TF_SetStatus(status, TF_UNKNOWN ,	"VEDA_ERROR_VEO_STATE_UNKNOWN");	return;
		case VEDA_ERROR_VEO_STATE_RUNNING:	TF_SetStatus(status, TF_OK,			"VEDA_ERROR_VEO_STATE_RUNNING");	return;
		case VEDA_ERROR_VEO_STATE_SYSCALL:	TF_SetStatus(status, TF_OK,			"VEDA_ERROR_VEO_STATE_SYSCALL");	return;
		case VEDA_ERROR_VEO_STATE_BLOCKED:	TF_SetStatus(status, TF_OK,			"VEDA_ERROR_VEO_STATE_BLOCKED");	return;
		case VEDA_SUCCESS:					TF_SetStatus(status, TF_OK,			"VEDA_SUCCESS");					return;
	}
	FAIL();
}

//------------------------------------------------------------------------------
static void memcpy_dtoh(const SP_Device* device, SP_Stream stream, void* host_dst, const SP_DeviceMemoryBase* device_src, uint64_t size, TF_Status* status) {
	GUARD(device);
	CVEDA(vedaMemcpyDtoHAsync(host_dst, (VEDAdeviceptr)device_src->opaque, size, stream->stream));
	TF_SetStatus(status, TF_OK,	"");
}

//------------------------------------------------------------------------------
static void memcpy_htod(const SP_Device* device, SP_Stream stream, SP_DeviceMemoryBase* device_dst, const void* host_src, uint64_t size, TF_Status* status) {
	GUARD(device);
	CVEDA(vedaMemcpyHtoDAsync((VEDAdeviceptr)device_dst->opaque, host_src, size, stream->stream));
	TF_SetStatus(status, TF_OK,	"");
}

//------------------------------------------------------------------------------
static void memcpy_dtod(const SP_Device* device, SP_Stream stream, SP_DeviceMemoryBase* device_dst, const SP_DeviceMemoryBase* device_src, uint64_t size, TF_Status* status) {
	GUARD(device);
	CVEDA(vedaMemcpyDtoDAsync((VEDAdeviceptr)device_dst->opaque, (VEDAdeviceptr)device_src->opaque, size, stream->stream));
	TF_SetStatus(status, TF_OK,	"");
}

//------------------------------------------------------------------------------
static void sync_memcpy_dtoh(const SP_Device* device, void* host_dst, const SP_DeviceMemoryBase* device_src, uint64_t size, TF_Status* status) {
	GUARD(device);
	CVEDA(vedaMemcpyDtoH(host_dst, (VEDAdeviceptr)device_src->opaque, size));
	TF_SetStatus(status, TF_OK,	"");
}

//------------------------------------------------------------------------------
static void sync_memcpy_htod(const SP_Device* device, SP_DeviceMemoryBase* device_dst, const void* host_src, uint64_t size, TF_Status* status) {
	GUARD(device);
	CVEDA(vedaMemcpyHtoD((VEDAdeviceptr)device_dst->opaque, host_src, size));
	TF_SetStatus(status, TF_OK,	"");
}

//------------------------------------------------------------------------------
static void sync_memcpy_dtod(const SP_Device* device, SP_DeviceMemoryBase* device_dst, const SP_DeviceMemoryBase* device_src, uint64_t size, TF_Status* status) {
	GUARD(device);
	CVEDA(vedaMemcpyDtoD((VEDAdeviceptr)device_dst->opaque, (VEDAdeviceptr)device_src->opaque, size));
	TF_SetStatus(status, TF_OK,	"");
}

//------------------------------------------------------------------------------
static void block_host_for_event(const SP_Device* device, SP_Event event, TF_Status* status) {
	L_TRACE("block_host_for_event");
	TODO();
}

//------------------------------------------------------------------------------
static void block_host_until_done(const SP_Device* device, SP_Stream stream, TF_Status* status) {
	GUARD(device);
	CVEDA(vedaStreamSynchronize(stream->stream));
	TF_SetStatus(status, TF_OK,	"");
}

//------------------------------------------------------------------------------
static void synchronize_all_activity(const SP_Device* device, TF_Status* status) {
	GUARD(device);
	CVEDA(vedaCtxSynchronize());
	TF_SetStatus(status, TF_OK,	"");
}

//------------------------------------------------------------------------------
struct HostCallbackData {
	SE_StatusCallbackFn	func;
	void*				arg;
	inline HostCallbackData(SE_StatusCallbackFn _func, void* _arg) : func(_func), arg(_arg) {}
};

//------------------------------------------------------------------------------
static void host_callback_helper(VEDAstream stream, VEDAresult result, void* args) {
	L_TRACE("host_callback_helper");
	auto data = (HostCallbackData*)args;
	data->func(data->arg, 0);
	delete data;
}

//------------------------------------------------------------------------------
static TF_Bool host_callback(const SP_Device* device, SP_Stream stream, SE_StatusCallbackFn callback_fn, void* callback_arg) {
	GUARD(device);
	CVEDA(vedaStreamAddCallback(stream->stream, host_callback_helper, new HostCallbackData(callback_fn, callback_arg), 0));
	return true;
}

//------------------------------------------------------------------------------
static void create_timer(const SP_Device* device, SP_Timer* timer, TF_Status* status) {
	L_TRACE("[VE#%i] create_timer()", device->ordinal);
	TF_SetStatus(status, TF_OK,	"");
	*timer = new SP_Timer_st;
}

//------------------------------------------------------------------------------
static void destroy_timer(const SP_Device* device, SP_Timer timer) {
	L_TRACE("[VE#%i] destroy_timer()", device->ordinal);
	delete timer;
}

//------------------------------------------------------------------------------
inline uint64_t time_ns(void) {
	auto duration = std::chrono::system_clock::now().time_since_epoch();
	return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
}

//------------------------------------------------------------------------------
static void start_timer(const SP_Device* device, SP_Stream stream, SP_Timer timer, TF_Status* status) {
	L_TRACE("[VE#%i] start_timer()", device->ordinal);
	timer->start = time_ns();
	TF_SetStatus(status, TF_OK,	"");
}

//------------------------------------------------------------------------------
static void stop_timer(const SP_Device* device, SP_Stream stream, SP_Timer timer, TF_Status* status) {
	L_TRACE("[VE#%i] stop_timer()", device->ordinal);
	timer->end = time_ns();
	TF_SetStatus(status, TF_OK,	"");
}

//------------------------------------------------------------------------------
void create_stream_executor(const SP_Platform* platform, SE_CreateStreamExecutorParams* params, TF_Status* status) {
	params->stream_executor->allocate					= &allocate;
	params->stream_executor->block_host_for_event		= &block_host_for_event;
	params->stream_executor->block_host_until_done		= &block_host_until_done;
	params->stream_executor->create_event				= &create_event;
	params->stream_executor->create_stream				= &create_stream;
	params->stream_executor->create_stream_dependency	= &create_stream_dependency;
	params->stream_executor->create_timer				= &create_timer;
	params->stream_executor->deallocate					= &deallocate;
	params->stream_executor->destroy_event				= &destroy_event;
	params->stream_executor->destroy_stream				= &destroy_stream;
	params->stream_executor->destroy_timer				= &destroy_timer;
	params->stream_executor->device_memory_usage		= &device_memory_usage;
	params->stream_executor->get_allocator_stats		= &get_allocator_stats;
	params->stream_executor->get_event_status			= &get_event_status;
	params->stream_executor->get_stream_status			= &get_stream_status;
	params->stream_executor->host_callback				= &host_callback;
	params->stream_executor->host_memory_allocate		= &host_memory_allocate;
	params->stream_executor->host_memory_deallocate		= &host_memory_deallocate;
	params->stream_executor->memcpy_dtod				= &memcpy_dtod;
	params->stream_executor->memcpy_dtoh				= &memcpy_dtoh;
	params->stream_executor->memcpy_htod				= &memcpy_htod;
	params->stream_executor->record_event				= &record_event;
	params->stream_executor->start_timer				= &start_timer;
	params->stream_executor->stop_timer					= &stop_timer;
	params->stream_executor->sync_memcpy_dtod			= &sync_memcpy_dtod;
	params->stream_executor->sync_memcpy_dtoh			= &sync_memcpy_dtoh;
	params->stream_executor->sync_memcpy_htod			= &sync_memcpy_htod;
	params->stream_executor->synchronize_all_activity	= &synchronize_all_activity;
	params->stream_executor->unified_memory_allocate	= &unified_memory_allocate;
	params->stream_executor->unified_memory_deallocate	= &unified_memory_deallocate;
	params->stream_executor->wait_for_event				= &wait_for_event;
	TF_SetStatus(status, TF_OK,	"");
}

//------------------------------------------------------------------------------
#include "__ns.h"