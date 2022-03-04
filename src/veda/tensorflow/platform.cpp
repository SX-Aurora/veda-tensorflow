#include <veda/tensorflow/api.h>

#include "__ns.h"
//------------------------------------------------------------------------------
void create_stream_executor(const SP_Platform* platform, SE_CreateStreamExecutorParams* params, TF_Status* status);

//------------------------------------------------------------------------------
static uint64_t nanoseconds(SP_Timer timer) {
	return timer->end - timer->start;
}

//------------------------------------------------------------------------------
static void create_timer_fns(const SP_Platform* platform, SP_TimerFns* timer, TF_Status* status) {
	timer->nanoseconds = &nanoseconds;
	TF_SetStatus(status, TF_OK,	"");
}

//------------------------------------------------------------------------------
static void get_device_count(const SP_Platform* platform, int* device_count, TF_Status* status) {
	CVEDA(vedaDeviceGetCount(device_count));
	TF_SetStatus(status, TF_OK,	"");
}

//------------------------------------------------------------------------------
static	void destroy_device			(const SP_Platform* platform, SP_Device* device)					{}
static	void destroy_device_fns		(const SP_Platform* platform, SP_DeviceFns* device_fns)				{}
static	void destroy_platform		(SP_Platform* platform)												{}
static	void destroy_platform_fns	(SP_PlatformFns* platform_fns)										{}
static	void destroy_stream_executor(const SP_Platform* platform, SP_StreamExecutor* stream_executor)	{}
static	void destroy_timer_fns		(const SP_Platform* platform, SP_TimerFns* timer_fns)				{}

//------------------------------------------------------------------------------
static void create_device(const SP_Platform* platform, SE_CreateDeviceParams* params, TF_Status* status) {
	VEDAcontext ctx;
	CVEDA(vedaDevicePrimaryCtxRetain(&ctx, params->ordinal));
	VEDATensors_handle handle;
	CVEDA(veda_tensors_create_handle_by_ctx(&handle, ctx));

	params->device->device_handle	= handle;
	params->device->hardware_name	= "SX-Aurora TSUBASA";
	params->device->device_vendor	= "NEC";
	params->device->pci_bus_id		= "0000:00:00.0"; // TODO:
	TF_SetStatus(status, TF_OK,	"");
}

//------------------------------------------------------------------------------
static int32_t get_numa_node(const SP_Device* device) {
	/** We use the device->ordinal here, as this is about socket location */
	return device->ordinal;
}

//------------------------------------------------------------------------------
static int64_t get_memory_bandwidth(const SP_Device* device) {
	return -1;
}

//------------------------------------------------------------------------------
static double get_gflops(const SP_Device* device) {
	return -1;
}

//------------------------------------------------------------------------------
static void create_device_fns(const SP_Platform* platform, SE_CreateDeviceFnsParams* params, TF_Status* status) {
	params->device_fns->get_gflops				= &get_gflops;
	params->device_fns->get_memory_bandwidth	= &get_memory_bandwidth;
	params->device_fns->get_numa_node			= &get_numa_node;
	TF_SetStatus(status, TF_OK,	"");
}

//------------------------------------------------------------------------------
#include "__ns.h"

extern "C" void SE_InitPlugin(SE_PlatformRegistrationParams* params, TF_Status* status) {
	using namespace veda::tensorflow;

	L_TRACE(">> SE_InitPlugin");
	CVEDA(vedaInit(0));

	params->destroy_platform						= &destroy_platform;
	params->destroy_platform_fns					= &destroy_platform_fns;

	params->platform->name							= "NEC_VECTOR_ENGINE";
	params->platform->type							= "VE";
	params->platform->supports_unified_memory		= true;
	params->platform->use_bfc_allocator				= false;

	params->platform_fns->create_device				= &create_device;
	params->platform_fns->create_device_fns			= &create_device_fns;
	params->platform_fns->create_stream_executor	= &create_stream_executor;
	params->platform_fns->create_timer_fns			= &create_timer_fns;
	params->platform_fns->destroy_device			= &destroy_device;
	params->platform_fns->destroy_device_fns		= &destroy_device_fns;
	params->platform_fns->destroy_stream_executor	= &destroy_stream_executor;
	params->platform_fns->destroy_timer_fns			= &destroy_timer_fns;
	params->platform_fns->get_device_count			= &get_device_count;
	
	L_TRACE("<< SE_InitPlugin");
}