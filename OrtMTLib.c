#include "OrtMTLib.h"


//const OrtApi* g_ort = NULL;
//OrtEnv* env = NULL;
//OrtSession* session = NULL;
//OrtSessionOptions* session_options = NULL;
//OrtMemoryInfo* memory_info = NULL;

int create_ort_session(char* model_path_char, int n_threads)
{
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    ORT_ABORT_ON_ERROR(g_ort->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "OrtMTLib", &env));
    ORT_ABORT_ON_ERROR(g_ort->CreateSessionOptions(&session_options));
    ORT_ABORT_ON_ERROR(g_ort->SetIntraOpNumThreads(session_options, n_threads));
    ORT_ABORT_ON_ERROR(g_ort->SetInterOpNumThreads(session_options, n_threads));
    ORT_ABORT_ON_ERROR(g_ort->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_ALL));
#ifndef USE_DML
    ORT_ABORT_ON_ERROR(g_ort->SetSessionExecutionMode(session_options, ORT_PARALLEL));
#endif
#ifdef USE_DML
    ORT_ABORT_ON_ERROR(g_ort->SetSessionExecutionMode(session_options, ORT_SEQUENTIAL));
    ORT_ABORT_ON_ERROR(g_ort->DisableMemPattern(session_options));
    OrtSessionOptionsAppendExecutionProvider_DML(session_options, 0);
#endif


    size_t model_path_len = strlen(model_path_char);
    size_t model_path_wlen;
    wchar_t* model_path_wchar = (wchar_t*) malloc((model_path_len + 1) * sizeof(wchar_t));
    mbstowcs_s(&model_path_wlen, model_path_wchar, model_path_len + 1, model_path_char, model_path_len + 1);

    ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, model_path_wchar, session_options, &session));

    //ORT_ABORT_ON_ERROR(g_ort->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &memory_info));
    //ORT_ABORT_ON_ERROR(g_ort->CreateAllocator(session, memory_info, &allocator));
    ORT_ABORT_ON_ERROR(g_ort->GetAllocatorWithDefaultOptions(&allocator));
    memory_info = allocator->Info(allocator);
    return 0;
}
int create_tensor_float(float* tensor_data, size_t data_len, int64_t* shape, size_t shape_len, void** tensor)
{
    ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(
            memory_info,
            tensor_data, sizeof(float) * data_len,
            shape, shape_len,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            (OrtValue**)tensor));
    return 0;
}
int create_tensor_int32(int32_t* tensor_data, size_t data_len, int64_t* shape, size_t shape_len, void** tensor)
{
    ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(
            memory_info,
            tensor_data, sizeof(int32_t) * data_len,
            shape, shape_len,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
            (OrtValue**)tensor));
    return 0;
}
int print_tensor_int32(void* tensor)
{
    OrtValue* output_tensor = (OrtValue*)tensor;
    OrtTensorTypeAndShapeInfo* info;
    size_t output_len;
    ORT_ABORT_ON_ERROR(g_ort->GetTensorTypeAndShape(output_tensor, &info));
    ORT_ABORT_ON_ERROR(g_ort->GetTensorShapeElementCount(info, &output_len));
    int32_t* output_ids;
    ORT_ABORT_ON_ERROR(g_ort->GetTensorMutableData(output_tensor, (void**)&output_ids));
    for (size_t i = 0; i < output_len; i++)
    {
        printf("%d, ", output_ids[i]);
    }
    printf("\n");
    return 0;
}
size_t get_input_count(void)
{
    size_t num_input_nodes;
    ORT_ABORT_ON_ERROR(g_ort->SessionGetInputCount(session, &num_input_nodes));
    return num_input_nodes;
}
size_t get_output_count(void)
{
    size_t num_output_nodes;
    ORT_ABORT_ON_ERROR(g_ort->SessionGetOutputCount(session, &num_output_nodes));
    return num_output_nodes;
}
int run_session(void** tensors, char* output_name, int** output_ids_raw, size_t* output_len)
{
    OrtValue** input_tensors = (OrtValue**)tensors;
    size_t input_count = get_input_count();
    char** input_names_list = (char**)malloc(input_count * sizeof(*input_names_list));
    for (size_t i = 0; i < input_count; i++)
    {
        ORT_ABORT_ON_ERROR(g_ort->SessionGetInputName(session, i, allocator, &input_names_list[i]));
    }
    OrtValue* output_tensor = NULL;

    ORT_ABORT_ON_ERROR(g_ort->Run(
            session, NULL,
            input_names_list, input_tensors, input_count,
            &output_name, 1, &output_tensor
            ));

    OrtTensorTypeAndShapeInfo* info;
    ORT_ABORT_ON_ERROR(g_ort->GetTensorTypeAndShape(output_tensor, &info));
    ORT_ABORT_ON_ERROR(g_ort->GetTensorShapeElementCount(info, output_len));
    int* output_ids;
    ORT_ABORT_ON_ERROR(g_ort->GetTensorMutableData(output_tensor, (void**)&output_ids));
    size_t out_memlen = (*output_len) * sizeof(int);
    *output_ids_raw = malloc(out_memlen);
    memcpy(*output_ids_raw, output_ids, out_memlen);
    return 0;
}

int release_all_globals()
{
    g_ort->ReleaseAllocator(allocator);
    g_ort->ReleaseSessionOptions(session_options);
    g_ort->ReleaseSession(session);
    g_ort->ReleaseEnv(env);
    return 0;
}

int release_ort_tensor(void* tensor)
{
    g_ort->ReleaseValue(tensor);
    return 0;
}
