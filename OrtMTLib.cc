#include "OrtMTLib.h"


//extern "C"
//{
    const OrtApi* g_ort = NULL;
    OrtEnv* env = NULL;
    OrtSession* session = NULL;
    OrtSessionOptions* session_options = NULL;
    OrtAllocator* allocator = NULL;
    const OrtMemoryInfo* memory_info = NULL;

    int create_ort_api(void)
    {
        g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        return 0;
    }
    int create_ort_session(char* model_path_char)
    {

        ORT_ABORT_ON_ERROR(g_ort->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "OrtMTLib", &env));
        ORT_ABORT_ON_ERROR(g_ort->CreateSessionOptions(&session_options));
        ORT_ABORT_ON_ERROR(g_ort->SetSessionGraphOptimizationLevel(session_options, ORT_DISABLE_ALL));
        g_ort->SetSessionExecutionMode(session_options, ORT_SEQUENTIAL);

        int bufsize = MultiByteToWideChar(CP_ACP, 0, model_path_char, -1, NULL, 0);
        wchar_t* model_path_wchar = new wchar_t[bufsize];
        MultiByteToWideChar(CP_ACP, 0, model_path_char, -1, model_path_wchar, bufsize);

        ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, model_path_wchar, session_options, &session));
        ORT_ABORT_ON_ERROR(g_ort->GetAllocatorWithDefaultOptions(&allocator));
        //ORT_ABORT_ON_ERROR(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
        ORT_ABORT_ON_ERROR(g_ort->AllocatorGetInfo(allocator, &memory_info));
        //ORT_ABORT_ON_ERROR(g_ort->SessionGetInputCount(session, num_input_nodes));
        //ORT_ABORT_ON_ERROR(g_ort->SessionGetOutputCount(session, num_output_nodes));

        //OrtAllocator* allocator;
        //ORT_ABORT_ON_ERROR(g_ort->GetAllocatorWithDefaultOptions(&allocator));

        return 0;
    }
    int run_session(int max_length, int min_length, int num_beams, int num_return_sequences, float length_penalty, float repetition_penalty, int32_t* input_ids_raw, size_t input_len, int** output_ids_raw, size_t* output_len)
    {

        int64_t* one = (int64_t*)malloc(sizeof(int64_t));
        one[0] = 1;

        OrtValue* max_length_tensor = NULL;
        ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(
                memory_info,
                &max_length, sizeof(int32_t),
                one, 1,
                ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
                &max_length_tensor));
        
        OrtValue* min_length_tensor = NULL;
        ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(
                memory_info,
                &min_length, sizeof(int32_t),
                one, 1,
                ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
                &min_length_tensor));
        
        OrtValue* num_beams_tensor = NULL;
        ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(
                memory_info,
                &num_beams, sizeof(int32_t),
                one, 1,
                ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
                &num_beams_tensor));
        
        OrtValue* num_return_sequences_tensor = NULL;
        ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(
                memory_info,
                &num_return_sequences, sizeof(int32_t),
                one, 1,
                ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
                &num_return_sequences_tensor));
        
        OrtValue* length_penalty_tensor = NULL;
        ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(
                memory_info,
                &length_penalty, sizeof(float),
                one, 1,
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                &length_penalty_tensor));
        
        OrtValue* repetition_penalty_tensor = NULL;
        ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(
                memory_info,
                &repetition_penalty, sizeof(float),
                one, 1,
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                &repetition_penalty_tensor));
        
        int32_t* input_ids = (int32_t*)malloc(input_len * sizeof(int32_t));
        for (int i = 0; i < input_len; i++)
            input_ids[i] = input_ids_raw[i];
        int64_t* input_ids_dims = (int64_t*)malloc(2 * sizeof(int64_t));
        input_ids_dims[0] = 1;
        input_ids_dims[1] = input_len;

        OrtValue* input_ids_tensor = NULL;
        ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(
                memory_info,
                input_ids, sizeof(int32_t) * input_len,
                input_ids_dims,1 * 2,
                ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
                &input_ids_tensor));
        
        OrtValue** input_tensors = (OrtValue**)malloc(7 * sizeof(*input_tensors));
        input_tensors[0] = (input_ids_tensor);
        input_tensors[1] = (max_length_tensor);
        input_tensors[2] = (min_length_tensor);
        input_tensors[3] = (num_beams_tensor);
        input_tensors[4] = (num_return_sequences_tensor);
        input_tensors[5] = (length_penalty_tensor);
        input_tensors[6] = (repetition_penalty_tensor);

        //g_ort->ReleaseMemoryInfo(memory_info);

        ////////// copied from create_session
        size_t num_input_nodes, num_output_nodes;
        ORT_ABORT_ON_ERROR(g_ort->SessionGetInputCount(session, &num_input_nodes));
        ORT_ABORT_ON_ERROR(g_ort->SessionGetOutputCount(session, &num_output_nodes));

        char** input_names_list = (char**)malloc(7 * sizeof(*input_names_list));
        input_names_list[0] = "input_ids";
        input_names_list[1] = "max_length";
        input_names_list[2] = "min_length";
        input_names_list[3] = "num_beams";
        input_names_list[4] = "num_return_sequences";
        input_names_list[5] = "length_penalty";
        input_names_list[6] = "repetition_penalty";


        char** output_names_list = (char**)malloc(sizeof(*output_names_list));
        output_names_list[0] = "sequences";

        int64_t* output_dims = (int64_t*)malloc(3 * sizeof(int64_t));
        output_dims[0] = 1;
        output_dims[1] = 1;
        output_dims[2] = max_length;
        OrtValue* output_tensor = NULL;
        
        ORT_ABORT_ON_ERROR(g_ort->Run(
                session, NULL,
                input_names_list, input_tensors, 7,
                output_names_list, 1, &output_tensor
                ));

        OrtTensorTypeAndShapeInfo* info;
        ORT_ABORT_ON_ERROR(g_ort->GetTensorTypeAndShape(output_tensor, &info));
        ORT_ABORT_ON_ERROR(g_ort->GetTensorShapeElementCount(info, output_len));
        int* output_ids;
        ORT_ABORT_ON_ERROR(g_ort->GetTensorMutableData(output_tensor, (void**)&output_ids));
        *output_ids_raw = output_ids;


        for (int i = 0; i < 7; i++)
        {
            g_ort->ReleaseValue(input_tensors[i]);
        }
        g_ort->ReleaseValue(output_tensor);
        g_ort->ReleaseTensorTypeAndShapeInfo(info);

        return 0;
    }

    int release()
    {
        //g_ort->ReleaseMemoryInfo((OrtMemoryInfo*)memory_info);
        //g_ort->ReleaseAllocator(allocator);
        //g_ort->ReleaseSessionOptions(session_options);
        //g_ort->ReleaseSession(session);
        //g_ort->ReleaseEnv(env);
        return 0;
    }
//}

int main()
{
    create_ort_api();
    create_ort_session("D:/source/OrtMT5/mt5-ja_zh_beam_search.onnx");
    int input_ids[] = {1042, 462, 338, 12001, 669, 14942, 43556};
    int* output_ids;
    size_t output_len;
        
    run_session(128, 1, 1, 1, (float)1.3, (float)1.3, (int*)&input_ids, 7, &output_ids, &output_len);
    return 0;
}
