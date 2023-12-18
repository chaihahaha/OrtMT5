#include "OrtMTLib.h"


extern "C"
{
    const OrtApi* g_ort = nullptr;
    OrtEnv* env = nullptr;
    OrtSession* session = nullptr;
    OrtSessionOptions* session_options = nullptr;
    OrtAllocator* allocator = nullptr;
    const OrtMemoryInfo* memory_info = nullptr;

    __declspec(dllexport) int create_ort_api(void)
    {
        g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        return 0;
    }
    __declspec(dllexport) int create_ort_session(char* model_path_char)
    {
        std::string model_path(model_path_char);

        ORT_ABORT_ON_ERROR(g_ort->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "OrtMTLib", &env));
        ORT_ABORT_ON_ERROR(g_ort->CreateSessionOptions(&session_options));
        ORT_ABORT_ON_ERROR(g_ort->SetSessionGraphOptimizationLevel(session_options, ORT_DISABLE_ALL));
        g_ort->SetSessionExecutionMode(session_options, ORT_PARALLEL);

#ifdef _WIN32
        std::wstring model_path_wstring = str2wstr(model_path);
        const wchar_t* model_path_wchar = model_path_wstring.c_str();
#endif

        if (!std::filesystem::exists(model_path))
        {
            std::cout << "Incorrect model path" << std::endl;
            return -1;
        }

#ifdef _WIN32
        ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, model_path_wchar, session_options, &session));
#else
        ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, model_path.c_str(), session_options, &session));
#endif
        ORT_ABORT_ON_ERROR(g_ort->GetAllocatorWithDefaultOptions(&allocator));
        //ORT_ABORT_ON_ERROR(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
        ORT_ABORT_ON_ERROR(g_ort->AllocatorGetInfo(allocator, &memory_info));
        //ORT_ABORT_ON_ERROR(g_ort->SessionGetInputCount(session, num_input_nodes));
        //ORT_ABORT_ON_ERROR(g_ort->SessionGetOutputCount(session, num_output_nodes));

        //OrtAllocator* allocator;
        //ORT_ABORT_ON_ERROR(g_ort->GetAllocatorWithDefaultOptions(&allocator));

        //std::vector<char*> input_names_list;
        //// iterate over all input nodes
        //for (size_t i = 0; i < *num_input_nodes; i++)
        //{
        //    char* input_name;
        //    ORT_ABORT_ON_ERROR(g_ort->SessionGetInputName(session, i, allocator, &input_name));
        //    input_names_list.push_back(input_name);
        //}
        //*input_names = input_names_list.data();

        //std::vector<char*> output_names_list;
        //// iterate over all input nodes
        //for (size_t i = 0; i < *num_output_nodes; i++)
        //{
        //    char* output_name;
        //    ORT_ABORT_ON_ERROR(g_ort->SessionGetOutputName(session, i, allocator, &output_name));
        //    output_names_list.push_back(output_name);
        //}
        //*output_names = output_names_list.data();
        return 0;
    }
    __declspec(dllexport) int run_session(int max_length, int min_length, int num_beams, int num_return_sequences, float length_penalty, float repetition_penalty, int* input_ids_raw, size_t input_len, int** output_ids_raw, size_t* output_len)
    {

        std::vector<int64_t> one = {1};

        OrtValue* max_length_tensor = NULL;
        ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(
                memory_info,
                &max_length, sizeof(int32_t),
                one.data(), 1,
                ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
                &max_length_tensor));
        
        OrtValue* min_length_tensor = NULL;
        ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(
                memory_info,
                &min_length, sizeof(int32_t),
                one.data(), 1,
                ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
                &min_length_tensor));
        
        OrtValue* num_beams_tensor = NULL;
        ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(
                memory_info,
                &num_beams, sizeof(int32_t),
                one.data(), 1,
                ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
                &num_beams_tensor));
        
        OrtValue* num_return_sequences_tensor = NULL;
        ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(
                memory_info,
                &num_return_sequences, sizeof(int32_t),
                one.data(), 1,
                ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
                &num_return_sequences_tensor));
        
        OrtValue* length_penalty_tensor = NULL;
        ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(
                memory_info,
                &length_penalty, sizeof(float),
                one.data(), 1,
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                &length_penalty_tensor));
        
        OrtValue* repetition_penalty_tensor = NULL;
        ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(
                memory_info,
                &repetition_penalty, sizeof(float),
                one.data(), 1,
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                &repetition_penalty_tensor));
        
        std::vector<int> input_ids(input_ids_raw, input_ids_raw + input_len);
        std::vector<int64_t> input_ids_dims{1, (int64_t)input_ids.size()};

        OrtValue* input_ids_tensor = NULL;
        ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(
                memory_info,
                input_ids.data(), sizeof(int32_t) * input_ids.size(),
                input_ids_dims.data(),1 * input_ids_dims.size(),
                ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
                &input_ids_tensor));
        
        std::vector<OrtValue*> input_tensors;
        input_tensors.push_back(input_ids_tensor);
        input_tensors.push_back(max_length_tensor);
        input_tensors.push_back(min_length_tensor);
        input_tensors.push_back(num_beams_tensor);
        input_tensors.push_back(num_return_sequences_tensor);
        input_tensors.push_back(length_penalty_tensor);
        input_tensors.push_back(repetition_penalty_tensor);

        //g_ort->ReleaseMemoryInfo(memory_info);

        ////////// copied from create_session
        size_t num_input_nodes, num_output_nodes;
        ORT_ABORT_ON_ERROR(g_ort->SessionGetInputCount(session, &num_input_nodes));
        ORT_ABORT_ON_ERROR(g_ort->SessionGetOutputCount(session, &num_output_nodes));

        std::vector<char*> input_names_list;
        // iterate over all input nodes
        for (size_t i = 0; i < num_input_nodes; i++)
        {
            char* input_name;
            ORT_ABORT_ON_ERROR(g_ort->SessionGetInputName(session, i, allocator, &input_name));
            input_names_list.push_back(input_name);
        }
        //char** input_names = input_names_list.data();

        std::vector<char*> output_names_list;
        // iterate over all input nodes
        for (size_t i = 0; i < num_output_nodes; i++)
        {
            char* output_name;
            ORT_ABORT_ON_ERROR(g_ort->SessionGetOutputName(session, i, allocator, &output_name));
            output_names_list.push_back(output_name);
        }
        //char** output_names = output_names_list.data();
        /////////////////copy end

        std::vector<int64_t> output_dims = {1, 1, max_length};
        OrtValue* output_tensor = NULL;
        
        ORT_ABORT_ON_ERROR(g_ort->Run(
                session, NULL,
                input_names_list.data(), input_tensors.data(), input_names_list.size(),
                output_names_list.data(), output_names_list.size(), &output_tensor
                ));

        OrtTensorTypeAndShapeInfo* info;
        ORT_ABORT_ON_ERROR(g_ort->GetTensorTypeAndShape(output_tensor, &info));
        ORT_ABORT_ON_ERROR(g_ort->GetTensorShapeElementCount(info, output_len));
        int* output_ids;
        ORT_ABORT_ON_ERROR(g_ort->GetTensorMutableData(output_tensor, (void**)&output_ids));
        *output_ids_raw = output_ids;
        //for (int i = 0; i < *output_len; i++)
        //{
        //    std::cout << output_ids[i] << ", ";
        //}
        //std::cout << std::endl;

        for (int i = 0; i < input_tensors.size(); i++)
        {
            g_ort->ReleaseValue(input_tensors[i]);
        }
        g_ort->ReleaseValue(output_tensor);
        std::cout << "released values" << std::endl;
        g_ort->ReleaseTensorTypeAndShapeInfo(info);
        return 0;
    }

    __declspec(dllexport) int release()
    {
        //g_ort->ReleaseMemoryInfo((OrtMemoryInfo*)memory_info);
        g_ort->ReleaseAllocator(allocator);
        g_ort->ReleaseSessionOptions(session_options);
        g_ort->ReleaseSession(session);
        g_ort->ReleaseEnv(env);
        return 0;
    }
}

int main()
{
    create_ort_api();
    create_ort_session("../mt5-ja_zh_beam_search.onnx");
    std::vector<int> input_ids = {1042, 462, 338, 12001, 669, 14942, 43556};
    int* output_ids;
    size_t output_len;
    run_session(128, 1, 1, 1, (float)1.3, (float)1.3, input_ids.data(), input_ids.size(), &output_ids, &output_len);
    std::cout << "finished inference" << std::endl;
    return 0;
}
