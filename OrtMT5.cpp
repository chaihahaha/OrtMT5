#include "onnxruntime_cxx_api.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <string>
#include <filesystem>

#pragma comment(lib, "onnxruntime.lib")

int wmain(int argc, wchar_t* argv[])
{
    try
    {
        assert(argc == 8);
    }
    catch (const std::exception& e)
    {
        std::cout << "Caught " << e.what() << std::endl;
        std::cout << "Incorrect number of commandline args,"
            << "should be `OrtMT5.exe [model_path] [max_length] [min_length] "
            << "[num_beams] [num_return_sequences] [length_penalty] [repetition_penalty]`" << std::endl;
    }

    int32_t max_length_int;
    int32_t min_length_int;
    int32_t num_beams_int;
    int32_t num_return_sequences_int;
    float length_penalty_float;
    float repetition_penalty_float;
    try
    {
        max_length_int = std::stoi(argv[2]);
        min_length_int = std::stoi(argv[3]);
        num_beams_int = std::stoi(argv[4]);
        num_return_sequences_int = std::stoi(argv[5]);
        length_penalty_float = std::stof(argv[6]);
        repetition_penalty_float = std::stof(argv[7]);
    }
    catch (const std::exception& e)
    {
        std::cout << "Caught " << e.what() << std::endl;
        std::cout << "Incorrect commandline args,"
            << "should be `OrtMT5.exe [model_path] [max_length] [min_length] "
            << "[num_beams] [num_return_sequences] [length_penalty] [repetition_penalty]`" << std::endl;
        return -1;
    }
    
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(8);
    session_options.SetInterOpNumThreads(8);
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);
    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, "OrtMT5");
    const wchar_t* model_path = argv[1];

    if (!std::filesystem::exists(model_path))
    {
        std::cout << "Incorrect model path" << std::endl;
        return -1;
    }
    Ort::Session session = Ort::Session(env, model_path, session_options);

    Ort::AllocatorWithDefaultOptions allocator;

    const size_t num_input_nodes = session.GetInputCount();
    std::vector<Ort::AllocatedStringPtr> input_names_ptr;
    std::vector<const char*> input_node_names;
    input_node_names.reserve(num_input_nodes);
    std::vector<int64_t> input_node_dims;

    // iterate over all input nodes
    for (size_t i = 0; i < num_input_nodes; i++) {
        auto input_name = session.GetInputNameAllocated(i, allocator);
        input_node_names.push_back(input_name.get());
        input_names_ptr.push_back(std::move(input_name));
    }

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    //size_t input_count = session.GetInputCount();
    int32_t input_length = -1;
    int32_t input_value = -1;

    while (std::cin >> input_length)
    {
        if (input_length > 0)
        {
            std::vector<int32_t> py_input_ids{};
            for (int i = 0; i < input_length; i++)
            {
                if (std::cin >> input_value)
                {
                    py_input_ids.push_back(input_value);
                }
                else
                {
                    std::cout << "invalid input, not a integer" << std::endl;
                    return -1;
                }
            }
            
            std::vector<int32_t> max_length_values{ max_length_int };
            std::vector<int64_t> max_length_dims{ 1 };
            Ort::Value max_length = Ort::Value::CreateTensor<int32_t>(memory_info, max_length_values.data(), max_length_values.size(),
                max_length_dims.data(), max_length_dims.size());

            std::vector<int32_t> min_length_values{ min_length_int };
            std::vector<int64_t> min_length_dims{ 1 };
            Ort::Value min_length = Ort::Value::CreateTensor<int32_t>(memory_info, min_length_values.data(), min_length_values.size(),
                min_length_dims.data(), min_length_dims.size());

            std::vector<int32_t> num_beams_values{ num_beams_int };
            std::vector<int64_t> num_beams_dims{ 1 };
            Ort::Value num_beams = Ort::Value::CreateTensor<int32_t>(memory_info, num_beams_values.data(), num_beams_values.size(),
                num_beams_dims.data(), num_beams_dims.size());

            std::vector<int32_t> num_return_sequences_values{ num_return_sequences_int };
            std::vector<int64_t> num_return_sequences_dims{ 1 };
            Ort::Value num_return_sequences = Ort::Value::CreateTensor<int32_t>(memory_info, num_return_sequences_values.data(), num_return_sequences_values.size(),
                num_return_sequences_dims.data(), num_return_sequences_dims.size());

            std::vector<float> length_penalty_values{ (float)length_penalty_float };
            std::vector<int64_t> length_penalty_dims{ 1 };
            Ort::Value length_penalty = Ort::Value::CreateTensor<float>(memory_info, length_penalty_values.data(), length_penalty_values.size(),
                length_penalty_dims.data(), length_penalty_dims.size());

            std::vector<float> repetition_penalty_values{ (float)repetition_penalty_float };
            std::vector<int64_t> repetition_penalty_dims = { 1 };
            Ort::Value repetition_penalty = Ort::Value::CreateTensor<float>(memory_info, repetition_penalty_values.data(), repetition_penalty_values.size(),
                repetition_penalty_dims.data(), repetition_penalty_dims.size());

            std::vector<int32_t> input_ids_values = py_input_ids;
            std::array<int64_t, 2> input_ids_dims{1, input_ids_values.size()};
            Ort::Value input_ids = Ort::Value::CreateTensor<int32_t>(memory_info, input_ids_values.data(), input_ids_values.size(),
                input_ids_dims.data(), input_ids_dims.size());

            std::vector<Ort::Value> input_tensors;
            input_tensors.push_back(std::move(input_ids));
            input_tensors.push_back(std::move(max_length));
            input_tensors.push_back(std::move(min_length));
            input_tensors.push_back(std::move(num_beams));
            input_tensors.push_back(std::move(num_return_sequences));
            input_tensors.push_back(std::move(length_penalty));
            input_tensors.push_back(std::move(repetition_penalty));

            Ort::AllocatedStringPtr output_name_ptr = session.GetOutputNameAllocated(0, allocator);
            const char* output_name = output_name_ptr.get();

            std::vector<const char*> output_names{output_name};
            output_names.resize(1);
            //size_t output_count = 1;

            std::vector<Ort::Value> output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), input_tensors.data(), input_tensors.size(), output_names.data(), output_names.size());
            Ort::Value output_tensor = std::move(output_tensors[0]);

            Ort::TensorTypeAndShapeInfo ts = output_tensor.GetTensorTypeAndShapeInfo();

            const int32_t* output_tokens = output_tensor.GetTensorData<int32_t>();
            for (int i = 0; i < ts.GetElementCount(); i++)
            {
                std::cout << output_tokens[i] << " ";
            }
            std::cout << std::endl;
        }
        else
        {
            std::cout << "invalid input, not a positive integer" << std::endl;
            return -1;
        }
    }
    return 0;
}
