#include "onnxruntime_cxx_api.h"
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <thread>
#include <atomic>
#include <functional>
#include <future>

#pragma comment(lib, "onnxruntime.lib")

//static std::thread::id caller_tid = std::this_thread::get_id();
//static std::atomic_int thread_cnt{0};
static std::atomic_bool atomic_wait{false};
//static std::thread::id running_inference_thread_id{};
std::unique_ptr<Ort::Session> session = nullptr;
std::unique_ptr<Ort::Env> env = nullptr;

//#define MAX_THREADS 999
//static std::array<std::atomic_bool, MAX_THREADS> atomic_wait = {false};

void print_tensor(Ort::Value& tensor)
{
    Ort::TensorTypeAndShapeInfo ts = tensor.GetTensorTypeAndShapeInfo();

    const int32_t* el = tensor.GetTensorData<int32_t>();
    for (int i = 0; i < ts.GetElementCount(); i++)
    {
        std::cout << el[i] << " ";
    }
    std::cout << std::endl;
    return;
}

void AsyncCallback(void* user_data, OrtValue** outputs, size_t num_outputs, OrtStatusPtr status_ptr) {
    //const int old_thread_cnt = *reinterpret_cast<const int*>(user_data);
    OrtValue** input_tensors = reinterpret_cast<OrtValue**>(user_data);
    for (int i=0; i < 5; i++)
    {
        Ort::Value input_value(input_tensors[i]);
        print_tensor(input_value);
        input_value.release();
    }
    Ort::Value output_value(outputs[0]);
    //if (old_thread_cnt == thread_cnt.load())
    std::cout << "output" << std::endl;
    print_tensor(output_value);
    //output_value.release();
    //atomic_wait[old_thread_cnt].store(true);
    atomic_wait.store(true);
    std::cout << "callback" << std::endl;
}

int wmain(int argc, wchar_t* argv[])
{
    if (argc != 8)
    {
        std::cout << "Incorrect number of commandline args,"
            << "should be `OrtMT5.exe [model_path] [max_length] [min_length] "
            << "[num_beams] [num_return_sequences] [length_penalty] [repetition_penalty]`" << std::endl;
        return -1;
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
    int max_threads = std::thread::hardware_concurrency();
    //session_options.SetIntraOpNumThreads(max_threads/2);
    //session_options.SetInterOpNumThreads(max_threads/2);
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);
    //session_options.DisablePerSessionThreads();

    //Ort::ThreadingOptions th_opt;
    //th_opt.SetGlobalSpinControl(1);
    //th_opt.SetGlobalIntraOpNumThreads(max_threads/2);
    //th_opt.SetGlobalInterOpNumThreads(max_threads/2);

    //auto env_local = std::make_unique<Ort::Env>(th_opt, OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, "OrtMT5");
    auto env_local = std::make_unique<Ort::Env>(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, "OrtMT5");
    env = std::move(env_local);

    const wchar_t* model_path = argv[1];

    if (!std::filesystem::exists(model_path))
    {
        std::cout << "Incorrect model path" << std::endl;
        return -1;
    }
    auto session_local = std::make_unique<Ort::Session>(*env, model_path, session_options);
    session = std::move(session_local);

    Ort::AllocatorWithDefaultOptions allocator;

    const size_t num_input_nodes = session->GetInputCount();
    std::vector<Ort::AllocatedStringPtr> input_names_ptr;
    std::vector<const char*> input_node_names;
    input_node_names.reserve(num_input_nodes);
    std::vector<int64_t> input_node_dims;

    // iterate over all input nodes
    for (size_t i = 0; i < num_input_nodes; i++) {
        auto input_name = session->GetInputNameAllocated(i, allocator);
        input_node_names.push_back(input_name.get());
        input_names_ptr.push_back(std::move(input_name));
    }

    //size_t input_count = session->GetInputCount();
    int32_t input_length = -1;
    int32_t input_value = -1;
    std::vector<std::vector<Ort::Value>> input_tensors_pool;
    std::vector<std::vector<Ort::Value>> output_tensors_pool;

    int scnt = 0;

    while (std::cin >> input_length)
    {
        if (input_length <= 0)
        {
            std::cout << "invalid input, not a positive integer" << std::endl;
            return -1;
        }
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
        

        //auto a = std::async(
        //        std::launch::async,
        //        run_async,
        //        py_input_ids,
        //        max_length_int,
        //        min_length_int,
        //        num_beams_int,
        //        num_return_sequences_int,
        //        length_penalty_float,
        //        repetition_penalty_float,
        //        input_node_names,
        //        allocator
        //);
        Ort::ConstMemoryInfo memory_info = allocator.GetInfo();
        //auto tid = std::this_thread::get_id();

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
        std::vector<int64_t> input_ids_dims{1, (int64_t)input_ids_values.size()};
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
        input_tensors_pool.push_back(std::move(input_tensors));

        Ort::AllocatedStringPtr output_name_ptr = session->GetOutputNameAllocated(0, allocator);
        const char* output_name = output_name_ptr.get();

        std::vector<const char*> output_names{output_name};
        output_names.resize(1);
        std::vector<int64_t> output_ids_dims{1, 1, max_length_int};
        std::vector<Ort::Value> output_tensors;

        output_tensors.push_back(
                Ort::Value::CreateTensor<int32_t>(
                    allocator,
                    output_ids_dims.data(),
                    output_ids_dims.size()
                )
        );
        output_tensors_pool.push_back(std::move(output_tensors));

        //thread_cnt.store( (thread_cnt.load() + 1) % MAX_THREADS);
        //int old_thread_cnt = thread_cnt.load() % MAX_THREADS;
        std::cout << "calling runasync" <<std::endl;
        session->RunAsync(
            Ort::RunOptions{nullptr},
            input_node_names.data(),
            input_tensors_pool[scnt].data(),
            input_tensors_pool[scnt].size(),
            output_names.data(),
            output_tensors_pool[scnt].data(),
            output_names.size(),
            AsyncCallback,
            input_tensors_pool[scnt].data()
            );
        std::chrono::duration<double, std::milli> dur{100};
        //while (!atomic_wait.load())
        //    std::this_thread::sleep_for(dur);
        std::this_thread::sleep_for(dur);
        //atomic_wait.store(false);
        scnt += 1;
        std::cout << "called runasync" <<std::endl;
    }
    std::chrono::duration<double, std::milli> dur{10000};
    std::this_thread::sleep_for(dur);
    return 0;
}
