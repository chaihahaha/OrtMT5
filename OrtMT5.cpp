#include "OrtMT5.h"

#pragma comment(lib, "onnxruntime.lib")

#define MAX_ASYNC_RUNS 6000

//static std::atomic_int thread_cnt{0};
static std::atomic_bool atomic_printing{false};
std::unique_ptr<Ort::Session> session = nullptr;
std::unique_ptr<Ort::Env> env = nullptr;
std::unique_ptr<sentencepiece::SentencePieceProcessor> sp = nullptr;
static size_t num_input_nodes = 0;
static int max_threads = -1;

void clear_ortvalue_vector(OrtValue** a, size_t len)
{
#ifdef DEBUG
    std::cout << "releasing " << len << std::endl;
#endif
    for (size_t i = 0; i < len; i++)
    {
        Ort::Value v = Ort::Value(a[i]);
        v.release();
    }
}
std::wstring str2wstr(std::string s)
{
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    std::wstring ws = converter.from_bytes(s.c_str());
    return ws;
}

std::string wstr2str(std::wstring ws)
{
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    std::string s = converter.to_bytes(ws);
    return s;
}

std::string to_utf8str(std::string ws)
{
    icu::UnicodeString us(ws.c_str());
    //std::wstring_convert<std::codecvt_utf8<char>> converter;
    //std::string s = converter.to_bytes(ws);
    std::string s;
    us.toUTF8String(s);
    return s;
}

void print_tensor(Ort::Value& tensor)
{
    Ort::TensorTypeAndShapeInfo ts = tensor.GetTensorTypeAndShapeInfo();

    const int32_t* el = tensor.GetTensorData<int32_t>();
    std::vector<int> ids;
    for (size_t i = 0; i < ts.GetElementCount(); i++)
    {
        //std::cout << el[i] << " ";
        ids.push_back((int)el[i]);
    }
    std::string translated;
    sp->Decode(ids, &translated);
    std::cout << translated << std::endl;
    //std::cout << std::endl;
    return;
}

void AsyncCallback(void* user_data, OrtValue** outputs, size_t num_outputs, OrtStatusPtr status_ptr)
{
    std::chrono::duration<double, std::milli> dur{1};
    // timeout in about 10 secs
    for (int i = 0; i < 1000 && atomic_printing.load(); ++i) {
      std::this_thread::sleep_for(dur);
    }
    
    atomic_printing.store(true);
#ifdef DEBUG
    std::cout << "in callback" << std::endl;
#endif
    Ort::Status status(status_ptr);
    Ort::Value output_value(outputs[0]);
    if (status.IsOK())
    {
        print_tensor(output_value);
    }
    else
    {
#ifdef DEBUG
        std::cout << "runasync failed" << std::endl;
#endif
    }
    atomic_printing.store(false);
}

int main(int argc, char* argv[])
{
    argparse::ArgumentParser program("OrtMT5");
    program.add_argument("--model_path").default_value(std::string("./mt5-ja_zh_beam_search.onnx"));
    program.add_argument("--spm_tokenizer_path").default_value(std::string("./vocabs_mc4.250000.100extra_sentencepiece.model"));
    program.add_argument("--spm_vocab_path").default_value(std::string("./vocabs_mc4.250000.100extra_sentencepiece.vocab"));
    program.add_argument("--max_length").default_value(128).scan<'i', int>();
    program.add_argument("--min_length").default_value(1).scan<'i', int>();
    program.add_argument("--num_beams").default_value(4).scan<'i', int>();
    program.add_argument("--num_return_sequences").default_value(1).scan<'i', int>();
    program.add_argument("--length_penalty").default_value((float)1.3).scan<'g', float>();
    program.add_argument("--repetition_penalty").default_value((float)1.3).scan<'g', float>();

    try
    {
        program.parse_args(argc, argv);
    }
    catch (const std::exception& e)
    {
        std::cout << "Caught " << e.what() << std::endl;
        std::cout << "Incorrect commandline args,"
            << "should be `OrtMT5.exe [model_path] [spm_tokenizer_path] [max_length] [min_length] "
            << "[num_beams] [num_return_sequences] [length_penalty] [repetition_penalty]`" << std::endl;
        return -1;
    }

    std::string arg_tokenizer_path = program.get<std::string>("--spm_tokenizer_path");
    std::string arg_vocab_path = program.get<std::string>("--spm_vocab_path");
    if (!std::filesystem::exists(arg_tokenizer_path))
    {
        std::cout << "Incorrect tokenizer path" << std::endl;
        return -1;
    }
    std::cout << "tokenizing" << std::endl;
    sp = std::make_unique<sentencepiece::SentencePieceProcessor>();
    const auto spm_status = sp->Load(arg_tokenizer_path);
    //const auto vocab_load_status = sp->LoadVocabulary(arg_vocab_path, 99999);
    //if ((!spm_status.ok()) | (!vocab_load_status.ok())) {
    //   std::cerr << spm_status.ToString() << std::endl;
    //}
    std::cout << sp->GetPieceSize() << std::endl;

    int32_t max_length_int;
    int32_t min_length_int;
    int32_t num_beams_int;
    int32_t num_return_sequences_int;
    float length_penalty_float;
    float repetition_penalty_float;
    max_length_int = program.get<int>("--max_length");
    min_length_int = program.get<int>("--min_length");
    num_beams_int = program.get<int>("--num_beams");
    num_return_sequences_int = program.get<int>("--num_return_sequences");
    length_penalty_float = program.get<float>("--length_penalty");
    repetition_penalty_float = program.get<float>("--repetition_penalty");
    
    Ort::SessionOptions session_options;
    max_threads = std::thread::hardware_concurrency();
#ifdef DEBUG
    std::cout << "max threads:" << max_threads << std::endl;
#endif
    session_options.SetIntraOpNumThreads(max_threads/2);
    session_options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
    session_options.SetInterOpNumThreads(max_threads/2);
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

    std::string arg_model_path = program.get<std::string>("--model_path");
    std::wstring model_path_wstring = str2wstr(arg_model_path);
    const wchar_t* model_path = model_path_wstring.c_str();

    if (!std::filesystem::exists(model_path))
    {
        std::cout << "Incorrect model path" << std::endl;
        return -1;
    }

    auto session_local = std::make_unique<Ort::Session>(*env, model_path, session_options);
    session = std::move(session_local);

    Ort::AllocatorWithDefaultOptions allocator;

    num_input_nodes = session->GetInputCount();
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

    Ort::ConstMemoryInfo memory_info = allocator.GetInfo();

    std::vector<int32_t> max_length_values{ max_length_int };
    std::vector<int64_t> max_length_dims{ 1 };

    std::vector<int32_t> min_length_values{ min_length_int };
    std::vector<int64_t> min_length_dims{ 1 };

    std::vector<int32_t> num_beams_values{ num_beams_int };
    std::vector<int64_t> num_beams_dims{ 1 };

    std::vector<int32_t> num_return_sequences_values{ num_return_sequences_int };
    std::vector<int64_t> num_return_sequences_dims{ 1 };

    std::vector<float> length_penalty_values{ (float)length_penalty_float };
    std::vector<int64_t> length_penalty_dims{ 1 };

    std::vector<float> repetition_penalty_values{ (float)repetition_penalty_float };
    std::vector<int64_t> repetition_penalty_dims = { 1 };

    Ort::AllocatedStringPtr output_name_ptr = session->GetOutputNameAllocated(0, allocator);
    const char* output_name = output_name_ptr.get();

    std::vector<const char*> output_names{output_name};
    output_names.resize(1);
    std::vector<int64_t> output_ids_dims{1, 1, max_length_int};

    std::array<std::vector<int32_t>, MAX_ASYNC_RUNS> input_ids_pool;
    std::array<std::vector<int64_t>, MAX_ASYNC_RUNS> input_ids_dims_pool;
    std::array<std::vector<Ort::Value>, MAX_ASYNC_RUNS> input_tensors_pool;
    std::array<std::vector<Ort::Value>, MAX_ASYNC_RUNS> output_tensors_pool;

    //int32_t input_length = -1;
    //int32_t input_value = -1;

    int scnt = 0;
    while (1)
    {
        //std::wstring input_wstr;
        //std::getline(std::wcin, input_wstr, L'\n');
        ////std::wcin >> input_wstr;
        //std::wcout << "user input:" << input_wstr << std::endl;
        //std::string input_str = wstr2str(input_wstr);
        std::string input_str_raw;
        std::getline(std::cin, input_str_raw, '\n');
        std::string input_str = to_utf8str(input_str_raw);
        std::cout << "user input:" << input_str_raw << std::endl;
        //if (input_length <= 0)
        //{
        //    std::cout << "invalid input, not a positive integer" << std::endl;
        //    return -1;
        //}
        //std::vector<int32_t> py_input_ids{};
        //for (int i = 0; i < input_length; i++)
        //{
        //    if (std::cin >> input_value)
        //    {
        //        py_input_ids.push_back(input_value);
        //    }
        //    else
        //    {
        //        std::cout << "invalid input, not a integer" << std::endl;
        //        return -1;
        //    }
        //}
        std::vector<std::string> input_pieces;
        sp->Encode(input_str, &input_pieces).IgnoreError();
        for (int ii = 0; ii < input_pieces.size(); ii++)
        {
            std::cout << input_pieces[ii] << "," << std::endl;
        }
        std::cout << std::endl;

        std::vector<int> py_input_ids;
        sp->Encode(input_str, &py_input_ids).IgnoreError();
        for (int ii = 0; ii < py_input_ids.size(); ii++)
        {
            std::cout << py_input_ids[ii] << "," << std::endl;
        }
        std::cout << std::endl;
        
        Ort::Value max_length = Ort::Value::CreateTensor<int32_t>(memory_info, max_length_values.data(), max_length_values.size(),
            max_length_dims.data(), max_length_dims.size());
        Ort::Value min_length = Ort::Value::CreateTensor<int32_t>(memory_info, min_length_values.data(), min_length_values.size(),
            min_length_dims.data(), min_length_dims.size());
        Ort::Value num_beams = Ort::Value::CreateTensor<int32_t>(memory_info, num_beams_values.data(), num_beams_values.size(),
            num_beams_dims.data(), num_beams_dims.size());
        Ort::Value num_return_sequences = Ort::Value::CreateTensor<int32_t>(memory_info, num_return_sequences_values.data(), num_return_sequences_values.size(),
            num_return_sequences_dims.data(), num_return_sequences_dims.size());
        Ort::Value length_penalty = Ort::Value::CreateTensor<float>(memory_info, length_penalty_values.data(), length_penalty_values.size(),
            length_penalty_dims.data(), length_penalty_dims.size());
        Ort::Value repetition_penalty = Ort::Value::CreateTensor<float>(memory_info, repetition_penalty_values.data(), repetition_penalty_values.size(),
            repetition_penalty_dims.data(), repetition_penalty_dims.size());

        input_ids_pool[scnt].clear();
        input_ids_pool[scnt] = std::move(py_input_ids);

        std::vector<int64_t> input_ids_dims{1, (int64_t)input_ids_pool[scnt].size()};
        input_ids_dims_pool[scnt].clear();
        input_ids_dims_pool[scnt] = std::move(input_ids_dims);

        Ort::Value input_ids = Ort::Value::CreateTensor<int32_t>(memory_info, input_ids_pool[scnt].data(), input_ids_pool[scnt].size(),
            input_ids_dims_pool[scnt].data(), input_ids_dims_pool[scnt].size());

        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(std::move(input_ids));
        input_tensors.push_back(std::move(max_length));
        input_tensors.push_back(std::move(min_length));
        input_tensors.push_back(std::move(num_beams));
        input_tensors.push_back(std::move(num_return_sequences));
        input_tensors.push_back(std::move(length_penalty));
        input_tensors.push_back(std::move(repetition_penalty));
        clear_ortvalue_vector(reinterpret_cast<OrtValue**>(input_tensors_pool[scnt].data()), input_tensors_pool[scnt].size());
        input_tensors_pool[scnt].clear();
        input_tensors_pool[scnt] = std::move(input_tensors);

        std::vector<Ort::Value> output_tensors;

        output_tensors.push_back(
                Ort::Value::CreateTensor<int32_t>(
                    allocator,
                    output_ids_dims.data(),
                    output_ids_dims.size()
                )
        );
        clear_ortvalue_vector(reinterpret_cast<OrtValue**>(output_tensors_pool[scnt].data()), output_tensors_pool[scnt].size());
        output_tensors_pool[scnt].clear();
        output_tensors_pool[scnt] = std::move(output_tensors);

        session->RunAsync(
            Ort::RunOptions{nullptr},
            input_node_names.data(),
            input_tensors_pool[scnt].data(),
            input_tensors_pool[scnt].size(),
            output_names.data(),
            output_tensors_pool[scnt].data(),
            output_names.size(),
            AsyncCallback,
            nullptr
            );
        scnt = (scnt + 1) % MAX_ASYNC_RUNS;
#ifdef DEBUG
        std::chrono::duration<double, std::milli> dur{5000};
        std::this_thread::sleep_for(dur);
#endif
    }
#ifdef DEBUG
    std::cout << "debug on" << std::endl;
    std::chrono::duration<double, std::milli> quit_delay{100000};
    std::this_thread::sleep_for(quit_delay);
#endif
    return 0;
}
