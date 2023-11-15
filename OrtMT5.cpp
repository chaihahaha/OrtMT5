#include "onnxruntime_cxx_api.h"
#include <iostream>
#include <vector>

#pragma comment(lib, "onnxruntime.lib")
template <typename T>
T vectorProduct(const std::vector<T>& v)
{
	return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

int main()
{
    Ort::SessionOptions session_options;
	session_options.SetIntraOpNumThreads(6);
	session_options.SetGraphOptimizationLevel(
		GraphOptimizationLevel::ORT_ENABLE_ALL);
	Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, "OrtMT5");
	std::wstring model_path = L"D:\\programs\\LunaTranslator\\mt5-translation-ja_zh-onnx\\mt5-ja_zh_beam_search.onnx";
	
	auto session = Ort::Session(env, model_path.c_str(), session_options);

	//*************************************************************************
    // print model input layer (node names, types, shape etc.)
	Ort::AllocatorWithDefaultOptions allocator;

	// print number of model input nodes
	const size_t num_input_nodes = session.GetInputCount();
	std::vector<Ort::AllocatedStringPtr> input_names_ptr;
	std::vector<const char*> input_node_names;
	//input_names_ptr.reserve(num_input_nodes);
	input_node_names.reserve(num_input_nodes);
	std::vector<int64_t> input_node_dims;

	//std::cout << "Number of inputs = " << num_input_nodes << std::endl;

	// iterate over all input nodes
	for (size_t i = 0; i < num_input_nodes; i++) {
		// print input node names
		auto input_name = session.GetInputNameAllocated(i, allocator);
		//std::cout << "Input " << i << " : name =" << input_name.get() << std::endl;
		input_node_names.push_back(input_name.get());
		input_names_ptr.push_back(std::move(input_name));

		// print input node types
		//auto type_info = session.GetInputTypeInfo(i);
		//auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

		//ONNXTensorElementDataType type = tensor_info.GetElementType();
		//std::cout << "Input " << i << " : type = " << type << std::endl;

		// print input shapes/dims
		/*input_node_dims = tensor_info.GetShape();
		std::cout << "Input " << i << " : num_dims = " << input_node_dims.size() << '\n';
		for (size_t j = 0; j < input_node_dims.size(); j++) {
			std::cout << "Input " << i << " : dim[" << j << "] =" << input_node_dims[j] << '\n';
		}
		std::cout << std::flush;*/
	}

	Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
		OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
	size_t input_count = session.GetInputCount();
	int32_t input_length = -1;
	int32_t input_value = -1;

	while (std::cin >> input_length)
	{
		//std::cout << "reading input" << std::endl;
		std::vector<int32_t> py_input_ids{};
		for (int i=0; i < input_length; i++)
		{
			std::cin >> input_value;
			py_input_ids.push_back(input_value);
		}
		//for (int i = 0; i < py_input_ids.size(); i++)
		//{
		//	std::cout << py_input_ids[i] << " ";
		//}
		//std::cout << std::endl;
		int32_t max_length_int = 128;
		std::vector<int32_t> max_length_values{ max_length_int };
		std::vector<int64_t> max_length_dims{ 1 };
		Ort::Value max_length = Ort::Value::CreateTensor<int32_t>(memory_info, max_length_values.data(), max_length_values.size(),
			max_length_dims.data(), max_length_dims.size());

		std::vector<int32_t> min_length_values{ 1 };
		std::vector<int64_t> min_length_dims{ 1 };
		Ort::Value min_length = Ort::Value::CreateTensor<int32_t>(memory_info, min_length_values.data(), min_length_values.size(),
			min_length_dims.data(), min_length_dims.size());

		std::vector<int32_t> num_beams_values{ 8 };
		std::vector<int64_t> num_beams_dims{ 1 };
		Ort::Value num_beams = Ort::Value::CreateTensor<int32_t>(memory_info, num_beams_values.data(), num_beams_values.size(),
			num_beams_dims.data(), num_beams_dims.size());

		std::vector<int32_t> num_return_sequences_values{ 1 };
		std::vector<int64_t> num_return_sequences_dims{ 1 };
		Ort::Value num_return_sequences = Ort::Value::CreateTensor<int32_t>(memory_info, num_return_sequences_values.data(), num_return_sequences_values.size(),
			num_return_sequences_dims.data(), num_return_sequences_dims.size());

		std::vector<float> length_penalty_values{ (float)1.1 };
		std::vector<int64_t> length_penalty_dims{ 1 };
		Ort::Value length_penalty = Ort::Value::CreateTensor<float>(memory_info, length_penalty_values.data(), length_penalty_values.size(),
			length_penalty_dims.data(), length_penalty_dims.size());

		std::vector<float> repetition_penalty_values{ (float)1.1 };
		std::vector<int64_t> repetition_penalty_dims = { 1 };
		Ort::Value repetition_penalty = Ort::Value::CreateTensor<float>(memory_info, repetition_penalty_values.data(), repetition_penalty_values.size(),
			repetition_penalty_dims.data(), repetition_penalty_dims.size());

		std::vector<int32_t> input_ids_values = py_input_ids;  //{1042, 264, 462, 338, 12001, 25134, 31150, 70694, 10978, 67916, 81634, 574, 86977, 56157, 66637, 309, 1};
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
		size_t output_count = 1;

		std::vector<Ort::Value> output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), input_tensors.data(), input_tensors.size(), output_names.data(), output_names.size());
		Ort::Value output_tensor = std::move(output_tensors[0]);
		//std::cout << "translation finished" << std::endl;

		Ort::TensorTypeAndShapeInfo ts = output_tensor.GetTensorTypeAndShapeInfo();
		/*std::vector<int64_t> shape = ts.GetShape();
		std::cout << "output tensor shape: ";
		for (int i = 0; i < shape.size(); i++)
		{
			std::cout << shape[i] << ", ";
		}
		std::cout << std::endl;*/

		//std::cout << "translated tokens: ";
		int32_t* output_tokens = output_tensor.GetTensorMutableData<int32_t>();
		for (int i = 0; i < ts.GetElementCount(); i++)
		{
			std::cout << output_tokens[i] << " ";
		}
		std::cout << std::endl;
	}
	
	
	return 0;
}
