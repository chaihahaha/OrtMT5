#include "onnxruntime_cxx_api.h"
#include "sentencepiece_processor.h"
#include "argparse/argparse.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <codecvt>
#include <filesystem>
#include <thread>
#include <atomic>
#include <unicode/unistr.h>
#include <unicode/ustream.h>

void AsyncCallback(void* user_data, OrtValue** outputs, size_t num_outputs, OrtStatusPtr status_ptr);
void print_tensor(Ort::Value& tensor);
void clear_ortvalue_vector(OrtValue** a, size_t len);
