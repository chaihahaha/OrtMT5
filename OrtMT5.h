#include "onnxruntime_cxx_api.h"
#include "sentencepiece_processor.h"
#include "argparse/argparse.hpp"
#include "utils.h"
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <thread>
#include <atomic>
#include <unicode/unistr.h>
#include <unicode/ustring.h>
#include <unicode/ustream.h>

void async_callback(void* user_data, OrtValue** outputs, size_t num_outputs, OrtStatusPtr status_ptr);
void print_translated(Ort::Value& tensor);
void clear_ortvalue_vector(OrtValue** a, size_t len);

