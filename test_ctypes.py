import ctypes
import time
import gc
import os

gc.disable()
ortmtlib = ctypes.CDLL("./ortmtlib.dll")
splib = ctypes.CDLL("./splib.dll")

model_path = ctypes.c_char_p(bytes("mt5-ja_zh_beam_search.onnx",'utf8'))
spm_path = ctypes.c_char_p(bytes("vocabs_mc4.250000.100extra_sentencepiece.model","utf8"))
max_length = 128
min_length = 1
num_beams = 1
num_return_sequences = 1
length_penalty = 1.3
repetition_penalty = 1.3


ortmtlib.create_ort_session.argtypes = (
        ctypes.c_char_p,
        )
ortmtlib.create_ort_session.restype = ctypes.c_int

splib.create_sp_tokenizer.argtypes = (ctypes.c_char_p, )
splib.create_sp_tokenizer.restype = ctypes.c_int

splib.encode_as_ids.argtypes = (
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
        ctypes.POINTER(ctypes.c_size_t),
        )
splib.encode_as_ids.restype = ctypes.c_int

splib.decode_from_ids.argtypes = (
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_char_p),
        )
splib.decode_from_ids.restype = ctypes.c_int

ortmtlib.run_session.argtypes=(
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_char_p,

        ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
        ctypes.POINTER(ctypes.c_size_t)
        )
ortmtlib.run_session.restype=ctypes.c_int


input_names = ctypes.POINTER(ctypes.c_char_p)()
num_input_nodes = ctypes.c_size_t()

output_names = ctypes.POINTER(ctypes.c_char_p)()
num_output_nodes = ctypes.c_size_t()

session = ctypes.c_void_p()
env = ctypes.c_void_p()
res = ortmtlib.create_ort_session(
    model_path,
    )
print("create ort session?", res)

res = splib.create_sp_tokenizer(spm_path)
print("create sp?", res)

input_str = ctypes.c_char_p(bytes("<ja2zh>愛してる\0", "utf8"))
token_ids = ctypes.POINTER(ctypes.c_int32)()
n_tokens = ctypes.c_size_t()
#token_ids = (ctypes.c_int * 7)(1042, 462, 338, 12001, 669, 14942, 43556)
#n_tokens = ctypes.c_size_t(7)

decoded_str = ctypes.c_char_p()
res = splib.encode_as_ids(
        input_str,
        ctypes.byref(token_ids),
        ctypes.byref(n_tokens)
        )
print("encode as ids?", res)
print("input ids:")
input_ids_len = n_tokens.value
input_ids_py = [token_ids[i] for i in range(input_ids_len)]
print(input_ids_py)
#input_ids_ctypes = (ctypes.c_int32 * len(input_ids_py))(*tuple(input_ids_py))

input_tensors = [ctypes.c_void_p() for i in range(7)]
input_ids_ctypes = (ctypes.c_int32 * input_ids_len)(*input_ids_py)
input_ids_len_ctypes = ctypes.c_size_t(input_ids_len)
input_shape_ctypes = (ctypes.c_longlong * 2)(1, input_ids_len)
input_shape_len_ctypes = ctypes.c_size_t(2)
shape_one = (ctypes.c_longlong * 1)(1)
len_one = ctypes.c_size_t(1)
res = ortmtlib.create_tensor_int32(input_ids_ctypes, input_ids_len_ctypes,input_shape_ctypes, input_shape_len_ctypes, ctypes.byref(input_tensors[0]))
print("create input ids tensor?", res)
max_length_ctypes = (ctypes.c_int32 * 1)(max_length)
res = ortmtlib.create_tensor_int32(max_length_ctypes, len_one, shape_one, len_one, ctypes.byref(input_tensors[1]))
print("create max length tensor?", res)
min_length_ctypes = (ctypes.c_int32 * 1)(min_length)
res = ortmtlib.create_tensor_int32(min_length_ctypes, len_one, shape_one, len_one, ctypes.byref(input_tensors[2]))
print("create min length tensor?", res)
num_beams_ctypes = (ctypes.c_int32 * 1)(num_beams)
res = ortmtlib.create_tensor_int32(num_beams_ctypes, len_one, shape_one, len_one, ctypes.byref(input_tensors[3]))
print("create num beams tensor?", res)
num_return_sequences_ctypes = (ctypes.c_int32 * 1)(num_return_sequences)
res = ortmtlib.create_tensor_int32(num_return_sequences_ctypes, len_one, shape_one, len_one, ctypes.byref(input_tensors[4]))
print("create num return sequences?", res)
length_penalty_ctypes = (ctypes.c_float * 1)(length_penalty)
res = ortmtlib.create_tensor_float(length_penalty_ctypes, len_one, shape_one, len_one, ctypes.byref(input_tensors[5]))
print("create length penalty tensor?", res)
repetition_penalty_ctypes = (ctypes.c_float * 1)(repetition_penalty)
res = ortmtlib.create_tensor_float(repetition_penalty_ctypes, len_one, shape_one, len_one, ctypes.byref(input_tensors[6]))
print("create repetition penalty tensor?", res)

#ortmtlib.print_tensor_int32(input_tensors[0])
input_tensors_ctypes = (ctypes.c_void_p * len(input_tensors))(*input_tensors)
output_ids = ctypes.POINTER(ctypes.c_int)()
output_len = ctypes.c_size_t()

output_name = ctypes.c_char_p(bytes("sequences", "utf8"))
res = ortmtlib.run_session(
        input_tensors_ctypes,
        output_name,

        ctypes.byref(output_ids),
        ctypes.byref(output_len)
        )
print("run session?", res)

print("output ids:")
for i in range(output_len.value):
    print(output_ids[i], end=", ")
print()

res = splib.decode_from_ids(
        output_ids,
        output_len,
        ctypes.byref(decoded_str)
        )
print("decode from ids?", res)

print(decoded_str.value.decode("utf8"))
print("translation finished")
res = ortmtlib.release()
print("release?", res)
