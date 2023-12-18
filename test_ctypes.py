import ctypes
import time
import gc
import os

gc.disable()
ortmtlib = ctypes.CDLL("./ortmtlib.dll")
splib = ctypes.CDLL("./splib.dll")

model_path = ctypes.c_char_p(bytes("test_opt.onnx",'utf8'))
spm_path = ctypes.c_char_p(bytes("vocabs_mc4.250000.100extra_sentencepiece.model","utf8"))
max_length = ctypes.c_int(128)
min_length = ctypes.c_int(1)
num_beams = ctypes.c_int(1)
num_return_sequences = ctypes.c_int(1)
length_penalty = ctypes.c_float(1.3)
repetition_penalty = ctypes.c_float(1.3)

ortmtlib.create_ort_api.argtypes=None
ortmtlib.create_ort_api.restype=ctypes.c_int

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
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,

        ctypes.POINTER(ctypes.c_int),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
        ctypes.POINTER(ctypes.c_size_t)
        )
ortmtlib.run_session.restype=ctypes.c_int


input_names = ctypes.POINTER(ctypes.c_char_p)()
num_input_nodes = ctypes.c_size_t()

output_names = ctypes.POINTER(ctypes.c_char_p)()
num_output_nodes = ctypes.c_size_t()

ort_api = ctypes.c_void_p()
res = ortmtlib.create_ort_api()
print("create ort api?", res)

session = ctypes.c_void_p()
env = ctypes.c_void_p()
res = ortmtlib.create_ort_session(
    model_path,
    )
print("create ort session?", res)
#for i in range(num_input_nodes.value):
#    print(input_names[i])

res = splib.create_sp_tokenizer(spm_path)
print("create sp?", res)

input_str = ctypes.c_char_p(bytes("<ja2zh>愛してる\0", "utf8"))
token_ids = ctypes.POINTER(ctypes.c_int)()
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
for i in range(n_tokens.value):
    print(token_ids[i], end=", ")
print()

output_ids = ctypes.POINTER(ctypes.c_int)()
output_len = ctypes.c_size_t()

res = ortmtlib.run_session(
        max_length,
        min_length,
        num_beams,
        num_return_sequences,
        length_penalty,
        repetition_penalty,

        token_ids,
        n_tokens,
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
