import ctypes
import time

dll = ctypes.CDLL("./bin/ortmtlib.dll")

model_path = ctypes.c_char_p(bytes("mt5-ja_zh_beam_search.onnx",'utf8'))
spm_path = ctypes.c_char_p(bytes("vocabs_mc4.250000.100extra_sentencepiece.model","utf8"))
max_length = ctypes.c_int(128)
min_length = ctypes.c_int(1)
num_beams = ctypes.c_int(1)
num_return_sequences = ctypes.c_int(1)
length_penalty = ctypes.c_float(1.3)
repetition_penalty = ctypes.c_float(1.3)

dll.create_ort_api.argtypes=None
dll.create_ort_api.restype=ctypes.c_void_p

dll.create_ort_session.argtypes = (
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_char_p)),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_char_p)),
        ctypes.POINTER(ctypes.c_size_t), 
        )
dll.create_ort_session.restype = ctypes.c_void_p

dll.create_sp_tokenizer.argtypes = (ctypes.c_void_p,)
dll.create_sp_tokenizer.restype = ctypes.c_void_p

dll.encode_as_ids.argtypes = (
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
        ctypes.POINTER(ctypes.c_size_t),
        )
dll.encode_as_ids.restype = None

dll.decode_from_ids.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_char_p),
        )
dll.decode_from_ids.restype = None

dll.run_session.argtypes=(
        ctypes.c_void_p,
        ctypes.c_void_p,

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
dll.run_session.restype=None


input_names = ctypes.POINTER(ctypes.c_char_p)()
num_input_nodes = ctypes.c_size_t()

output_names = ctypes.POINTER(ctypes.c_char_p)()
num_output_nodes = ctypes.c_size_t()

ort_api = ctypes.c_void_p(dll.create_ort_api())
session = ctypes.c_void_p(dll.create_ort_session(
    ort_api,
    model_path,
    ctypes.byref(input_names),
    ctypes.byref(num_input_nodes),
    ctypes.byref(output_names),
    ctypes.byref(num_output_nodes)
    ))
for i in range(num_input_nodes.value):
    print(input_names[i])

sp = ctypes.c_void_p(dll.create_sp_tokenizer(spm_path))

input_str = ctypes.c_char_p(bytes("<ja2zh>愛いしでる\0", "utf8"))
token_ids = ctypes.POINTER(ctypes.c_int)()
n_tokens = ctypes.c_size_t()
decoded_str = ctypes.c_char_p()
dll.encode_as_ids(
        sp,
        input_str,
        ctypes.byref(token_ids),
        ctypes.byref(n_tokens)
        )

output_ids = ctypes.POINTER(ctypes.c_int)()
output_len = ctypes.c_size_t()
dll.run_session(
        ort_api,
        session,

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
for i in range(output_len.value):
    print(output_ids[i], end=", ")
print()

dll.decode_from_ids(
        sp,
        output_ids,
        output_len,
        ctypes.byref(decoded_str)
        )

print(decoded_str.value.decode("utf8"))
#output_str = ctypes.c_char_p()
#out = dll.run_translate(session, sp, input_str, max_length,min_length,num_beams,num_return_sequences,length_penalty,repetition_penalty)
#print("translation finished:")
#outp = ctypes.c_char_p(out)
#print(output_str, outp)
#print(output_str.value, outp.value)
#print(outp.value.decode("utf8"))
