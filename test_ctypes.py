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

#DLL_create_translator_session = dll.create_translator_session
dll.create_ort_session.argtypes=(ctypes.c_char_p,)
dll.create_ort_session.restype=ctypes.c_void_p

dll.create_sp_tokenizer.argtypes=(ctypes.c_void_p,)
dll.create_sp_tokenizer.restype=ctypes.c_void_p

dll.run_translate.argtypes=(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float)
dll.run_translate.restype=ctypes.c_char_p

session = ctypes.c_void_p(dll.create_ort_session(model_path))
sp = ctypes.c_void_p(dll.create_sp_tokenizer(spm_path))

input_str = ctypes.c_char_p(bytes("<ja2zh>愛いしでる\0", "utf8"))
output_str = ctypes.c_char_p()
out = dll.run_translate(session, sp, input_str, max_length,min_length,num_beams,num_return_sequences,length_penalty,repetition_penalty)
print("translation finished:")
outp = ctypes.c_char_p(out)
print(output_str, outp)
print(output_str.value, outp.value)
print(outp.value.decode("utf8"))
