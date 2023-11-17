import subprocess
import time

ort_mt5_path = "ortmt5.exe"
model_path = r"mt5-ja_zh_beam_search.onnx"
max_length_int = 128
min_length_int = 1
num_beams_int = 8
num_return_sequences_int = 1
length_penalty_float = 1.1
repetition_penalty_float = 1.1

mt5_proc = subprocess.Popen([ort_mt5_path, model_path, str(max_length_int), str(min_length_int), str(num_beams_int), str(num_return_sequences_int), str(length_penalty_float), str(repetition_penalty_float)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True)

in_str = '17 1042 264 462 338 12001 25134 31150 70694 10978 67916 81634 574 86977 56157 66637 309 1\n'
tik = time.time()
mt5_proc.stdin.write(in_str)
od1 = mt5_proc.stdout.readline()
print(od1)
mt5_proc.stdin.write(in_str)
od2 = mt5_proc.stdout.readline()
print(od2)
print("time elapsed:", time.time() - tik)
mt5_proc.communicate("exit")
