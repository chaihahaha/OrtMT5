import subprocess
import time
import tokenizers
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='ortmt5 test script')
    parser.add_argument('-m', '--onnx_model_path', default='mt5-ja_zh_beam_search.onnx', type=str, help='the path to mT5 onnx model')
    parser.add_argument('-t', '--tokenizer_path', default='tokenizer.json', type=str, help='the path to mT5 tokenizer model')
    parser.add_argument('-e', '--ortmt5_executable_path', default='ortmt5.exe', type=str, help='the path to ortmt5.exe')
    parser.add_argument('-max', '--max_length', default=128, type=int, help='max generation length')
    parser.add_argument('-min', '--min_length', default=1, type=int, help='min generation length')
    parser.add_argument('-nb', '--n_beams', default=8, type=int, help='number of beams to search')
    parser.add_argument('-ns', '--n_sequences', default=1, type=int, help='number of output sequences')
    parser.add_argument('-lp', '--length_penalty', default=1.3, type=float, help='penalty of generating long text')
    parser.add_argument('-rp', '--repetition_penalty', default=1.3, type=float, help='penalty of generating repeating text')
    args = parser.parse_args()
    return args

def translate_user_input(mt5_proc):
    in_str = input("Setence to be translated: ")
    pipe_in_str = "<-ja2zh-> " + in_str
    print("mt5 subprocess stdin:", pipe_in_str)

    tik = time.time()
    mt5_proc.stdin.write(pipe_in_str, encoding='utf8')
    pipe_out_str = mt5_proc.stdout.readline(encoding='utf8')
    print("time elapsed:", time.time() - tik)
    print(pipe_out_str)
    while pipe_out_str:
        pipe_out_str = mt5_proc.stdout.readline(encoding='utf8')
        print(pipe_out_str)
    return

if __name__=='__main__':
    args = parse_args()

    ort_mt5_path = args.ortmt5_executable_path
    model_path = args.onnx_model_path
    tok_path = args.tokenizer_path
    max_length_int = args.max_length
    min_length_int = args.min_length
    num_beams_int = args.n_beams
    num_return_sequences_int = args.n_sequences
    length_penalty_float = args.length_penalty
    repetition_penalty_float = args.repetition_penalty
    
    mt5_args = [ort_mt5_path]
    print("mt5 subprocess args:", mt5_args)
    mt5_proc = subprocess.Popen(mt5_args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True)
    
    #tok = tokenizers.Tokenizer.from_file(tok_path)

    while True:
        #translate_user_input(mt5_proc, tok)
        translate_user_input(mt5_proc)
