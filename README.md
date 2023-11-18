# OrtMT5
onnxruntime mT5 translator

### Usage example:

```bash
ortmt5 "path/to/mt5-ja_zh_beam_search.onnx" 128 1 8 1 1.1 1.1 < test_input_token_ids.txt
python test_sp.py
```

### Build:

```bash
"path/to/VsDevCmd.bat" -arch=amd64
mkdir build
cd build
cmake .. -DONNXRUNTIME_ROOTDIR='path/to/onnxruntime'
cmake --build .
```

### Convert MT5 Model:

First, download model to folder `mt5-translation-ja_zh` with the following Python code:

```
from huggingface_hub import snapshot_download
snapshot_download(repo_id="larryvrh/mt5-translation-ja_zh")
```

Second, clone onnxruntime repo:

```
git clone https://github.com/microsoft/onnxruntime
```

Finally, run the commands to convert model to onnx:

```
python -m onnxruntime.transformers.convert_generation -m ./mt5-translation-ja_zh --model_type mt5  --output mt5-translation-ja_zh.onnx --disable_perf_test --disable_parity -e

python -m onnxruntime.transformers.convert_generation -m ./mt5-translation-ja_zh --model_type mt5 --decoder_onnx mt5-translation-ja_zh_decoder.onnx --encoder_decoder_init_onnx mt5-translation-ja_zh_encoder_decoder_init.onnx --output mt5-ja_zh_beam_search.onnx  --disable_perf_test --disable_parity -e
```

