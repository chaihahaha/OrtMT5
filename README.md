# OrtMT5
onnxruntime mT5 translator

### Usage example:

```bash
ortmt5 --model_path="path/to/mt5-ja_zh_beam_search.onnx" --spm_tokenizer_path="path/to/vocabs_mc4.250000.100extra_sentencepiece.model" --max_length=128 --min_length=1 --num_beams=8 --num_return_sequences=1 --length_penalty=1.1 --repetition_penalty=1.1
```

### Build:

```bash
"path/to/VsDevCmd.bat" -arch=amd64
mkdir build
cd build

# if win64
cmake .. -DONNXRUNTIME_ROOTDIR='path/to/onnxruntime-win-x64' -DSENTENCEPIECE_ROOTDIR='path/to/sentencepiece' -DARGPARSE_ROOTDIR='path/to/argparse' -DICU_ROOTDIR='path/to/icu'

# if win32
cmake .. -A win32 -DONNXRUNTIME_ROOTDIR='path/to/onnxruntime-win-x64' -DSENTENCEPIECE_ROOTDIR='path/to/sentencepiece' -DARGPARSE_ROOTDIR='path/to/argparse' -DICU_ROOTDIR='path/to/icu-win32'

# if use DirectML
cmake .. -DONNXRUNTIME_ROOTDIR='path/to/onnxruntime-win-dml' -DSENTENCEPIECE_ROOTDIR='path/to/sentencepiece' -DARGPARSE_ROOTDIR='path/to/argparse' -DICU_ROOTDIR='path/to/icu' -DPROVIDER="dml"

cmake --build . --config Release
xcopy Release/* path/to/install
```

### SentencePiece Tokenizer Download Link:

```bash
https://console.cloud.google.com/storage/browser/t5-data/vocabs/mc4.250000.100extra
```

### Convert MT5 Model:

First, download model to folder `mt5-translation-ja_zh` with the following Python code:

```
from huggingface_hub import snapshot_download
snapshot_download(repo_id="larryvrh/mt5-translation-ja_zh")
```

Second, install onnxruntime:

```
pip install onnxruntime
```

Finally, run the commands to convert model to int8 quantized onnx:

```
# In command line
python -m onnxruntime.transformers.convert_generation -m ./mt5-translation-ja_zh --model_type mt5  --output mt5-translation-ja_zh.onnx --disable_perf_test --disable_parity -e

python -m onnxruntime.quantization.preprocess --input mt5-translation-ja_zh_encoder_decoder_init.onnx --output mt5-translation-ja_zh_encoder_decoder_init_infer.onnx --skip_optimization 1  --save_as_external_data

python -m onnxruntime.quantization.preprocess --input mt5-translation-ja_zh_decoder.onnx --output mt5-translation-ja_zh_decoder_infer.onnx --skip_optimization 1  --save_as_external_data

# In Python shell:
from onnxruntime.quantization import QuantType, quantize_dynamic
quantize_dynamic(model_input="mt5-translation-ja_zh_encoder_decoder_init_infer.onnx", model_output="mt5-translation-ja_zh_encoder_decoder_init-int8.onnx", weight_type=QuantType.QUInt8,use_external_data_format=True)
quantize_dynamic(model_input="mt5-translation-ja_zh_decoder_infer.onnx", model_output="mt5-translation-ja_zh_decoder-int8.onnx", weight_type=QuantType.QUInt8,use_external_data_format=True)

# In command line
python -m onnxruntime.transformers.convert_generation -m path/to/mt5-translation-ja_zh --model_type mt5 --decoder_onnx mt5-translation-ja_zh_decoder-int8.onnx --encoder_decoder_init_onnx mt5-translation-ja_zh_encoder_decoder_init-int8.onnx --output mt5-ja_zh_beam_search.onnx  --disable_perf_test --disable_parity -e
```

