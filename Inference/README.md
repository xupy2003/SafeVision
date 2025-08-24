# Inference Code
## 1. Inference with SafeVision in the comprehension mode
```
python inference_comprehension_mode.py --model xx --image xx
```
## 2. Inference with SafeVision in the fast mode
```
python inference_fast_mode.py --model xxx --image xx
```

## 3. Evaluate SafeVision on different benchmarks
```
python evaulation.py --groundtruth xxx  --model xxx --output xxx --analyse xxx --detail xxx
```

- --groundtruth: benchmark ground truth path
- --model: model path
- --output: output file path
- --analyse: result analyse file path
- --detail: fail case detail file path

## 4. Evaluate SafeVision on different benchmarks with in-context learning
```
python evaulation_with_icl.py --groundtruth xxx  --model xxx --output xxx --analyse xxx --detail xxx
```