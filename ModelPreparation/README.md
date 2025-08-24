# Model Perparation Stage
## 1. Modify the model tokenizer
```
python change_tokenizer.py --model xx
```
## 2. Use a LLM-based Policy Parser to transform user-defined prompts into well-structured policy prompts, 
```
python policy_parser.py --rawfile xxx --outputfile xxx
```