import torch
import os
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
from lmdeploy import pipeline, GenerationConfig
from lmdeploy.vl import load_image
import json
import argparse
from typing import List, Optional, Sequence,  Union
from lmdeploy.tokenizer import Tokenizer,DetokenizeState
from lmdeploy.messages import (EngineGenerationConfig, GenerationConfig, Response)
from typing import  Dict, List,  Optional,Union
import json
import random
import asyncio
from lmdeploy.serve.utils import _get_event_loop
from lmdeploy.serve.async_engine import AsyncEngine

def new_detokenize_incrementally(self,
                                 all_input_ids: Sequence[int],
                                 state: DetokenizeState,
                                 skip_special_tokens: bool = True,
                                 spaces_between_special_tokens: bool = True):
        output = self.model.detokenize_incrementally(
                    all_input_ids,
                    state=state,
                    skip_special_tokens=skip_special_tokens,
                    spaces_between_special_tokens=spaces_between_special_tokens) 
        mapping = {
            '0': "<|SAFE|>",
            '1': "<|Hate_Humiliation_Harassment|>",
            '2': "<|Violence_Harm_Cruelty|>",
            '3': "<|Sexual|>",
            '4': "<|Criminal_Planning|>",
            '5': "<|Weapons_Substance_Abuse|>",
            '6': "<|Self_Harm|>",
            '7': "<|Animal_Cruelty|>",
            '8': "<|Disasters_Emergencies|>",
            '9': "<|Political|>"
        }      
        if output[0] in mapping:
            output = (mapping[output[0]], *output[1:]) 

        return output
    

def new_batch_infer(
            self,
            prompts: Union[List[str], str, List[Dict], List[List[Dict]]],
            gen_config: Optional[Union[GenerationConfig,
                                       List[GenerationConfig],
                                       EngineGenerationConfig,
                                       List[EngineGenerationConfig]]] = None,
            do_preprocess: bool = True,
            adapter_name: Optional[str] = None,
            use_tqdm: bool = False,
            **kwargs):
        """Inference a batch of prompts.

        Args:
            prompts (List[str] | str | List[Dict] | List[Dict]): a batch of
                prompts. It accepts: string prompt, a list of string prompts,
                a chat history in OpenAI format or a list of chat history.
            gen_config (GenerationConfig | None): a instance of or a list of
                GenerationConfig. Default to None.
            do_preprocess (bool): whether pre-process the messages. Default to
                True, which means chat_template will be applied.
            adapter_name (str): the adapter name of slora for pytorch backend.
                Pick one from adapters. Default to None, using the base model.
            use_tqdm (bool): Whether use the progress bar. Default to False
        """
        need_list_wrap = isinstance(prompts, str) or isinstance(
            prompts[0], Dict)
        prompts = [prompts] if need_list_wrap else prompts
        assert isinstance(prompts, List), 'prompts should be a list'
        if gen_config is None:
            gen_config = GenerationConfig()
        # set random if it is not set
        if not isinstance(gen_config, List) and gen_config.random_seed is None:
            gen_config.random_seed = random.getrandbits(64)
        if not isinstance(gen_config, List):
            gen_config = [gen_config] * len(prompts)
        assert len(prompts) == len(gen_config),\
                'input gen_confg length differs from the length of prompts' # noqa
        prompt_num = len(prompts)
        outputs = [Response('', 0, 0, i) for i in range(prompt_num)]
        generators = []
        if use_tqdm:
            import tqdm
            pbar = tqdm.tqdm(total=len(prompts))
        for i, prompt in enumerate(prompts):
            generators.append(
                self.generate(prompt,
                              i,
                              gen_config=gen_config[i],
                              stream_response=True,
                              sequence_start=True,
                              sequence_end=True,
                              do_preprocess=do_preprocess,
                              adapter_name=adapter_name,
                              **kwargs))

        async def _inner_call(i, generator):
            async for out in generator:
                outputs[i].text += out.response
                outputs[i].generate_token_len = out.generate_token_len
                outputs[i].input_token_len = out.input_token_len
                outputs[i].finish_reason = out.finish_reason
                if out.token_ids:
                    outputs[i].token_ids.extend(out.token_ids)
                if out.logprobs:
                    if outputs[i].logprobs is None:
                        outputs[i].logprobs = []
                    outputs[i].logprobs.extend(out.logprobs)
                if use_tqdm and out.finish_reason is not None:
                    pbar.update(1)

        async def gather():
            await asyncio.gather(
                *[_inner_call(i, generators[i]) for i in range(len(prompts))])

        _get_event_loop().run_until_complete(gather())
        outputs = outputs[0] if need_list_wrap else outputs
        outputs.text = outputs.text.strip()
        mapping = ["<|SAFE|>", "<|Hate_Humiliation_Harassment|>", "<|Violence_Harm_Cruelty|>", "<|Sexual|>", "<|Criminal_Planning|>", "<|Weapons_Substance_Abuse|>", "<|Self_Harm|>", "<|Animal_Cruelty|>", "<|Disasters_Emergencies|>","<|Political|>"] 
        if outputs.text in mapping:
            outputs.text = json.dumps({"MODERATION_RESULT": {outputs.text:True}})
        elif any(outputs.text.startswith(class_name) and "<|REASON|>" in outputs.text for class_name in mapping):
            class_name = next(class_name for class_name in mapping if outputs.text.startswith(class_name))
            reason = outputs.text.split("<|REASON|>", 1)[1].strip()
            outputs.text = json.dumps({"MODERATION_RESULT": {class_name:True}, "MODERATION_REASON": reason})
        return outputs
    

Tokenizer.detokenize_incrementally = new_detokenize_incrementally
AsyncEngine.batch_infer = new_batch_infer

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    default="./SafeVision-8B",
    help="path to load model.",
)
parser.add_argument(
    "--image",
    type=str,
    default="./test.jpg",
    help="path to load model.",
)
args = parser.parse_args()

pipe = pipeline(args.model)
gen_config = GenerationConfig(top_k=1, top_p=0.0001, temperature=0.0)

moderation_prompt = '''
What category does the image belong to?
'''


def main():

    question = moderation_prompt
    image = load_image(args.image)
    response = pipe((question, image),gen_config=gen_config)
    response = response.text
    print(response)
        
       
if __name__ == "__main__":
    main()
