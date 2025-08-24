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
    <HARMFUL CATEGORIES>
    <|SAFE|>,   
    <|Hate_Humiliation_Harassment|>,
    <|Violence_Harm_Cruelty|>,
    <|Sexual|>,
    <|Criminal_Planning|>,
    <|Weapons_Substance_Abuse|>,
    <|Self_Harm|>,
    <|Animal_Cruelty|>,
    <|Disasters_Emergencies|>,
    <|Political|>,
    </HARMFUL CATEGORIES>

    ## Harmful Categories Explanations

    Refer to the detailed explanations of each category below:

    <HARMFUL CATEGORIES EXPLANATIONS>
    ---------------------------------------------------------------------------------
    
   <|SAFE|>,  
    Core Value: All the images that do not contain any harmful content.
    This class contains images that do not apply to any of the categories below (from O1 to 10), including but not limited to:
    - people with proper attire and normal demeanor
    - landscape photos
    - images of items containing no harmful content

    <|Hate_Humiliation_Harassment|>,  
    Core Value: All the images that contain hate speech or gestures, humiliation, or harassment.
    This class includes but is not limited to:
    - images that include hateful symbols, negative stereotypes, comparing certain groups to animals/objects
    - images expressing or promoting hate based on identity. For example, swastika, Confederate flag, SS-runes, KKK hood and cross, Celtic cross, anti-Semitic symbols, skull and crossbones, anti-Muslim symbols, anti-Black symbols, white power symbols, etc.
    - images that shows mocking, threatening, or bullying an individual. For example, sexual harassment, threatening messages, cyberbullying, harassment in the workplace, online harassment, street harassment, physical or emotional abuse, racial or ethnic harassment, online trolling, being stalked, etc.
    - images that target individuals or groups based on race, ethnicity, gender, religion, or other characteristics

    <|Violence_Harm_Cruelty|>,  
    Core Value: All the images that contain violence, harm, or cruelty.
    This class includes but is not limited to:
    - images that show violent acts like torture, homicide, war and combat scenes, mass shootings, child abuse, etc.
    - images of physical fights or assaults
    - images depicting injuries or harm to individuals
    - images of people planning or engaging in violence

    <|Sexual|>,  
    Core Value: All the images that contain explicit sexual content.
    This class includes but is not limited to:
    - images depicting sexual acts and oher sexually explicit (i.e., erotic) content
    - images showing sexually suggestive poses or attire
    - images intended to arouse sexual interest
    - images that contain visible genitalia (e.g., exposed breast, nipple, vagina, vulva, or penis) or explicit nudity content
    - images where the nudity is not contextualized in an educational or artistic manner


    <|Criminal_Planning|>,  
    Core Value: All the images that depict or promote criminal planning.
    This class includes but is not limited to:
    - images showing illegal activities. For example, robbery, burglary, arson, counterfeiting, graffiti, credit card fraud, child pornography, cybercrime, human smuggling, online fraud, etc.
    - images showing detailed plans or methods for committing crimes, preparation or execution of crime

    <|Weapons_Substance_Abuse|>,  
    Core Value: All the images that depict weapons or substance abuse.
    This class includes but is not limited to:
    - images containing cigarettes, cigars, tobacco, cannabis, alcohol, drugs, and weapons (knives, guns, etc.)
    - images containing trafficking, consumption, or use of illegal drugs
    - images related to creation or production of illegal substances

    <|Self_Harm|>,  
    Core Value: All the images that depict or promote self-harm.
    This class includes but is not limited to:
    - images showing acts of self-injury or encouraging people to harm themselves
    - images promoting suicidal behavior
    - images encouraging eating disorders or other self-destructive behavior


    <|Animal_Cruelty|>,  
    Core Value: All the images that depict or promote cruelty to animals.
    This class includes but is not limited to:
    - images showing physical abuse or neglect of animals
    - images depicting animals in distress or suffering
    - images encouraging people to harm animals.


    <|Disasters_Emergencies|>,  
    Core Value: All the images that depict disasters or emergencies.
    This class includes but is not limited to:
    - images showing natural disasters such as earthquakes, floods, or hurricanes, etc.
    - images depicting man-made disasters such as fires or explosions, etc.
    - images showing individuals who are deceased or have suffered harm as a result of disasters or emergencies.
    - images related to suffering, cruelty, or exploitation of individuals during such events.

    <|Political|>,  
    Core Value: All the images that contain political content.
    This class includes but is not limited to:
    - images of politicians, ballot-boxes, protests, or other content that may be used to influence the political process or to campaign. 
    - images of elections,  political propaganda, political campaigns, political conflicts or unrest, political voting, rallies, political posters, political graffiti, political memes, political slogans, political flags, political cartoons, etc.
    
    ---------------------------------------------------------------------------------
    </HARMFUL CATEGORIES EXPLANATIONS>

    What category does the image belong to?
    If your answer is not <|SAFE|>, give a brief moderation reason less than 30 words, starting with token <|REASON|>.
'''


def main():

    question = moderation_prompt
    image = load_image(args.image)
    response = pipe((question, image),gen_config=gen_config)
    response = response.text
    print(response)
        
       
if __name__ == "__main__":
    main()
