# generate fine-tune dataset for comprehensive harmful dataset
import torch
import os
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor
from transformers import AutoTokenizer
import json
import re
import argparse
import pandas as pd
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
from lmdeploy import pipeline
from lmdeploy.vl import load_image
from tqdm import tqdm
import json
parser = argparse.ArgumentParser()
parser.add_argument(
    "--groundtruth",
    type=str,
    default="./VisionHarm-500K/train_val_set/train_ground_truth_v2.csv",
    help="path to ground truth file.",
)
parser.add_argument(
    "--output",
    type=str,
    default="./VisionHarm-500K/train_val_set/train_gt.jsonl",
    help="path to dataset jsonl file.",
)
parser.add_argument(
    "--model",
    type=str,
    default="OpenGVLab/InternVL2_5-8B",
    help="path to load model.",
)

args = parser.parse_args()

pipe = pipeline(args.model)

def reformat_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            data = json.loads(line)
            image_path = data["image"]
            conversations = data["conversations"]

            for i in range(0, len(conversations), 2):
                if conversations[i]["from"] == "human" and conversations[i+1]["from"] == "gpt":
                    query = conversations[i]["value"].replace("<image>\n\n", "")
                    response = conversations[i+1]["value"]
                    new_entry = {
                        "query": query,
                        "response": response,
                        "images": [image_path]
                    }
                    outfile.write(json.dumps(new_entry, ensure_ascii=False) + '\n')
                    
def process_moderation_prompt(gt_label):
    harmful_categories = {
        "NA":"<|Safe|>",
        "O1": "<|Hate_Humiliation_Harassment|>",
        "O2": "<|Violence_Harm_Cruelty|>",
        "O3": "<|Sexual|>",
        "O4": "<|Criminal_Planning|>",
        "O5": "<|Weapons_Substance_Abuse|>",
        "O6": "<|Self_Harm|>",
        "O7": "<|Animal_Cruelty|>",
        "O8": "<|Disasters_Emergencies|>",
        "O9": "<|Political|>"
    }

    harmful_explanations = {
        "NA":'''
    <|Safe|>,  
    Core Value: All the images that do not contain any harmful content.
    This class contains images that do not apply to any of the categories below (from O1 to 10), including but not limited to:
    - people with proper attire and normal demeanor
    - landscape photos
    - images of items containing no harmful content
        ''',
        "O1": '''
    <|Hate_Humiliation_Harassment|>,  
    Core Value: All the images that contain hate speech or gestures, humiliation, or harassment.
    This class includes but is not limited to:
    - images that include hateful symbols, negative stereotypes, comparing certain groups to animals/objects
    - images expressing or promoting hate based on identity. For example, swastika, Confederate flag, SS-runes, KKK hood and cross, Celtic cross, anti-Semitic symbols, skull and crossbones, anti-Muslim symbols, anti-Black symbols, white power symbols, etc.
    - images that shows mocking, threatening, or bullying an individual. For example, sexual harassment, threatening messages, cyberbullying, harassment in the workplace, online harassment, street harassment, physical or emotional abuse, racial or ethnic harassment, online trolling, being stalked, etc.
    - images that target individuals or groups based on race, ethnicity, gender, religion, or other characteristics
''',
        "O2": '''
    <|Violence_Harm_Cruelty|>,  
    Core Value: All the images that contain violence, harm, or cruelty.
    This class includes but is not limited to:
    - images that show violent acts like torture, homicide, war and combat scenes, mass shootings, child abuse, etc.
    - images of physical fights or assaults
    - images depicting injuries or harm to individuals
    - images of people planning or engaging in violence
''',
        "O3": '''
    <|Sexual|>,  
    Core Value: All the images that contain explicit sexual content.
    This class includes but is not limited to:
    - images depicting sexual acts and oher sexually explicit (i.e., erotic) content
    - images showing sexually suggestive poses or attire
    - images intended to arouse sexual interest
    - images that contain visible genitalia (e.g., exposed breast, nipple, vagina, vulva, or penis) or explicit nudity content
    - images where the nudity is not contextualized in an educational or artistic manner
''',
        "O4": '''
    <|Criminal_Planning|>,  
    Core Value: All the images that depict or promote criminal planning.
    This class includes but is not limited to:
    - images showing illegal activities. For example, robbery, burglary, arson, counterfeiting, graffiti, credit card fraud, child pornography, cybercrime, human smuggling, online fraud, etc.
    - images showing detailed plans or methods for committing crimes, preparation or execution of crime
''',
        "O5": '''
    <|Weapons_Substance_Abuse|>,  
    Core Value: All the images that depict weapons or substance abuse.
    This class includes but is not limited to:
    - images containing cigarettes, cigars, tobacco, cannabis, alcohol, drugs, and weapons (knives, guns, etc.)
    - images containing trafficking, consumption, or use of illegal drugs
    - images related to creation or production of illegal substances
''',
        "O6": '''
    <|Self_Harm|>,  
    Core Value: All the images that depict or promote self-harm.
    This class includes but is not limited to:
    - images showing acts of self-injury or encouraging people to harm themselves
    - images promoting suicidal behavior
    - images encouraging eating disorders or other self-destructive behavior
''',
        "O7": '''
    <|Animal_Cruelty|>,  
    Core Value: All the images that depict or promote cruelty to animals.
    This class includes but is not limited to:
    - images showing physical abuse or neglect of animals
    - images depicting animals in distress or suffering
    - images encouraging people to harm animals.
''',
        "O8": '''
    <|Disasters_Emergencies|>,  
    Core Value: All the images that depict disasters or emergencies.
    This class includes but is not limited to:
    - images showing natural disasters such as earthquakes, floods, or hurricanes, etc.
    - images depicting man-made disasters such as fires or explosions, etc.
    - images showing individuals who are deceased or have suffered harm as a result of disasters or emergencies.
    - images related to suffering, cruelty, or exploitation of individuals during such events.
''',
    "O9": '''
    <|Political|>,  
    Core Value: All the images that contain political content.
    This class includes but is not limited to:
    - images of politicians, ballot-boxes, protests, or other content that may be used to influence the political process or to campaign. 
    - images of elections,  political propaganda, political campaigns, political conflicts or unrest, political voting, rallies, political posters, political graffiti, political memes, political slogans, political flags, political cartoons, etc.
'''
    }

    if gt_label in harmful_categories:
        del harmful_categories[gt_label]
        del harmful_explanations[gt_label]

    new_harmful_categories = "\n".join([f"{key}: {value}" for key, value in harmful_categories.items()])
    new_harmful_explanations = "\n".join(harmful_explanations.values())

    new_prompt = f'''
    # Image Moderation Assistant Instructions
    As a image moderation assistant, your task is to classify a potentially harmful image into a category.

    <HARMFUL CATEGORIES>
    {new_harmful_categories}
    </HARMFUL CATEGORIES>

    ## Harmful Categories Explanations

    Refer to the detailed explanations of each category below:

    <HARMFUL CATEGORIES EXPLANATIONS>
    ---------------------------------------------------------------------------------

    {new_harmful_explanations}
    ---------------------------------------------------------------------------------
    </HARMFUL CATEGORIES EXPLANATIONS>
    
    Now please classify the image into a category.
    '''

    return new_prompt

def generate_response(gt_label, gt_image):
    results = []
    moderation_result = [False, False, False, False, False,False, False, False, False, False,False]
    instruction = ""
    reason_instruction = ""
    new_label = ""
    # generate specific image description
    if gt_label == "[0]":
        moderation_result[0] = True
        gt_label = "<|Safe|>"
        new_label = "NA(Safe)"
        instruction = '''
        The image is classified as Safe. It means that the image does not contain any potentially harmful content.
        You just need to give more specific description of the image based on the image caption and image content. 
        '''
        reason_instruction = '''
   <|SAFE|>,  
    Core Value: All the images that do not contain any harmful content.
    This class contains images that do not apply to any of the categories below (from O1 to 10), including but not limited to:
    - people with proper attire and normal demeanor
    - landscape photos
    - images of items containing no harmful content
        '''
    elif gt_label == "[1]":
        moderation_result[1] = True
        gt_label = "<|Hate_Humiliation_Harassment|>"
        new_label = "O1(Hate_Humiliation_Harassment)"
        instruction = '''
        The image is classified as Hate, Humiliation, Harassment. It means that the image contains hate speech or gestures, humiliation, or harassment.
        In your answer, you need to emphasize on content related to hate speech or gestures, humiliation, or harassment, including but not limit to:
    - images that include hateful symbols, negative stereotypes, comparing certain groups to animals/objects
    - images expressing or promoting hate based on identity. For example, swastika, Confederate flag, SS-runes, KKK hood and cross, Celtic cross, anti-Semitic symbols, skull and crossbones, anti-Muslim symbols, anti-Black symbols, white power symbols, etc.
    - images that shows mocking, threatening, or bullying an individual. For example, sexual harassment, threatening messages, cyberbullying, harassment in the workplace, online harassment, street harassment, physical or emotional abuse, racial or ethnic harassment, online trolling, being stalked, etc.
    - images that target individuals or groups based on race, ethnicity, gender, religion, or other characteristics
        '''
        reason_instruction = '''
    <|Hate_Humiliation_Harassment|>,  
    Core Value: All the images that contain hate speech or gestures, humiliation, or harassment.
    This class includes but is not limited to:
    - images that include hateful symbols, negative stereotypes, comparing certain groups to animals/objects
    - images expressing or promoting hate based on identity. For example, swastika, Confederate flag, SS-runes, KKK hood and cross, Celtic cross, anti-Semitic symbols, skull and crossbones, anti-Muslim symbols, anti-Black symbols, white power symbols, etc.
    - images that shows mocking, threatening, or bullying an individual. For example, sexual harassment, threatening messages, cyberbullying, harassment in the workplace, online harassment, street harassment, physical or emotional abuse, racial or ethnic harassment, online trolling, being stalked, etc.
    - images that target individuals or groups based on race, ethnicity, gender, religion, or other characteristics
        '''
    elif gt_label == "[2]":
        moderation_result[2] = True
        gt_label = "<|Violence_Harm_Cruelty|>"
        new_label = "O2(Violence_Harm_Cruelty)"
        instruction = '''
        The image is classified as Violence, Harm, or Cruelty. It means that the image contains violence, harm, or cruelty.
        In your answer, you need to emphasize on content related to violence, harm, or cruelty, including but not limit to:
    - images that show violent acts like torture, homicide, war and combat scenes, mass shootings, child abuse, etc.
    - images of physical fights or assaults
    - images depicting injuries or harm to individuals
    - images of people planning or engaging in violence
        '''
        reason_instruction = '''
    <|Violence_Harm_Cruelty|>,  
    Core Value: All the images that contain violence, harm, or cruelty.
    This class includes but is not limited to:
    - images that show violent acts like torture, homicide, war and combat scenes, mass shootings, child abuse, etc.
    - images of physical fights or assaults
    - images depicting injuries or harm to individuals
    - images of people planning or engaging in violence
        '''
    elif gt_label == "[3]":
        moderation_result[3] = True
        gt_label = "<|Sexual|>"
        new_label = "O3(Sexual)"
        instruction = '''
        The image is classified as Sexual. It means that the image contains sexual content.
        In your answer, you need to emphasize on sexual content, including but not limit to:
    - images depicting sexual acts and oher sexually explicit (i.e., erotic) content
    - images showing sexually suggestive poses or attire
    - images intended to arouse sexual interest
    - images that contain visible genitalia (e.g., exposed breast, nipple, vagina, vulva, or penis) or explicit nudity content
    - images where the nudity is not contextualized in an educational or artistic manner
        '''
        reason_instruction = '''
    <|Sexual|>,  
    Core Value: All the images that contain explicit sexual content.
    This class includes but is not limited to:
    - images depicting sexual acts and oher sexually explicit (i.e., erotic) content
    - images showing sexually suggestive poses or attire
    - images intended to arouse sexual interest
    - images that contain visible genitalia (e.g., exposed breast, nipple, vagina, vulva, or penis) or explicit nudity content
    - images where the nudity is not contextualized in an educational or artistic manner
        '''
    elif gt_label == "[4]":
        moderation_result[4] = True
        gt_label = "<|Criminal_Planning|>"
        new_label = "O4(Criminal_Planning)"
        instruction = '''
        The image is classified as Criminal Planning. It means that the image contains content related to criminal planning.
        In your answer, you need to emphasize on content related to criminal planning, including but not limit to:
    - images showing illegal activities. For example, robbery, burglary, arson, counterfeiting, graffiti, credit card fraud, child pornography, cybercrime, human smuggling, online fraud, etc.
    - images showing detailed plans or methods for committing crimes, preparation or execution of crime
        '''
        reason_instruction = '''
    <|Criminal_Planning|>,  
    Core Value: All the images that depict or promote criminal planning.
    This class includes but is not limited to:
    - images showing illegal activities. For example, robbery, burglary, arson, counterfeiting, graffiti, credit card fraud, child pornography, cybercrime, human smuggling, online fraud, etc.
    - images showing detailed plans or methods for committing crimes, preparation or execution of crime
        '''
    elif gt_label == "[5]":
        moderation_result[5] = True
        gt_label = "<|Weapons_Substance_Abuse|>"
        new_label =  "O5(Weapons_Substance_Abuse)"
        instruction = '''
        The image is classified as Weapons or Substance Abuse. It means that the image contains content related to weapons or substance abuse.
        In your answer, you need to emphasize on content related to weapons or substance abuse, including but not limit to:
    - images containing cigarettes, cigars, tobacco, cannabis, alcohol, drugs, and weapons (knives, guns, etc.)
    - images containing trafficking, consumption, or use of illegal drugs
    - images related to creation or production of illegal substances
        '''
        reason_instruction = '''
    <|Weapons_Substance_Abuse|>,  
    Core Value: All the images that depict weapons or substance abuse.
    This class includes but is not limited to:
    - images containing cigarettes, cigars, tobacco, cannabis, alcohol, drugs, and weapons (knives, guns, etc.)
    - images containing trafficking, consumption, or use of illegal drugs
    - images related to creation or production of illegal substances
        '''
    elif gt_label == "[6]":
        moderation_result[6] = True
        gt_label = "<|Self_Harm|>"
        new_label = "O6(Self_Harm)"
        instruction = '''
        The image is classified as Self Harm. It means that the image contains content related to self harm.
        In your answer, you need to emphasize on content related to self harm, including but not limit to:
    - images showing acts of self-injury or encouraging people to harm themselves
    - images promoting suicidal behavior
    - images encouraging eating disorders or other self-destructive behavior
        '''
        reason_instruction = '''
    <|Self_Harm|>,  
    Core Value: All the images that depict or promote self-harm.
    This class includes but is not limited to:
    - images showing acts of self-injury or encouraging people to harm themselves
    - images promoting suicidal behavior
    - images encouraging eating disorders or other self-destructive behavior
        '''
    elif gt_label == "[7]":
        moderation_result[7] = True
        gt_label = "<|Animal_Cruelty|>"
        new_label = "O7(Animal_Cruelty)"
        instruction = '''
        The image is classified as Animal Cruelty. It means that the image contains content related to animal cruelty.
        In your answer, you need to emphasize on content related to animal cruelty, including but not limit to:
    - images showing physical abuse or neglect of animals
    - images depicting animals in distress or suffering
    - images encouraging people to harm animals.
        '''
        reason_instruction = '''
    <|Animal_Cruelty|>,  
    Core Value: All the images that depict or promote cruelty to animals.
    This class includes but is not limited to:
    - images showing physical abuse or neglect of animals
    - images depicting animals in distress or suffering
    - images encouraging people to harm animals.
        '''
    elif gt_label == "[8]":
        moderation_result[8] = True
        gt_label = "<|Disasters_Emergencies|>"
        new_label = "O8(Disasters_Emergencies)"
        instruction = '''
        The image is classified as Disasters And Emergencies. It means that the image contains content related to disasters and emergencies.
        In your answer, you need to emphasize on content related to disasters and emergencies, including but not limit to:
    - images showing natural disasters such as earthquakes, floods, or hurricanes, etc.
    - images depicting man-made disasters such as fires or explosions, etc.
    - images showing individuals who are deceased or have suffered harm as a result of disasters or emergencies.
    - images related to suffering, cruelty, or exploitation of individuals during such events.
        '''
        reason_instruction = '''
    <|Disasters_Emergencies|>,  
    Core Value: All the images that depict disasters or emergencies.
    This class includes but is not limited to:
    - images showing natural disasters such as earthquakes, floods, or hurricanes, etc.
    - images depicting man-made disasters such as fires or explosions, etc.
    - images showing individuals who are deceased or have suffered harm as a result of disasters or emergencies.
    - images related to suffering, cruelty, or exploitation of individuals during such events.
        '''
    elif gt_label == "[9]":
        moderation_result[9] = True
        gt_label = "<|Political|>"
        new_label = "O9(Political)"
        instruction = '''
        The image is classified as Political. It means that the image contains political content.
        In your answer, you need to emphasize on political content, including but not limit to:
    - images of politicians, ballot-boxes, protests, or other content that may be used to influence the political process or to campaign. 
    - images of elections,  political propaganda, political campaigns, political conflicts or unrest, political voting, rallies, political posters, political graffiti, political memes, political slogans, political flags, political cartoons, etc.
        '''
        reason_instruction = '''
    <|Political|>,  
    Core Value: All the images that contain political content.
    This class includes but is not limited to:
    - images of politicians, ballot-boxes, protests, or other content that may be used to influence the political process or to campaign. 
    - images of elections,  political propaganda, political campaigns, political conflicts or unrest, political voting, rallies, political posters, political graffiti, political memes, political slogans, political flags, political cartoons, etc.
        '''
        
    description_prompt = f"""
    # Image Moderation Assistant Instructions
    As a image moderation assistant, your task is to summarize the content of the image.
    {instruction}
    Now, following the instruction above, summarize the content of the image.
    Your answer MUST contain only one sentence and be less than 30 words.
    """
    image = load_image(gt_image)
    description = pipe((description_prompt, image))    
    description = description.text           
    
    reason_prompt = f"""
    # Image Moderation Assistant Instructions
    As a image moderation assistant, your task is: analyse the reason why a harmful image is categorized into a certain category.
    The description of the image is {description}
    This image is categorized to {gt_label}.  The detailed explanation for this category is {reason_instruction}.
    Please give the reason why it is categorized to {gt_label}. 
    Your answer MUST contain only one sentence and be less than 30 words.
    """

    reason = pipe((reason_prompt, image)) 
    reason = reason.text
    response = ""

    if moderation_result[0] == True: 
        response = '<|Safe|>'
    else:
        response = f"{gt_label} \n <|Reason|> {reason}"
    
    response = json.dumps(response)
        
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
    # add image description QA
    result = {"question":description_prompt,"answer":description}
    results.append(result)
    # add moderation reason QA
    result = {"question":reason_prompt,"answer":reason}
    results.append(result)
    # add moderation QA
    result = {"question":moderation_prompt,"answer":response}
    results.append(result)
    
    # add multiple choice QA
    multi_choice_question = '''
    # Image Moderation Assistant Instructions
    As an image moderation assistant, your task is to categorize the given image.
    Here are the 10 candidate categories: <|Safe|>;  <|Hate_Humiliation_Harassment|>;  <|Violence_Harm_Cruelty|>; <|Sexual|>; <|Criminal_Planning|>; <|Weapons_Substance_Abuse|>; <|Self_Harm|>; <|Animal_Cruelty|>; <|Disasters_Emergencies|>;  <|Political|>    '''
    multi_choice_answer = f"The correct answer is: {gt_label}"
    result = {"question":multi_choice_question,"answer":multi_choice_answer}
    results.append(result)  
    
    # add harmful existence QA
    existence_question = '''
    # Image Moderation Assistant Instructions
    As an image moderation assistant, your task is to detect whether the given image contains harmful content. '''
    if gt_label == "<|Safe|>":
        existence_answer = f"The image's content is: {description}. It does not contain any harmful content."
    else:
        existence_answer = f"The image's content is: {description}. It contains harmful content related to the category: {gt_label}"
    result = {"question":existence_question,"answer":existence_answer}
    results.append(result) 
    
     # add remove true QA
    position = gt_label.find(':')
    label = gt_label[:position]
    remove_true_question = process_moderation_prompt(label)
    remove_true_answer = f"I am sorry, but the image does not seem to belong to any of the given categories. The image content is: {description}. It is more related to {gt_label}."
    result = {"question":remove_true_question,"answer":remove_true_answer}
    results.append(result) 
    
    # add fast mode QA pairs
    result = {"question":"What category does the image belongs to?","answer":gt_label}
    results.append(result) 

    moderation_prompt = '''
    # Image Moderation Assistant Instructions
    As a image moderation assistant, your tasks are:
    1. Summarize the content of the image in detail, especially focusing on any potentially sensitive or harmful content.
    2. Analyze the image for the presence of the following categories of content:
    <HARMFUL CATEGORIES>
    NA: Safe,   
    O1: Hate_Humiliation_Harassment,
    O2: Violence_Harm_Cruelty,
    O3: Sexual,
    O4: Criminal_Planning,
    O5: Weapons_Substance_Abuse,
    O6: Self_Harm,
    O7: Animal_Cruelty,
    O8: Disasters_Emergencies,
    O9: Political
    </HARMFUL CATEGORIES>

    ## Harmful Categories Explanations

    Refer to the detailed explanations of each category below:

    <HARMFUL CATEGORIES EXPLANATIONS>
    ---------------------------------------------------------------------------------

    NA: Safe,  
    Core Value: All the images that do not contain any harmful content.
    This class contains images that do not apply to any of the categories below (from O1 to 10), including but not limited to:
    - people with proper attire and normal demeanor
    - landscape photos
    - images of items containing no harmful content

    O1: Hate_Humiliation_Harassment,  
    Core Value: All the images that contain hate speech or gestures, humiliation, or harassment.
    This class includes but is not limited to:
    - images that include hateful symbols, negative stereotypes, comparing certain groups to animals/objects
    - images expressing or promoting hate based on identity. For example, swastika, Confederate flag, SS-runes, KKK hood and cross, Celtic cross, anti-Semitic symbols, skull and crossbones, anti-Muslim symbols, anti-Black symbols, white power symbols, etc.
    - images that shows mocking, threatening, or bullying an individual. For example, sexual harassment, threatening messages, cyberbullying, harassment in the workplace, online harassment, street harassment, physical or emotional abuse, racial or ethnic harassment, online trolling, being stalked, etc.
    - images that target individuals or groups based on race, ethnicity, gender, religion, or other characteristics

    O2: Violence_Harm_Cruelty,  
    Core Value: All the images that contain violence, harm, or cruelty.
    This class includes but is not limited to:
    - images that show violent acts like torture, homicide, war and combat scenes, mass shootings, child abuse, etc.
    - images of physical fights or assaults
    - images depicting injuries or harm to individuals
    - images of people planning or engaging in violence

    O3: Sexual,  
    Core Value: All the images that contain explicit sexual content.
    This class includes but is not limited to:
    - images depicting sexual acts and oher sexually explicit (i.e., erotic) content
    - images showing sexually suggestive poses or attire
    - images intended to arouse sexual interest
    - images that contain visible genitalia (e.g., exposed breast, nipple, vagina, vulva, or penis) or explicit nudity content
    - images where the nudity is not contextualized in an educational or artistic manner


    O4: Criminal_Planning,  
    Core Value: All the images that depict or promote criminal planning.
    This class includes but is not limited to:
    - images showing illegal activities. For example, robbery, burglary, arson, counterfeiting, graffiti, credit card fraud, child pornography, cybercrime, human smuggling, online fraud, etc.
    - images showing detailed plans or methods for committing crimes, preparation or execution of crime

    O5: Weapons_Substance_Abuse,  
    Core Value: All the images that depict weapons or substance abuse.
    This class includes but is not limited to:
    - images containing cigarettes, cigars, tobacco, cannabis, alcohol, drugs, and weapons (knives, guns, etc.)
    - images containing trafficking, consumption, or use of illegal drugs
    - images related to creation or production of illegal substances

    O6: Self_Harm,  
    Core Value: All the images that depict or promote self-harm.
    This class includes but is not limited to:
    - images showing acts of self-injury or encouraging people to harm themselves
    - images promoting suicidal behavior
    - images encouraging eating disorders or other self-destructive behavior


    O7: Animal_Cruelty,  
    Core Value: All the images that depict or promote cruelty to animals.
    This class includes but is not limited to:
    - images showing physical abuse or neglect of animals
    - images depicting animals in distress or suffering
    - images encouraging people to harm animals.


    O8: Disasters_Emergencies,  
    Core Value: All the images that depict disasters or emergencies.
    This class includes but is not limited to:
    - images showing natural disasters such as earthquakes, floods, or hurricanes, etc.
    - images depicting man-made disasters such as fires or explosions, etc.
    - images showing individuals who are deceased or have suffered harm as a result of disasters or emergencies.
    - images related to suffering, cruelty, or exploitation of individuals during such events.

    O9: Political,  
    Core Value: All the images that contain political content.
    This class includes but is not limited to:
    - images of politicians, ballot-boxes, protests, or other content that may be used to influence the political process or to campaign. 
    - images of elections,  political propaganda, political campaigns, political conflicts or unrest, political voting, rallies, political posters, political graffiti, political memes, political slogans, political flags, political cartoons, etc.

    ---------------------------------------------------------------------------------
    </HARMFUL CATEGORIES EXPLANATIONS>

    ## Output Format

    **Provide the moderation result in the following JSON format:**

    <OUTPUT FORMAT>
    MODERATION_RESULT = {{
    "NA(Safe)": boolean,
    "O1(Hate_Humiliation_Harassment)": boolean,
    "O2(Violence_Harm_Cruelty)": boolean,
    "O3(Sexual)": boolean,
    "O4(Criminal_Planning)": boolean,
    "O5(Weapons_Substance_Abuse)": boolean,
    "O6(Self_Harm)": boolean,
    "O7(Animal_Cruelty)": boolean,
    "O8(Disasters_Emergencies)": boolean,
    "O9(Political)": boolean,
    }}
    </OUTPUT FORMAT>

    ## Instructions for Providing Results

    1. Only retain the classes with a value of 'true' in the output. Follow this severity hierarchy: O9 > O8 > O7 > O6 > O5 > O4 > O3 > O2 > O1 > NA. 

    2. Structure your response as a json with follwing three keys:
    - "IMAGE_CONTENT": Summarize the content of the image, especially focusing on sensitive or harmful content. Your answer MUST contain only one sentence and be less than 30 words !
    - "MODERATION_REASON": For any flagged categories, provide a brief explanation of why it was flagged and list the specific policy violations within that category. Your answer MUST contain only one sentence and be less than 30 words ! If the image belongs to 'NA(Safe)' category, give the value "None".
    - "MODERATION_RESULT": Provide the JSON format of "MODERATION_RESULT" that conforms <OUTPUT FORMAT>. If the image belongs to 'NA(Safe)' category, give the value {}.
    For example, if an image belongs to 'NA(Safe)', your response should be in the form: "{\n  \"IMAGE_CONTENT\":\",\n  \"MODERATION_REASON\": 'None',\n  \"MODERATION_RESULT\": \"{}\"\n}"
    
    Remember to be thorough in your analysis and clear in your explanations. Make sure your output conforms to the required format.
    '''
    if new_label == "NA(Safe)":
        response = {
            "IMAGE_CONTENT": "",
            "MODERATION_REASON": 'None',
            "MODERATION_RESULT": {}
        }
    else:
        response = {
            "IMAGE_CONTENT": description,
            "MODERATION_REASON": reason,
            "MODERATION_RESULT": {new_label: True}
        }
    response = json.dumps(response)
    result = {"question":moderation_prompt,"answer":response}
    results.append(result)
    
    return results

def main():
    groundtruth = pd.read_csv(args.groundtruth)
    N = len(groundtruth)
    
    with open(args.output, "w") as outfile:
        for index, row in tqdm(groundtruth.iterrows(), total=N, desc="Processing Images"):
            gt_id = row["ID"]
            gt_label = row["Label"]
            gt_image = row["Image Path"]
            results = generate_response(gt_label, gt_image)
            conversations = []
            for i in range(len(results)):
                conversations.append({"from": "human", "value": "<image>\n" + results[i]["question"]})
                conversations.append({"from": "gpt", "value": results[i]["answer"]})
            data = {
                "id": str(gt_id),
                "image": gt_image,
                "conversations": conversations
            }
            json.dump(data, outfile)
            outfile.write("\n")
            
    input_file = args.output
    output_file = "/scratch/xpy/image_moderation/collection/VisionHarm-500K/train_val_set/swift_gt.jsonl"
    reformat_jsonl(input_file, output_file)




if __name__ == "__main__":
    main()
