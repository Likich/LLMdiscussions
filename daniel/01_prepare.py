"""
This script takes JSON-files from Lika and brings them into my format :))
"""
import json

MAX_QUESTIONS = 1

with open('lika.json') as fi, open('input.json', 'w') as fo:
    data_input = json.load(fi)
    data_output = {
        'meta': {
            'llms': {
                # 'llm1': {
                #     'llm_model_name': 'llama3.2:3b',
                #     'llm_display_name': 'Model A'
                # },
                # 'llm2': {
                #     'llm_model_name': 'phi4:14b',
                #     'llm_display_name': 'Model B'
                # },
                # 'llm3': {
                #     'llm_model_name': 'mistral-small3.1:24b',
                #     'llm_display_name': 'Model C'
                # },
                # 'llm4': {
                #     'llm_model_name': 'deepseek-r1:7b',
                #     'llm_display_name': 'Model D'
                # },
                # 'llm5': {
                #     'llm_model_name': 'gemma3:4b',
                #     'llm_display_name': 'Model E'
                # },
                'llm1': {
                    'llm_model_name': 'llama-4-scout-17b-16e-instruct',
                    'llm_display_name': 'Model A'
                },
                'llm2': {
                    'llm_model_name': 'deepseek-llama3.3-70b',
                    'llm_display_name': 'Model B'
                },
                'llm3': {
                    'llm_model_name': 'hermes3-8b',
                    'llm_display_name': 'Model C'
                },
            }
        },
        'prompt': {
            'system_initial': "A code is often a word or short phrase that symbolically assigns a summative, salient, essence-capturing and/or evocative attribute to a portion of language-based or visual data. Perform thematic analysis on the following comment in the user message and generate a concise qualitative code.",
            'system_debate': "You are a helpful coding AI. You are collaborating with other coding AIs to refine qualitative codes. You will be given a comment and a code. Your task is to discuss the code with the other AIs and refine it collaboratively.",
            'questions': {
                f"q{idx:05}": item['comment'] for idx, item in enumerate(data_input[:MAX_QUESTIONS])
            }
        },
        'debate': {
            f"q{idx:05}": {} for idx, item in enumerate(data_input[:MAX_QUESTIONS])
        }
    }
    
    json.dump(data_output, fo, indent=4, ensure_ascii=False)
    
