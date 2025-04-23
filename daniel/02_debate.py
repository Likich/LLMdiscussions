"""
This resumes / starts a debate between multiple LLMs.
We expect the input file to be a JSON file with the following structure:

{
    meta: {
        llms: {
            [llm_key]: {
                llm_model_name: str,
                llm_display_name: str
            }
        },
    },
    prompt: {
        system_initial: str,
        system_debate: str,
        
        questions: {
            [question_key]: [question_text]
        }
    },
    debate: {
        [question_key]: {
            [llm_key]: {
                responses: [response_text]
            }
        }
    }
}

Where each debate corresponds to a different LLM.
The responses are the responses from the LLM to the questions in the prompt.
"""

import argparse
import json
import os
import requests as r

args = argparse.ArgumentParser()
args.add_argument('--input', type=str, required=True)
args.add_argument('--max-rounds', type=int, required=False, default=2)
args.add_argument('--ollama-url', type=str, required=False, default='http://10.162.42.104:11434/api/chat')
args.add_argument('--auth-header', type=str, required=False)

args = args.parse_args()

assert args.input.endswith('.json'), 'Input file must be a JSON file'
assert os.path.exists(args.input), 'Input file does not exist'

with open(args.input, 'r') as f:
    data = json.load(f)
    
# Check if there alreay is a checkpoint
if os.path.exists(args.input + ".tmp"):
    with open(args.input + ".tmp", 'r') as f:
        data = json.load(f)

    print(f"Checkpoint found, resuming from {args.input}.tmp")

class Message:
    def __init__(self, role, content):
        self.role = role
        self.content = content
        
    def __str__(self):
        return f"<<{self.role}>> {self.content}"

class AlreadyRespondedError(Exception):
    pass

class MaxRoundsExceededError(Exception):
    pass



def extract_response(response_json: dict) -> str:
    is_groq = 'groq' in args.ollama_url
    is_lambda = 'lambda.ai' in args.ollama_url
    is_ollama = not is_groq
    
    if is_ollama:
        return response_json['message']['content']
    elif is_groq or is_lambda:
        return response_json['choices'][0]['message']['content']
    


def generate_response(messages: list[Message], llm_idx: str) -> tuple[str, str]:
    """
    We query the LLM with the messages and get a response.
    We also return the token probabilities for the response.
    """
    
    llm_model = data['meta']['llms'][llm_idx]['llm_model_name']
    initial_response = r.post(f"{args.ollama_url}", json={
        "model": llm_model,
        "messages": [ { 'role': message.role, 'content': message.content } for message in messages ],
        "temperature": 0.2,
        "stream": False,
    }, headers={
        'Authorization': args.auth_header
    })
    
    initial_response.raise_for_status()
    inital_response = initial_response.json()['message']['content']
    
    print(f"LLM {llm_idx} responded with: {inital_response}")
    
    # Now we summarize the response
    summary_response = r.post(f"{args.ollama_url}", json={
        "model": llm_model,
        "messages": [
            { 'role': 'system', 'content': "Please summarize the users decision or code in one sentence. Just give the code and the justification, as if it were your own." },
            { 'role': 'user', 'content': inital_response }
        ],
        "temperature": 0.2,
        "stream": False,
    }, headers={
        'Authorization': args.auth_header
    })
    
    summary_response.raise_for_status()
    summary_response = summary_response.json()['message']['content']
    
    print(f"\n\n\nPrompting {llm_idx}, Messages were:\n" + ("\n".join(map(str, messages))))
    return inital_response, summary_response


def format_debate_prompt(data, q_idx, llm_idx):
    """
    Given the data and the question index, format the prompt for the debate.
    This function will return the messages to be sent to the LLM and a boolean indicating if this is the initial round.
    """
    
    messages: list[Message] = []
    debate_data = data['debate'][q_idx]
    
    # Check if this is the initial round
    if llm_idx not in debate_data or len(debate_data[llm_idx]['responses']) == 0:
        messages.append(Message('system', data['prompt']['system_initial']))
        messages.append(Message('user', data['prompt']['questions'][q_idx]))
        
        return messages, True
        
    else:
        messages.append(Message('system', data['prompt']['system_debate']))
        messages.append(Message('user', data['prompt']['questions'][q_idx]))
        
        # Now we append all the responses from all rounds that are already completed
        round_idx = 0
        all_llm_keys = list(data['meta']['llms'].keys())
        
        while True:
            # Check if the round is completed, i.e., all LLMs have responded
            all_llms_responded = [ len(debate_data[llm_key]['responses']) > round_idx for llm_key in all_llm_keys ]
            if not all(all_llms_responded): break

            # Append the responses
            message = f"Round {round_idx}:\n\n"
            for llm_key in all_llm_keys:
                llm_display_name = data['meta']['llms'][llm_key]['llm_display_name']
                llm_response = debate_data[llm_key]['responses'][round_idx]
                llm_response_full, llm_response_summary = llm_response
                
                message += f"{llm_display_name}: {llm_response_summary}\n"
            
            messages.append(Message('assistant', message))
            round_idx += 1
        
        # We need to check if our LLM has already responded to this round
        if len(debate_data[llm_idx]['responses']) > round_idx:
            raise AlreadyRespondedError(f"LLM {llm_idx} has already responded to round {round_idx}")
        
        # We need to check if we have exceeded the max rounds
        if round_idx >= args.max_rounds:
            raise MaxRoundsExceededError(f"Max rounds exceeded for question {q_idx}")
        
        return messages, False

def debate_llm_question(data, q_idx, llm_idx):
    messages, is_initial = format_debate_prompt(data, q_idx, llm_idx)
    
    if is_initial:
        response = generate_response(messages, llm_idx)
        data['debate'][q_idx][llm_idx] = { 'responses': [response] }
    
    else:
        response = generate_response(messages, llm_idx)
        data['debate'][q_idx][llm_idx]['responses'].append(response)

    return data


# We try to keep the same model in cache as long as possible.
# Hence, we will loop through all the questions for each LLM

while True:
    any_llm_responded = False
    
    for llm_idx in data['meta']['llms']:
        for q_idx in data['prompt']['questions']:
            try:
                data = debate_llm_question(data, q_idx, llm_idx)
                any_llm_responded = True
            except AlreadyRespondedError as e:
                print(e)
                continue
            except MaxRoundsExceededError as e:
                print(e)
                continue
            except Exception as e:
                print(e)
                # Stack Trace for debugging
                raise e

            with open(args.input + ".tmp", 'w') as f:
                json.dump(data, f, indent=4, sort_keys=True, ensure_ascii=False)
    
    with open(args.input + ".debate", 'w') as f:
                json.dump(data, f, indent=4, sort_keys=True, ensure_ascii=False)
               
    if not any_llm_responded: break
