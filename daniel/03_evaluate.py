import argparse
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import math

def calculate_perplexity(token_log_probabilities: list[float]) -> float:
    """
    Calculate perplexity from a list of token log probabilities.
    Perplexity = exp(- average log probability)
    """
    if not token_log_probabilities:
        return float('inf') # Avoid division by zero

    average_log_probability = sum(token_log_probabilities) / len(token_log_probabilities)
    perplexity = math.exp(-average_log_probability)
    return perplexity

def get_log_probabilities(model, tokenizer, context_messages, target_response):
    """
    Computes the log probabilities of the target_response given the context_messages
    using the provided model and tokenizer.
    """
    # Format messages for the model - assuming a chat format like "user: ... assistant: ..."
    # This might need adjustment based on the specific model's required format
    formatted_context = tokenizer.apply_chat_template(
        [{"role": msg.role, "content": msg.content} for msg in context_messages],
        tokenize=False,
        add_generation_prompt=True
    )

    full_text = formatted_context + target_response

    # Tokenize the full text
    encoded_full_text = tokenizer(full_text, return_tensors='pt')
    input_ids = encoded_full_text.input_ids
    attention_mask = encoded_full_text.attention_mask

    # Tokenize the context separately to find where the target response starts
    encoded_context = tokenizer(formatted_context, return_tensors='pt')
    context_length = encoded_context.input_ids.shape[1]

    # Get model outputs (logits)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # Get the logits corresponding to the target response tokens
    # We need to shift the logits to the left by one position to get the logits
    # that predict the current token based on the previous tokens.
    logits_target_response = logits[:, context_length - 1:-1, :]

    # Get the token IDs of the target response
    target_response_ids = input_ids[:, context_length:]

    # Calculate the log probabilities of the target response tokens
    log_probabilities = torch.log_softmax(logits_target_response, dim=-1)

    # Gather the log probabilities for the actual target response tokens
    # Use .squeeze(0) if batch size is 1
    log_probs = torch.gather(log_probabilities, 2, target_response_ids.unsqueeze(-1)).squeeze(-1).squeeze(0)

    return log_probs.tolist(), target_response_ids.squeeze(0).tolist()


# Custom Message class definition (assuming it's not imported from 02_debate)
class Message:
    def __init__(self, role, content):
        self.role = role
        self.content = content

    def __str__(self):
        return f"<<{self.role}>> {self.content}"


def format_debate_context(data, q_idx, llm_idx, target_round_idx):
    """
    Given the data, question index, the scoring LLM index, and the target round index,
    format the messages to be sent to the LLM to get the context up to the round before the target response.
    """
    messages: list[Message] = []
    debate_data = data['debate'][q_idx]

    messages.append(Message('system', data['prompt']['system_debate'])) # Always include the debate system prompt
    messages.append(Message('user', data['prompt']['questions'][q_idx]))

    # Append all the responses from all rounds *before* the target round
    all_llm_keys = list(data['meta']['llms'].keys())

    for current_round_idx in range(target_round_idx):
         # Check if the round is completed, i.e., all LLMs have responded in this round
        all_llms_responded_in_round = [ len(debate_data.get(llm_key, {}).get('responses', [])) > current_round_idx for llm_key in all_llm_keys ]
        if not all(all_llms_responded_in_round):
            # If this round is not complete, we cannot use it as context for a later round
            # This should not happen if we are processing an existing debate file round by round,
            # but it's a safeguard.
            break

        # Append the responses summaries from the current round as assistant messages
        message_content = f"Round {current_round_idx}:\n\n"
        for llm_key in all_llm_keys:
            llm_display_name = data['meta']['llms'][llm_key]['llm_display_name']
            # Use the summary response as it's used in the debate prompt
            llm_response_full, llm_response_summary = debate_data[llm_key]['responses'][current_round_idx]

            message_content += f"{llm_display_name}: {llm_response_summary}\n"

        messages.append(Message('assistant', message_content))

    return messages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to the input .json debate file')
    parser.add_argument('--output', type=str, required=True, help='Path to save the output .json.perplex file')
    args = parser.parse_args()

    assert args.input.endswith('.json'), 'Input file must be a JSON file'
    assert os.path.exists(args.input), 'Input file does not exist'
    assert args.output.endswith('.json.perplex'), 'Output file must have .json.perplex extension'

    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    perplexity_results = {}

    llm_meta = data['meta']['llms']
    question_keys = data['prompt']['questions'].keys()

    # Process perplexity for each scoring LLM
    for scoring_llm_key, scoring_llm_info in llm_meta.items():
        scoring_model_name = scoring_llm_info['llm_model_name']
        print(f"Loading model and tokenizer for scoring LLM: {scoring_model_name}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(scoring_model_name)
            model = AutoModelForCausalLM.from_pretrained(scoring_model_name)
            if torch.cuda.is_available():
                model.to('cuda')
        except Exception as e:
            print(f"Error loading model {scoring_model_name}: {e}")
            continue # Skip this scoring LLM if model loading fails

        perplexity_results[scoring_llm_key] = {}

        # Iterate through each question
        for q_idx in question_keys:
            perplexity_results[scoring_llm_key][q_idx] = {}
            debate_data = data['debate'].get(q_idx, {})

            # Determine the number of rounds for this question
            num_rounds = 0
            if debate_data:
                # Assuming all LLMs have the same number of rounds for a given question if they participated
                # Find the maximum number of responses for any LLM in this question
                num_rounds = max([len(llm_data.get('responses', [])) for llm_data in debate_data.values()], default=0)


            # Iterate through each round
            for round_idx in range(num_rounds):
                perplexity_results[scoring_llm_key][q_idx][round_idx] = {}

                # Iterate through each target LLM's response in this round
                for target_llm_key, target_llm_data in debate_data.items():
                    if target_llm_key == scoring_llm_key:
                        continue # Don't score an LLM's response with its own model

                    if len(target_llm_data.get('responses', [])) > round_idx:
                        target_response_full, target_response_summary = target_llm_data['responses'][round_idx]

                        print(f"  Scoring target LLM {target_llm_key}'s response in q '{q_idx}', round {round_idx} using {scoring_llm_key}'s model")

                        # Get the context messages for the scoring LLM up to the round before the target response
                        context_messages = format_debate_context(data, q_idx, scoring_llm_key, round_idx)

                        # Compute token log probabilities and get tokenized IDs
                        try:
                            token_log_probabilities, tokenized_response_ids = get_log_probabilities(
                                model, tokenizer, context_messages, target_response_full
                            )

                            # Store the results
                            if target_llm_key not in perplexity_results[scoring_llm_key][q_idx][round_idx]:
                                perplexity_results[scoring_llm_key][q_idx][round_idx][target_llm_key] = {}

                            perplexity_results[scoring_llm_key][q_idx][round_idx][target_llm_key] = {
                                'tokenized_response_ids': tokenized_response_ids,
                                'token_log_probabilities': token_log_probabilities
                            }
                            # Optionally calculate and store perplexity score here too
                            # perplexity_score = calculate_perplexity(token_log_probabilities)
                            # perplexity_results[scoring_llm_key][q_idx][round_idx][target_llm_key]['perplexity'] = perplexity_score

                        except Exception as e:
                            print(f"    Error computing log probabilities: {e}")
                            # Store error or empty results if computation fails
                            perplexity_results[scoring_llm_key][q_idx][round_idx][target_llm_key] = {
                                'error': str(e)
                            }


    # Save the results to the output file
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(perplexity_results, f, indent=4, ensure_ascii=False)

    print(f"Perplexity results saved to {args.output}")


if __name__ == "__main__":
    main()