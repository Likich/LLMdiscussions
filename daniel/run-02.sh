LAMBDA_TOKEN=$(cat .lambda-token)

python3 02_debate.py --input input.json \
    --ollama-url "https://api.lambda.ai/v1/chat/completions" \
    --auth-header "Bearer $LAMBDA_TOKEN" 