# - Use this code to estimate the input and output token usage of batch files after running _02_api_batch_vehicle_matching (recommend using a test limit for this process) to inform estimated token costs entered in the _00_congfig.py file.

import json

with open("Data/Match Result/batch_input.jsonl", "r", encoding="utf-8") as f:
    input_data = [json.loads(line) for line in f]

with open("Data/Match Result/batch_output.jsonl", "r", encoding="utf-8") as f:
    output_data = [json.loads(line) for line in f if line.strip()]

input_tokens = []
output_tokens = []

for out_row in output_data:
    try:
        usage = out_row["response"]["body"]["usage"]
        input_tokens.append(usage["prompt_tokens"])
        output_tokens.append(usage["completion_tokens"])
    except KeyError:
        continue

avg_input = sum(input_tokens) / len(input_tokens)
avg_output = sum(output_tokens) / len(output_tokens)

print("Average input tokens:", avg_input)
print("Average output tokens:", avg_output)
