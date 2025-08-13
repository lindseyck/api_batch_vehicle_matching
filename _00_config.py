"""
Updated May the 4th, 2025
"""

import os, openai

# === API Key ===
OPENAI_API_KEY = "your openai key here"
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# === Directory Paths ===
INPUT_DIR = "Data/Original Data/"
OUTPUT_DIR = "Data/Input Files/"
RESULTS_DIR = "Data/Match Result/"

# === Directory Creation ===
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# === Batch Input/Output File Names ===
BATCH_INPUT = "batch_input.jsonl"
BATCH_OUTPUT = "batch_output.jsonl"
MATCH_FILE = "matched_vehicle_models.csv"
OUTPUT_FILE = "matched_parts.csv"

# === Vendor and Master File Paths ===
VENDOR_FILE = "Part_Info_and_Fitment.xlsx" # Update this path to the correct vendor file. 
VENDOR_SHEET_NAME = "PART FITMENT" # Update this path to the correct vendor file sheet name for vehicle matching. If there is no specific sheet name, set to None.
MASTER_FILE = "(Bentley Copy) CPP - GOLDEN COPY year_make_model_master_ultimate.xlsx"
MASTER_SHEET_NAME = "year_make_model_master_ultimate"

# === Columns for Matching ===
COLS_TO_USE = ["Year", "Make", "Model"]

# === Define High Confidence Threshold ===
HIGH_CONFIDENCE_THRESHOLD = 0.9 # Set the threshold for high confidence vehicle matches to include in the final output.

# === Model Names ===
MODEL_FULL = "gpt-4-1106-preview" # Make sure to use models that are compatible with batch processing.
MODEL_COST_SAVER = "gpt-3.5-turbo-1106" # ""

# === Model Cost Rates (per 1K tokens) ===
MODEL_COSTS = {
MODEL_FULL: {"input": 0.01, "output": 0.03}, # Monitor OpenAI for any changes to the model cost rates.
MODEL_COST_SAVER: {"input": 0.001, "output": 0.002} # Monitor OpenAI for any changes to the model cost rates.
}

# === Estimated Input Tokens ===
EST_TOKENS = {
MODEL_FULL: {"input": 350, "output": 150}, # Estimated input and output tokens per row for the full model. Update as needed.
MODEL_COST_SAVER: {"input": 300, "output": 75} # Estimated input and output tokens per row for the cost-saving model. Update as needed.
}

# === Cost Saver Mode Configuration ===
COST_SAVER_CONFIG = {
    "model": MODEL_COST_SAVER,          # Model to use for cost-saving mode.
    "alt_match_threshold": 80,          # Only show alternatives if confidence < this threshold. Set to 100 to include all.
    "alt_match_count": 1,               # Number of alternative matches to show. Set to 0 to disable.
    "retry_enabled": False,             # Disables retrying failed rows. Set to True to retry failed rows.
    "include_reason": True,             # Include the reason for the match. Set to False to exclude.
    "max_token_budget_dollars": 2.00   # Warn or exit if projected cost exceeds this
}

# === Enable or Disable Cost Saver Mode ===
COST_SAVER_MODE = True # Set to True to enable cost-saving mode. Set to False to disable cost-saving mode.

# === Set row limit for testing ===
TEST_ROW_LIMIT = None # Set to None to process all rows or set a specific number for testing. 
