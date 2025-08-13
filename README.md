Vehicle Matching Pipeline - README
==================================

Last Updated: August 2025

This pipeline uses AI to match antique vehicle models from vendor files to a master database. It works by processing your files, submitting vehicle model data to OpenAI's API, and generating the best match results based on compatibility.
Data files have been omitted from the repo for privacy purposes.

Required Setup
--------------
1. Python 3.9+ must be installed.
2. Install required packages:
   Run this in your terminal:
       pip install openai pandas openpyxl python-dotenv

3. File Structure:
   Ensure the following folder structure exists in your working directory:
       Data/
         └── Original Data/
         └── Input Files/
         └── Match Result/
         └── Metadata Output/

4. Add Your Input Files:
   Place your vendor file (Excel) and master vehicle list (Excel) into the 
   Data/Original Data/ folder. Update the file names in _00_config.py.

Step-by-Step Instructions
-------------------------

Step 1: Prepare Input Files
---------------------------
Run this script:
    python _01_file processing.py

What it does:
- Extracts vehicle Year, Make, Model from your vendor and master files
- Saves filtered data into Input Files/ for matching

Step 2: Match Vehicle Models Using AI
-------------------------------------
Run:
    python _02_api_batch_vehicle_matching.py

What it does:
- Builds prompts and submits them to OpenAI's API for model matching
- Saves GPT responses to Match Result/batch_output.jsonl
- Outputs final matches (including exact matches and AI-predicted matches) to:
    Data/Match Result/matched_vehicle_models.csv

If Cost Saver Mode is enabled in _00_config.py:
- Uses a lower-cost GPT model
- Shows alternative matches only when confidence is low
- May skip match "reasons" to save tokens
- Disables retries for failed rows

Step 3 (Optional): Estimate Token Usage
---------------------------------------
Run:
    python _00_estimate_token_usage.py

What it does:
- Reads the latest batch_output.jsonl
- Calculates average input/output token usage across responses

Configuration Notes (_00_config.py)
-----------------------------------
This file controls:
- File paths and sheet names
- Model settings and token cost thresholds
- Whether cost-saver mode is enabled
- OpenAI API key (stored securely using a .env file)

To switch cost-saver mode:
    COST_SAVER_MODE = True  # or False

Output Files
------------
- batch_input.jsonl         → All AI prompts sent to OpenAI
- batch_output.jsonl        → Raw AI response data
- matched_vehicle_models.csv → Final matching results
- matched_parts.xlsx        → Final grouped part fitment output (multi-sheet Excel)
- matched_parts_with_metadata.xlsx   → Final output with metadata
