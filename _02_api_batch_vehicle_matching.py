"""
Updated May 7, 2025
"""

# === SCRIPT OVERVIEW ===
# - configure(): Loads API key from .env and initializes OpenAI client
# - load_data(): Loads vendor and master vehicle CSV files
# - build_master_lookup(): Builds Year_Make lookup and valid Car_IDs
# - estimate_token_costs(): Calculates token usage and cost estimates
# - detect_engine(): Extracts engine size info from model name
# - build_prompts(): Creates prompts for GPT batch matching
# - parse_response(): Extracts structured info from GPT response
# - save_batch_input(): Saves batch job to .jsonl
# - submit_batch(): Uploads and monitors GPT batch job
# - download_batch_output(): Downloads GPT batch results
# - parse_batch_output(): Parses results, retries failures
# - append_exact_matches(): Appends rows from exact match file
# - save_final_output(): Outputs final match results to CSV
# - setup_mode(): Configures cost-saving mode and model settings
# - main(): Orchestrates full pipeline

# === SCRIPT USAGE ===
# - Run the _01_file_processing.py script first to generate the vendor and master match lists.
# - Make sure the correct paths and filenames are set in _00_config.py.

import os, time, json, re, pandas as pd, openai
from _00_config import (
    RESULTS_DIR,
    OUTPUT_DIR,
    BATCH_INPUT,
    BATCH_OUTPUT,
    MATCH_FILE,
    EST_TOKENS,
    MODEL_COSTS,
    COST_SAVER_CONFIG,
    COST_SAVER_MODE,
    TEST_ROW_LIMIT,
    MODEL_FULL
)

# === CONFIGURATION ===
def configure():
    from _00_config import OPENAI_API_KEY
    return openai.OpenAI(api_key=OPENAI_API_KEY)

# === DATA LOADING AND PREPROCESSING ===
def load_data(output_dir, test_limit):
    """
    Loads the vendor and master car model CSV files. Applies an optional row limit.

    Args:
        output_dir (str): Path to directory containing CSV input files.
        test_limit (int or None): Optional cap on rows for vendor_df.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: vendor_df and master_df
    """
    vendor_path = os.path.join(output_dir, "vendor_match_list.csv")
    master_path = os.path.join(output_dir, "master_match_list.csv")

    vendor_df = pd.read_csv(vendor_path)
    master_df = pd.read_csv(master_path)

    if test_limit:
        vendor_df = vendor_df.head(test_limit)
    return vendor_df, master_df

# === COST ESTIMATION ===
def estimate_token_costs(batch_rows, model_name, model_costs, cost_saver_mode=False, cost_saver_config=None):
    """
    Dynamically estimates token usage and cost based on batch rows and cost saver config.

    Args:
        batch_rows (list): List of API requests.
        model_name (str): Model name.
        model_costs (dict): Dict of cost per 1K tokens.
        cost_saver_mode (bool): If cost saver config is used.
        cost_saver_config (dict): Optional config for tuning token estimate.

    Returns:
        tuple: (total_prompt_tokens, avg_prompt_tokens, input_cost, output_cost, total_cost)
    """
    token_estimates = EST_TOKENS.get(model_name, {"input": 350, "output": 150})

    n = len(batch_rows)
    est_input = token_estimates["input"]
    est_output = token_estimates["output"]

    total_input = n * est_input
    total_output = n * est_output
    avg_tokens = est_input

    input_cost = (total_input / 1000) * model_costs["input"]
    output_cost = (total_output / 1000) * model_costs["output"]
    total_cost = input_cost + output_cost

    return total_input, avg_tokens, input_cost, output_cost, total_cost

# === ISOLATE VALID POTENTIAL MATCHES ===
def build_master_lookup(vendor_df, master_df):
    """
    Constructs a lookup dictionary of master models and a set of valid composite Car_IDs
    based only on Year+Make combinations found in the vendor data.

    Args:
        vendor_df (pd.DataFrame): Vendor input dataframe.
        master_df (pd.DataFrame): Master reference dataframe.

    Returns:
        Tuple[dict, set]: 
            - master_lookup: dict mapping Year_Make to list of Models
            - valid_ids: set of valid Year_Make_Model strings
    """
    vendor_keys = {f"{row['Year']}_{row['Make']}" for _, row in vendor_df.iterrows()}
    master_lookup, valid_ids = {}, set()
    added_count, skipped_count = 0, 0

    for _, row in master_df.iterrows():
        key = f"{row['Year']}_{row['Make']}"
        if key in vendor_keys:
            model = row['Model']
            car_id = f"{key}_{model}"
            valid_ids.add(car_id)
            master_lookup.setdefault(key, []).append(model)
            added_count += 1
        else:
            skipped_count += 1

    print(f"Master Car IDs added to lookup: {added_count}")
    print(f"Master Car IDs skipped (no vendor match): {skipped_count}")

    return master_lookup, valid_ids

# === ENGINE DETECTION ===
def detect_engine(model):
    """
    Detects engine size or cylinder count from a model string, supporting:
    - numeric formats (e.g., '350')
    - spelled-out numbers (e.g., 'Three Fifty')
    - cylinder indicators (e.g., 'Four', 'Six', 'Eight' as standalone terms)

    Args:
        model (str): Car model string from vendor file.

    Returns:
        str or None: Detected engine identifier like '350' or '6' if found.
    """
    model_lower = model.lower()

    cylinder_words = {
        "four": "4",
        "six": "6",
        "eight": "8"
    }

    for word, digit in cylinder_words.items():
        if re.search(rf"\b{word}\b", model_lower):
            return digit

    hundreds = {
        "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7",
        "eight": "8", "nine": "9"
    }
    tens = {
        "twenty": "2", "thirty": "3", "forty": "4",
        "fifty": "5", "sixty": "6", "seventy": "7",
        "eighty": "8", "ninety": "9"
    }
    units = {
        "zero": "0", "one": "1", "two": "2",
        "three": "3", "four": "4", "five": "5",
        "six": "6", "seven": "7", "eight": "8",
        "nine": "9"
    }

    spelled_to_numeric = {}
    for h_word, h_digit in hundreds.items():
        for t_word, t_digit in tens.items():
            for u_word, u_digit in units.items():
                number = h_digit + t_digit + u_digit
                for sep in [" ", "-"]:
                    phrase = sep.join([h_word, t_word, u_word])
                    spelled_to_numeric[phrase] = number

    for h_word, h_digit in hundreds.items():
        for t_word, t_digit in tens.items():
            number = h_digit + t_digit + "0"
            for sep in [" ", "-"]:
                phrase = sep.join([h_word, t_word])
                spelled_to_numeric[phrase] = number

    for phrase, numeric in spelled_to_numeric.items():
        if phrase in model_lower:
            return numeric

    match = re.findall(r"\b(2\d{2}|3\d{2}|4\d{2})\b", model)
    return match[0] if match else None

# === PROMPT GENERATION ===
def build_prompts(vendor_df, master_lookup, valid_ids, model_name, COST_SAVER_MODE, COST_SAVER_CONFIG):
    """
    Builds structured prompts for OpenAI GPT vehicle matching based on vendor and master lists.

    Args:
    vendor_df (pd.DataFrame): Vendor vehicle data.
    master_lookup (dict): Dictionary of potential match candidates.
    valid_ids (set): Set of valid composite match IDs.
    model_name (str): OpenAI model to use for completions.
    cost_saver_mode (bool): Whether cost saver mode is enabled.
    cost_saver_config (dict): Cost saver mode configuration settings.

Returns:
    Tuple[list, dict, dict]: List of API batch rows, prompts per car_id, metadata per car_id.
"""
    batch_rows, id_to_prompt, id_to_meta = [], {}, {}
    for _, row in vendor_df.iterrows():
        year, make, model, car_id = row["Year"], row["Make"], row["Model"], row["Car_ID"]
        key = f"{year}_{make}"
        candidates = master_lookup.get(key, [])
        if not candidates:
            continue
        engine = detect_engine(model)
        if engine:
            candidates = [m for m in candidates if engine in m]
        if not candidates:
            continue
        candidates = [c for c in candidates if f"{year}_{make}_{c}" in valid_ids]
        if not candidates:
            continue

        if COST_SAVER_MODE:
            alt_count = COST_SAVER_CONFIG.get("alt_match_count", 3)
            include_reason = COST_SAVER_CONFIG.get("include_reason", True)
        else:
            alt_count = 3
            include_reason = True

        prompt = (
            f"You are an expert AI specializing in vehicle model matching for accurate part fitment.\n"
            f"Your task is to match the provided Vendor Model to the most appropriate Master Model from the list of candidates.\n\n"
            f"Vendor Model Provided: {year}_{make}_{model}\n"
            f"Potential Master Models (Candidates): {' | '.join(candidates)}\n\n"
            f"**Engine Consideration Rules:**\n"
            f"Analyze the Vendor Model: Use 'Four'/'Six'/'Eight' for engine size match priority.\n\n"
            f"Implicit Knowledge: Use your internal knowledge base. Some models or trims only come with specific engine sizes; consider this even if engine size isn't explicitly mentioned.\n"
            f"Primary Goal: Use this engine information as a crucial factor in determining the *correctness* of the match for part fitment.\n\n"
            f"**Output Requirements:**\n"
            f"- Vendor Car ID: <The original Vendor Model string>\n"
            f"- Best Match: <The single most accurate Master Model string from the Candidates list>\n"
            f"- Confidence: <An integer score from 0-100 representing your certainty in the match's correctness for part fitment>\n"
        )

        if alt_count > 0:
            prompt += f"- Alternatives (list up to {alt_count}):\n"
            for i in range(1, alt_count + 1):
                prompt += f"  {i}. <alternative model string>\n"

        if include_reason:
            prompt += (
                "- Reason: <An explanation justifying the Best Match. Focus on model generation, trim level, body style, "
                "or other compatibility factors relevant to part fitment. Constrain explanation to 500 characters.>"
            )

        body = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a vehicle model fitment expert."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 200
        }

        batch_rows.append({
            "custom_id": car_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body
        })

        id_to_prompt[car_id] = body
        id_to_meta[car_id] = (year, make)

    return batch_rows, id_to_prompt, id_to_meta

# === RESPONSE PARSING ===
def parse_response(content, car_id, year, make, valid_ids, COST_SAVER_MODE, COST_SAVER_CONFIG):
    """
    Parses an OpenAI GPT response and returns a structured dictionary.
    Applies config settings for alt match threshold, alt count, and reason inclusion.

    Args:
        content (str): The GPT text response.
        car_id (str): The original vendor car ID.
        year (str): The vehicle year.
        make (str): The vehicle make.
        valid_ids (set): A set of valid Year_Make_Model strings.
        cost_saver_mode (bool): Whether cost saver mode is enabled.
        cost_saver_config (dict): Cost saver mode configuration settings.

    Returns:
        dict: A dictionary containing best match, confidence, alternatives, and optional reason.
    """
    best = re.search(r"Best Match[::]?\s*(.+)", content, re.IGNORECASE)
    conf = re.search(r"Confidence[::]?\s*(\d+)", content, re.IGNORECASE)
    reason = re.search(r"Reason[::]?\s*(.+)", content, re.IGNORECASE)
    alts = re.findall(r"[\n\r]\s*\d[.)\-:]\s*(.+)", content)

    best_val = best.group(1).strip() if best else ""
    conf_val = int(conf.group(1)) if conf else 0
    reason_val = reason.group(1).strip() if reason else "No reason parsed"

    full_best = f"{year}_{make}_{best_val}" if best_val else ""
    full_alts = [f"{year}_{make}_{alt.strip()}" for alt in alts]
    valid_alts = [alt for alt in full_alts if alt in valid_ids]

    if full_best not in valid_ids:
        full_best = ""
        print(f"⚠️ No valid Best Match for {car_id}\nResponse:\n{content}")

    if full_best not in valid_ids:
        print(f"⚠️ INVALID BEST MATCH: {full_best} not in valid_ids")
        print(f"🔍 Raw GPT content:\n{content}\n")
        full_best = ""  

    if COST_SAVER_MODE:
        alt_thresh = COST_SAVER_CONFIG.get("alt_match_threshold", 80)
        alt_count = COST_SAVER_CONFIG.get("alt_match_count", 3)
        include_reason = COST_SAVER_CONFIG.get("include_reason", True)
    else:
        alt_thresh = 100 
        alt_count = 3
        include_reason = True

    include_alts = (not COST_SAVER_MODE) or (conf_val < alt_thresh)

    result = {
        "Vendor_Car_ID": car_id,
        "Best_Match_Car_ID": full_best,
        "Match_Confidence": conf_val,
        "Reason": reason_val if include_reason else ""
    }

    for i in range(1, 4):
        key = f"Alt_{i}"
        result[key] = valid_alts[i - 1] if include_alts and i <= alt_count and i <= len(valid_alts) else ""

    return result

# === BATCH JOB HANDLING ===
def save_batch_input(batch_rows, batch_input_path):
    """
    Writes the prepared batch prompts to a JSONL file for OpenAI batch submission.

    Args:
        batch_rows (list): List of dicts formatted for OpenAI batch API.
        batch_input_path (str): Output path for the .jsonl input file.
    """
    with open(batch_input_path, "w", encoding="utf-8") as f:
        for row in batch_rows:
            f.write(json.dumps(row) + "\n")

# === BATCH JOB SUBMISSION ===
def submit_batch(client, batch_input_path):
    """
    Uploads the batch input file to OpenAI and starts the batch job. Polls until completion.

    Args:
        client (OpenAI): Authenticated OpenAI API client.
        batch_input_path (str): Path to .jsonl input file for batch job.

    Returns:
        str: Output file ID to later retrieve results.
    """
    upload = client.files.create(file=open(batch_input_path, "rb"), purpose="batch")
    batch = client.batches.create(input_file_id=upload.id, endpoint="/v1/chat/completions", completion_window="24h")
    print(f"🚀 Submitted batch: {batch.id}")
    while True:
        status = client.batches.retrieve(batch.id)
        if status.status in ["completed", "failed", "expired"]:
            break
        print(f"...batch status: {status.status}")
        time.sleep(10)
    if status.status != "completed":
        raise RuntimeError(f"Batch failed: {status.status}")
    return status.output_file_id

# === BATCH OUTPUT HANDLING ===
def download_batch_output(client, output_file_id, output_path):
    """
    Downloads the OpenAI batch results and writes them to a local file.

    Args:
        client (OpenAI): OpenAI API client.
        output_file_id (str): File ID returned by the batch job.
        output_path (str): Where to save the batch result file.

    Returns:
        str: Raw text content of results if successful, else empty string.
    """
    try:
        result_text = client.files.with_raw_response.retrieve_content(output_file_id).text
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result_text)
        return result_text
    except Exception as e:
        print(f"❌ Error downloading batch output: {e}")
        return ""

# === BATCH OUTPUT PARSING ===
def parse_batch_output(output_path, id_to_meta, id_to_prompt, client, valid_ids, COST_SAVER_MODE, COST_SAVER_CONFIG):
    """
    Parses the GPT responses from the batch output file and retries failed rows if needed.

    Args:
        output_path (str): File path to saved GPT batch response output.
        id_to_meta (dict): Mapping of car_id to (year, make).
        id_to_prompt (dict): Original request prompts keyed by car_id.
        client (OpenAI): OpenAI API client.
        valid_ids (set): Set of valid master car IDs.

    Returns:
        list: List of result dictionaries including successful and retried matches.
    """
    results, failed_ids = [], []
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            car_id = obj.get("custom_id", "")
            body = obj.get("response", {}).get("body", {})
            content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
            year, make = id_to_meta[car_id]
            parsed = parse_response(content, car_id, year, make, valid_ids, COST_SAVER_MODE, COST_SAVER_CONFIG)
            results.append({**parsed, "Match_Method": "GPT Batch"})
            if not parsed["Best_Match_Car_ID"]:
                failed_ids.append(car_id)
    for car_id in failed_ids:
        try:
            year, make = id_to_meta[car_id]
            response = client.chat.completions.create(**id_to_prompt[car_id])
            content = response.choices[0].message.content
            parsed = parse_response(content, car_id, year, make, valid_ids, COST_SAVER_MODE, COST_SAVER_CONFIG)
            if not parsed["Best_Match_Car_ID"] and parsed["Match_Confidence"] == 0:
                parsed["Match_Method"] = "GPT Retry"
                results.append(parsed)
            else:
                print(f"⏭️ Skipped retry for {car_id}")
        except Exception as e:
            print(f"❌ Retry failed for {car_id}: {e}")
    return results

# === EXACT MATCH UNION ===
def append_exact_matches(results, output_dir):
    """
    Appends exact matches from a pre-labeled file and combines them with GPT batch results.

    Args:
        results (list): List of result dicts from GPT batch output.
        output_dir (str): Input directory containing exact_match_list.csv.

    Returns:
        pd.DataFrame: Combined DataFrame of batch and exact match results.
    """
    final_df = pd.DataFrame(results)
    exact_match_path = os.path.join(output_dir, "exact_match_list.csv")
    if os.path.exists(exact_match_path):
        exact_df = pd.read_csv(exact_match_path)
        columns = ["Vendor_Car_ID", "Best_Match_Car_ID", "Match_Confidence", "Alt_1", "Alt_2", "Alt_3", "Reason", "Match_Method"]
        for col in columns:
            if col not in exact_df.columns:
                exact_df[col] = ""
            if col not in final_df.columns:
                final_df[col] = ""
        exact_df["Match_Method"] = "Exact Match"
        exact_df = exact_df[columns]
        final_df = final_df[columns]
        final_df = pd.concat([final_df, exact_df], ignore_index=True)
        print(f"✅ Appended exact matches from: {exact_match_path}")
    else:
        print(f"⚠️ Exact match file not found: {exact_match_path}")
    return final_df

# === FINAL OUTPUT ===
def save_final_output(final_df, csv_output_path):
    """
    Saves the combined GPT and exact match results to a final CSV file.

    Args:
        final_df (pd.DataFrame): Final DataFrame containing all match results.
        csv_output_path (str): Output file path for CSV export.
    """
    final_df.to_csv(csv_output_path, index=False)

# === SETUP MODE ===
def setup_mode():
    if COST_SAVER_MODE:
        config = COST_SAVER_CONFIG
        print("⚡ Cost Saver Mode ENABLED:")
        print(f"- Model: {config['model']}")
        print(f"- Alternatives shown only if confidence < {config['alt_match_threshold']}")
        print(f"- Retries enabled: {config['retry_enabled']}")
        print(f"- Include reason for match: {config['include_reason']}")
        print(f"- Max allowed token cost: ${config['max_token_budget_dollars']:.2f}")
        return (
            config["model"],
            config["retry_enabled"],
            config["alt_match_threshold"],
            config["include_reason"],
            config["max_token_budget_dollars"]
        )
    else:
        print("🚀 Full Mode ENABLED:")
        print(f"- Model: {MODEL_FULL}")
        print("- All features enabled")
        return MODEL_FULL, True, 100, True, float("inf")

# === MAIN FUNCTION === 
def main():
    """
    Orchestrates the full matching pipeline: loading data, building prompts,
    estimating costs, submitting to GPT, handling retries, and saving results.
    """
    start_time = time.time()
    test_limit = TEST_ROW_LIMIT
    try:
        # === Setup Mode & Client ===
        model_name, retry_enabled, _, _, max_budget = setup_mode()
        client = configure()

        # === Load and Prepare Data ===
        vendor_df, master_df = load_data(OUTPUT_DIR, test_limit)
        master_lookup, valid_ids = build_master_lookup(vendor_df, master_df)

        batch_rows, id_to_prompt, id_to_meta = build_prompts(
            vendor_df, master_lookup, valid_ids, model_name, COST_SAVER_MODE, COST_SAVER_CONFIG
        )

        if not batch_rows:
            print("🚫 No prompts were generated. Exiting.")
            return

        batch_input_path = os.path.join(RESULTS_DIR, BATCH_INPUT)
        save_batch_input(batch_rows, batch_input_path)

        # === Estimate Token Usage & Cost ===
        total_tokens, avg_tokens, input_cost, output_cost, total_cost = estimate_token_costs(
            batch_rows,
            model_name,
            MODEL_COSTS[model_name],
            COST_SAVER_MODE,
            COST_SAVER_CONFIG
        )
        batch_rows, model_name, MODEL_COSTS[model_name]

        print(f"📏 Estimated total prompt tokens: {total_tokens}")
        print(f"📏 Average tokens per prompt: {avg_tokens:.2f}")
        print(f"💰 Input token cost:  ${input_cost:.2f}")
        print(f"💰 Output token cost: ${output_cost:.2f}")
        print(f"💰 Total estimated cost: ${total_cost:.2f}")

        if total_cost > max_budget:
            print(f"⚠️ Estimated cost (${total_cost:.2f}) exceeds max token budget (${max_budget:.2f}) — exiting.")
            return

        # === Run Batch Job ===
        output_file_id = submit_batch(client, batch_input_path)
        batch_output_path = os.path.join(RESULTS_DIR, BATCH_OUTPUT)
        download_batch_output(client, output_file_id, batch_output_path)

        # === Parse and Retry ===
        results = parse_batch_output(
            batch_output_path, id_to_meta, id_to_prompt, client, valid_ids,
            COST_SAVER_MODE, COST_SAVER_CONFIG
        )

        if not retry_enabled:
            results = [r for r in results if r["Best_Match_Car_ID"]]

        print(f"✅ Number of results parsed: {len(results)}")
        if not results:
            print("⚠️ No results returned — check GPT responses and model config.")
            return

        # === Combine with Exact Matches and Save ===
        final_df = append_exact_matches(results, OUTPUT_DIR)
        final_output_path = os.path.join(RESULTS_DIR, MATCH_FILE)
        save_final_output(final_df, final_output_path)

        print(f"📊 Final results saved to: {final_output_path}")
        print(f"⏱ Total runtime: {time.time() - start_time:.2f} seconds")

    except Exception as e:
        print(f"❌ Fatal error in main(): {e}")

if __name__ == "__main__":
    main()
