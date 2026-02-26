import os
import json
import asyncio
import pandas as pd
import gabriel

def load_config(config_path="api_config.json", profile_name=None):
    """
    Load API configuration from a JSON file.
    Supports multiple profiles (e.g., openai, deepseek, local).
    """
    with open(config_path, "r", encoding="utf-8") as f:
        full_config = json.load(f)
        
    # If the config has a 'profiles' dict, use the profile logic
    if "profiles" in full_config:
        active = profile_name or full_config.get("active_profile")
        if not active or active not in full_config["profiles"]:
            # fallback to the first profile if not found
            active = list(full_config["profiles"].keys())[0]
        return full_config["profiles"][active]
    
    # Otherwise assume it's a flat dictionary (backwards compatible)
    return full_config

def run_gabriel_task(task_type, data_dict, attributes, save_dir, config_path="api_config.json", profile_name=None, **kwargs):
    """
    Generic wrapper to run GABRIEL tasks (rate, classify, extract, etc.) using any API.
    Designed to be easily callable from MATLAB using py.gabriel_wrapper.run_gabriel_task.

    Args:
        task_type (str): The type of task to run ('rate', 'classify', 'extract', etc.).
        data_dict (dict): A dictionary representation of the dataframe. E.g., {'text_column': ['text1', 'text2']}.
        attributes (dict): The attributes or labels dictionary for the task.
        save_dir (str): Directory to save the results.
        config_path (str): Path to the configuration JSON file.
        profile_name (str): The name of the API profile to use (e.g., 'deepseek', 'openai'). Pass None to use default.
        **kwargs: Additional arguments to pass to the GABRIEL task.

    Returns:
        dict: A dictionary representation of the resulting DataFrame.
    """
    # Load configuration
    config = load_config(config_path, profile_name)

    # Universally set API keys and base URL for the openai python package used underneath
    api_key = str(config.get("API_KEY", "sk-none"))
    base_url = str(config.get("BASE_URL", "https://api.openai.com/v1"))
    
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_BASE_URL"] = base_url
    
    # Debug print to MATLAB console
    print(f"[gabriel_wrapper] Using API_KEY: {api_key[:5]}...{api_key[-4:]}")
    print(f"[gabriel_wrapper] Using BASE_URL: {base_url}")

    # Model and concurrency settings
    model = config.get("MODEL", "gpt-4o")
    n_parallels = config.get("N_PARALLELS", 20)

    # Convert MATLAB struct/dict to Pandas DataFrame
    df = pd.DataFrame(data_dict)

    # Identify the primary text column (assuming it's the first key if not specified)
    column_name = kwargs.pop("column_name", list(data_dict.keys())[0]) if data_dict else None

    # HACK: Force refresh gabriel's internal client cache
    try:
        from gabriel.utils import openai_utils
        openai_utils._clients_async.clear()
        print("[gabriel_wrapper] Internal client cache cleared.")
    except Exception as e:
        print(f"[gabriel_wrapper] Warning: Could not clear client cache: {e}")

    # Mapping of task types to GABRIEL async functions
    task_map = {
        "rate": gabriel.rate,
        "classify": gabriel.classify,
        "extract": gabriel.extract,
        "deidentify": gabriel.deidentify,
        "rank": gabriel.rank,
        "codify": gabriel.codify,
        "paraphrase": gabriel.paraphrase,
        "compare": gabriel.compare,
        "discover": gabriel.discover,
        "bucket": gabriel.bucket,
        "seed": gabriel.seed,
        "ideate": gabriel.ideate,
        "debias": gabriel.debias,
        "filter": gabriel.filter,
        "whatever": gabriel.whatever,
    }

    if task_type not in task_map:
        raise ValueError(f"Unsupported task_type: {task_type}. Supported: {list(task_map.keys())}")

    task_func = task_map[task_type]

    # Helper function to run the async task
    async def _run_task():
        # Common parameters
        common_kwargs = {
            "save_dir": save_dir,
            "model": model,
            "n_parallels": n_parallels,
        }
        # Merge with user provided kwargs
        final_kwargs = {**common_kwargs, **kwargs}

        # Parameter Normalization: 
        # GABRIEL functions use different names for the same concept (attributes/labels/categories/instructions)
        if task_type in ("classify", "discover"):
            final_kwargs["labels"] = attributes
        elif task_type == "codify":
            final_kwargs["categories"] = attributes
        elif task_type in ("paraphrase", "seed", "deidentify"):
            # If attributes is a string, use it as instructions. 
            # If it's a dict with 'instructions' key, use that.
            if isinstance(attributes, str):
                final_kwargs["instructions"] = attributes
            elif isinstance(attributes, dict) and "instructions" in attributes:
                 final_kwargs["instructions"] = attributes["instructions"]
            else:
                final_kwargs["instructions"] = str(attributes)
        elif task_type in ("rate", "extract", "rank", "ideate"):
            final_kwargs["attributes"] = attributes

        # Specialized logic for tasks
        if task_type in ("seed", "ideate"):
            # These tasks take 'instructions' or 'topic' as the first positional arg
            first_arg = attributes
            if isinstance(attributes, dict) and len(attributes) == 1:
                first_arg = list(attributes.values())[0]
            return await task_func(first_arg, **final_kwargs)
        
        elif task_type in ("compare", "discover"):
            return await task_func(df=df, **final_kwargs)
        
        else:
            return await task_func(df=df, column_name=column_name, **final_kwargs)

    # Execute the async function
    result = asyncio.run(_run_task())

    # result might be a dict for 'discover', convert to serializable if needed
    if isinstance(result, dict):
        output = {}
        for k, v in result.items():
            if isinstance(v, pd.DataFrame):
                output[k] = v.where(pd.notnull(v), None).to_dict(orient="list")
            else:
                output[k] = v
        return output

    # Convert Result DataFrame back to Dictionary for MATLAB
    return result.where(pd.notnull(result), None).to_dict(orient="list")
