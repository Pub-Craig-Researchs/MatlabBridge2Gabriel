# GabrielBridge: Universal MATLAB Interface for GABRIEL

**GabrielBridge** is a powerful and flexible MATLAB interface for the [GABRIEL](https://github.com/openai/GABRIEL) (Generalizing Analytical Benchmarks for Robust Inference from Extensive LLMs) toolkit. It acts as a bridge between MATLAB and the Python-based GABRIEL library, enabling social scientists and researchers to leverage state-of-the-art LLMs for qualitative data analysis directly within their MATLAB workflow.

This repository features an **enhanced version of GABRIEL** that is fully compatible with **DeepSeek API** and other OpenAI-compatible endpoints.

---

## 🌟 Key Features

- **Unified MATLAB Interface**: Use a single function `GabrielBridge.m` to access all GABRIEL functionalities.
- **DeepSeek & Custom API Support**: Seamlessly switch between OpenAI, DeepSeek, or other local/custom endpoints.
- **Intelligent Parameter Mapping**: Automatically maps MATLAB inputs (structs, labels, instructions) to task-specific requirements.
- **Support for 15+ Tasks**:
  - `rate`: Numeric scoring (0-100) on multiple attributes.
  - `classify`: Text classification into user-defined labels.
  - `extract`: Structured data extraction from natural language.
  - `discover`: Automatic feature discovery and contrastive analysis.
  - `paraphrase`, `rank`, `deidentify`, `ideate`, and more.
- **High Concurrency**: Built-in support for parallel API calls to process large datasets quickly.

---

## 📦 Installation

### 1. Requirements

- MATLAB (R2021a or later recommended).
- Python 3.7+ installed and configured in MATLAB.
- Python dependencies: `pip install -r requirements.txt`.

### 2. Setup

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/GABRIEL-MATLAB.git
   cd GABRIEL-MATLAB
   ```

2. Configure your Python environment in MATLAB:

   ```matlab
   pe = pyenv;
   if pe.Status == "NotLoaded"
       pyenv('Version', 'path_to_your_python_executable');
   end
   ```

3. Add the project directory to your MATLAB path.

---

## ⚙️ Configuration

Edit `api_config.json` to add your API keys:

```json
{
    "profiles": {
        "deepseek": {
            "API_KEY": "your_deepseek_key_here",
            "BASE_URL": "https://api.deepseek.com",
            "MODEL": "deepseek-chat",
            "N_PARALLELS": 10
        }
    },
    "active_profile": "deepseek"
}
```

---

## 🚀 Quick Usage

### 📊 Metric Rating

```matlab
attrs = struct('Safety', 'How safe is this action? (0-100)');
data = {'Walking across the street', 'Jumping off a cliff'};
results = GabrielBridge(data, attrs, 'Task', 'rate');
```

### 🏷️ Topic Classification

```matlab
labels = struct('Finance', 'Related to money/markets', 'Tech', 'Related to IT/Software');
results = GabrielBridge({'The market is bullish'}, labels, 'Task', 'classify');
```

### 🔍 Data Extraction

```matlab
fields = struct('Name', 'Full Name', 'Age', 'Years old');
results = GabrielBridge({'John Doe is a 30-year-old engineer'}, fields, 'Task', 'extract');
```

---

## 🛡️ License & Acknowledgments

This project is licensed under the **GPL-3.0 License**.

It includes a modified version of the **GABRIEL** library (located in `src/`), which was originally developed by Hemanth Asirvatham and Elliott Mokski and released under the **Apache License 2.0**.

- Please refer to the `NOTICE` file for original contributions.
- See the `LICENSE` file for the full GPL-3.0 text.

---

## 📬 Contact

If you have questions or suggestions, please open an issue or reach out to the repository maintainer.
