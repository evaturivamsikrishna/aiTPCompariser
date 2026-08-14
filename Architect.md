Act as a Senior AI Systems Engineer and Lead QA Automation Architect. Build a production-ready, end-to-end Python pipeline using LangGraph, Instructor, Pydantic, and Pandas that accepts ONLY raw CSV files as input to benchmark multiple LLMs on test plan generation.

---

### 📥 Input Constraint
The application must operate strictly on CSV file uploads (e.g., `requirements.csv` or `test_cases.csv`). It must not rely on interactive CLI text prompts or manual string inputs. The system must automatically parse, clean, and map CSV columns regardless of exact header naming variations.

---

### 🛠 Tech Stack Requirements
* **Data Ingestion:** `pandas`
* **Orchestration:** `langgraph` (StateGraph)
* **Schema Enforcement & Parsing:** `instructor` + `pydantic`
* **LLM Clients:** `langchain-openai`, `langchain-anthropic`, `langchain-google-genai` (or native SDKs)
* **Reporting:** Markdown file export + Rich/Tabulate CLI formatting

---

### 📁 Required Architecture & Code Structure

Provide fully functional, modular code across the following files:

#### 1. `csv_parser.py` (Robust CSV Ingestion Engine)
* Build an automated CSV parser that:
  * Accepts any `.csv` file path.
  * Uses fuzzy/semantic matching or LLM-assisted column mapping to auto-detect key fields (e.g., mapping headers like `Requirement_ID`, `req_id`, `ID` to a unified ID field; and `Description`, `User Story`, `Spec`, `Requirement` to a unified Description field).
  * Converts the CSV rows into a clean, structured list of requirement dictionaries.

#### 2. `schemas.py`
Define exact Pydantic models for:
* `CSVRequirement`: `id` (str), `description` (str), `metadata` (dict for extra CSV columns).
* `TestCase`: `id`, `title`, `preconditions`, `steps`, `expected_result`, `edge_cases_addressed`.
* `GeneratedTestPlan`: `model_name`, `test_strategy`, `requirements_covered`, `test_cases`.
* `MetricScore`: `score` (int 1–5), `reasoning` (str).
* `ModelEvaluation`: `model_name`, `requirement_coverage`, `edge_case_quality`, `actionability`, `redundancy`, `scope_drift`, `overall_score`.
* `BenchmarkState`: TypedDict state for the LangGraph state machine.

#### 3. `generators.py`
* Implement async functions that take the parsed `CSVRequirement` list and concurrently pass it as context to Model A (Claude 3.5 Sonnet), Model B (GPT-4o), and Model C (Gemini 1.5 Pro).
* Include fallback error handling for API timeouts or failures.

#### 4. `evaluator.py`
* Build a **Single Judge Engine** using a fixed-temperature LLM (`temperature=0.0`).
* Grade candidate test plans derived from the CSV requirements across 5 dimensions:
  1. *Requirement Coverage* (1–5)
  2. *Edge Case Quality* (1–5)
  3. *Actionability* (1–5)
  4. *Redundancy* (1–5, inverse penalty)
  5. *Scope Drift* (1–5, inverse penalty)
* Calculate a weighted overall score:
  $$\text{Overall} = (0.3 \times \text{Coverage}) + (0.25 \times \text{Edge Cases}) + (0.25 \times \text{Actionability}) + (0.1 \times \text{Redundancy}) + (0.1 \times \text{Scope Drift})$$

#### 5. `graph.py`
* Construct the **LangGraph StateGraph**:
  * Node 1: `ingest_csv` (Parses and normalizes CSV input)
  * Node 2: `fanout_generators` (Runs AI models concurrently using CSV requirements)
  * Node 3: `normalize_schemas` (Ensures output conforms to JSON schema)
  * Node 4: `evaluate_candidates` (Single Judge evaluation)
  * Node 5: `export_results` (Saves comparative report to `benchmark_report.md` and `results.csv`)

#### 6. `main.py`
* A clean entry point that executes the pipeline by passing a path to a target CSV file:
  `python main.py --csv path/to/requirements.csv`

---

### ⚠️ Execution Constraints
* Provide complete, runnable code—no placeholders, missing functions, or `// TODO` comments.
* Include explicit handling for messy CSV edge cases (empty rows, missing column headers, special characters).