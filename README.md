[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/0zf8GytR)
# CS 264 HW - ReAct Agent for Software Engineering

This project implements a ReAct (Reasoning and Acting) agent for software engineering tasks using large language models. The codebase is adapted from [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent/tree/main). 

## Installation

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   ```
   export OPENAI_API_KEY=your_api_key_here
   ```

## Implement the ReAct Agent
We provide a skeleton code for you to implement your ReAct agent. Refer to [CODE.md](./CODE.md) for more details about the structure of the code and what you should implement.


## Running the Code

To run the ReAct agent on SWE-bench instances:
```bash
python run_agent.py --model gpt-5-mini --max-steps 100 --output results
```

The agent will process SWE-bench instances and save results to the `results/` directory.

**Note**: We suggest testing the agent on a single instance first by setting `instances = instances[:1]` in run_agent.py.


## Evaluation
### Running SWEBench's Evaluation Harness

After generating predictions, run SWEBench's evaluation harness to evaluate the submissions:

```bash
python -m swebench.harness.run_evaluation \
    --dataset_name lynnliu030/swebench-eval-subset \
    --predictions_path ./results/preds.json \
    --max_workers 8 \
    --run_id my_evaluation_run
```

## 📋 Evaluation Results Format

The evaluation will generate a results file with the following structure:

```json
{
    "total_instances": 20,
    "submitted_instances": 20,
    "completed_instances": 19,
    "resolved_instances": 9,
    "unresolved_instances": 10,
    "empty_patch_instances": 1,
    "error_instances": 0,
    "completed_ids": ["astropy__astropy-7166", ...],
    "resolved_ids": ["astropy__astropy-7166", ...],
    "unresolved_ids": ["django__django-10973", ...],
    "schema_version": 2
}
```

## 📤 Submission

### 1. Report (PDF) (as `your_repo/report.pdf`)
Your report should:
- Report your accuracy number. What is your observation?
- Describe the custom tools you created and explain the reason behind making them

### 2. Final Evaluation Results (as `your_repo/final_results.json`)
The evaluation result file with the format shown above, containing your agent's performance metrics on the SWE-Bench subset.

---

*Good luck optimizing your agent!* 🤖
