#!/usr/bin/env python3
import concurrent.futures
import subprocess
from pathlib import Path
import os
import traceback

import typer
from datasets import load_dataset

from utils import save_traj, update_preds_file, remove_from_preds_file, get_sb_environment

app = typer.Typer(rich_markup_mode="rich", add_completion=False)

DATASET_MAPPING = {
    "cs264": "lynnliu030/swebench-eval-subset",
}

from agent import ReactAgent
from llm import OpenAIModel
from response_parser import ResponseParser
from envs import SWEEnvironment, DumbEnvironment

def process_instance(
    instance: dict,
    output_dir: Path,
    model_name: str,
    max_steps: int,
) -> None:
    """Process a single SWEBench instance."""
    instance_id = instance["instance_id"]
    instance_dir = output_dir / instance_id
    
    # Avoid inconsistent state if something here fails and there's leftover previous files
    remove_from_preds_file(output_dir / "preds.json", instance_id)
    (instance_dir / f"{instance_id}.traj.json").unlink(missing_ok=True)
    
    # Initialize the model and parser
    llm = OpenAIModel(ResponseParser.END_CALL, model_name)
    parser = ResponseParser()
    task = instance["problem_statement"]
    
    print(f"Processing instance {instance_id}")
    agent = None    
    result = ""
    
    try:
        # Initialize the environment
        env = SWEEnvironment(instance)
        # Initialize the agent
        agent = ReactAgent("swe-agent", parser, llm)
        
        # Add environment functions to the agent
        # Core bash command tool (required)
        agent.add_functions([env.run_bash_cmd])

        # Custom tools for better SWE-bench performance
        agent.add_functions([
            env.show_file,        # View files with line numbers
            env.replace_in_file,  # Precise code editing
            env.search_code,      # Search for code patterns
            env.find_files,       # Find files by name pattern
            env.list_directory,   # List directory contents
            env.create_file,      # Create new files
            env.view_around_line, # View context around a specific line
        ])
        
        # Run the agent
        output = agent.run(task, max_steps) 
        
        # Generate patch for SWE-Bench
        result = env.generate_patch(output)
        
    except Exception as e:
        print(f"Error processing instance {instance_id}: {e}")
        print(traceback.format_exc())
        # Store error in result for debugging
        result = f"Error: {type(e).__name__}: {str(e)}"

    finally:
        # Save the trajectory and update the predictions file
        save_traj(
            agent,
            instance_dir / f"{instance_id}.traj.json",
            result=result,
            instance_id=instance_id,
        )
        update_preds_file(output_dir / "preds.json", instance_id, model_name, result)
        print(f"Completed instance {instance_id}, result: {result}")

@app.command(help="Run CS 264 HW on subset of SWEBench instances.")
def main(
    subset: str = typer.Option("cs264", "--subset", help="SWEBench subset used or path to a dataset", rich_help_panel="Data selection"),
    split: str = typer.Option("test", "--split", help="Dataset split", rich_help_panel="Data selection"),
    output: str = typer.Option("outputs", "-o", "--output", help="Output directory", rich_help_panel="Basic"),
    model_name: str = typer.Option("gpt-5-mini", "--model", help="Model used", rich_help_panel="Basic"),
    max_steps: int = typer.Option(100, "--max-steps", help="Maximum number of steps", rich_help_panel="Basic"),
    run_evaluation: bool = typer.Option(False, "--run-evaluation", help="Run SWEBench evaluation after generating predictions", rich_help_panel="Evaluation"),
    max_workers: int = typer.Option(8, "--eval-max-workers", help="Max workers for evaluation harness", rich_help_panel="Evaluation"),
) -> None:
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to {output_path}")

    dataset_path = DATASET_MAPPING.get(subset, subset)
    print(f"Loading dataset {dataset_path}, split {split}...")
    instances = list(load_dataset(dataset_path, split=split))
    # limit to 1 instance for testing
    # instances = instances[:1]
    print(f"Running on {len(instances)} instances...")

    def process_futures(futures: dict[concurrent.futures.Future, str]):
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except concurrent.futures.CancelledError:
                pass
            except Exception as e:
                instance_id = futures[future]
                print(f"Error in future for instance {instance_id}: {e}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = {
            executor.submit(process_instance, instance, output_path, model_name, max_steps): instance[
                "instance_id"
            ]
            for instance in instances
        }
        try:
            process_futures(futures)
        except KeyboardInterrupt:
            print("Cancelling all pending jobs. Press ^C again to exit immediately.")
            for future in futures:
                if not future.running() and not future.done():
                    future.cancel()
            process_futures(futures)
    
    # Run evaluation if requested
    if run_evaluation:
        print("\n" + "="*80)
        print("Running SWEBench evaluation harness...")
        print("="*80 + "\n")
        
        predictions_path = output_path / "preds.json"
        run_id = output_path.name  # Use output directory name as run_id
        
        # Construct the evaluation command
        eval_cmd = [
            "python", "-m", "swebench.harness.run_evaluation",
            "--dataset_name", dataset_path,
            "--predictions_path", str(predictions_path),
            "--max_workers", str(max_workers),
            "--run_id", run_id
        ]
        
        # Set DOCKER_HOST environment variable for rootless docker
        env = os.environ.copy()
        if "DOCKER_HOST" not in env:
            # Try to set rootless docker socket
            user_id = os.getuid()
            docker_socket = f"unix:///run/user/{user_id}/docker.sock"
            if os.path.exists(f"/run/user/{user_id}/docker.sock"):
                env["DOCKER_HOST"] = docker_socket
                print(f"Setting DOCKER_HOST={docker_socket}")
        
        print(f"Running command: {' '.join(eval_cmd)}")
        print()
        
        try:
            result = subprocess.run(eval_cmd, env=env, check=True)
            print("\n" + "="*80)
            print("Evaluation completed successfully!")
            print("="*80)
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] Evaluation failed with exit code {e.returncode}")
            print("You can run the evaluation manually with:")
            print(f"{' '.join(eval_cmd)}")
        except Exception as e:
            print(f"\n[ERROR] Failed to run evaluation: {e}")
            print("You can run the evaluation manually with:")
            print(f"{' '.join(eval_cmd)}")


if __name__ == "__main__":
    app()
