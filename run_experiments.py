"""Chạy lại các thực nghiệm bằng một lệnh sau khi cài requirements."""
import argparse
import sys
from diabetes_demo.data import download_dataset
from diabetes_demo.paths import RAW
from diabetes_demo.train import run_training
from diabetes_demo.rl_experiments import run_experiments
from diabetes_demo.evaluate import evaluate_retrieval, evaluate_workflow

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-rl", action="store_true", help="Giữ artifact RL hiện có để tiết kiệm thời gian")
    args = parser.parse_args()
    if not RAW.is_file():
        download_dataset()
    run_training()
    if not args.skip_rl:
        run_experiments()
    print("Retrieval evaluation:", evaluate_retrieval())
    print("Workflow evaluation:", evaluate_workflow())
