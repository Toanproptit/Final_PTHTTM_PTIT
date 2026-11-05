"""Sinh bảng kết quả báo cáo từ JSON/CSV/XML thực, không nhập tay metric."""
from pathlib import Path
import csv
import json
import statistics
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"

def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))

def main() -> None:
    ml = json.loads((ARTIFACTS / "ml" / "evaluation.json").read_text(encoding="utf-8"))
    meta = ml["metadata"]
    retrieval = json.loads((ARTIFACTS / "retrieval" / "evaluation.json").read_text(encoding="utf-8"))
    workflow = json.loads((ARTIFACTS / "workflow_evaluation.json").read_text(encoding="utf-8"))
    rl_manifest = json.loads((ARTIFACTS / "rl" / "manifest.json").read_text(encoding="utf-8"))
    lines = ["# 03. Kết quả sinh từ artifact thực", "",
             "File này được tạo bởi `python scripts/export_report.py`. Không chỉnh tay metric; thay cấu hình/dữ liệu thì chạy thực nghiệm và xuất lại.", "",
             "## Phân loại dataset Kaggle", "",
             f"Nguồn: {meta['dataset_ref']}. SHA-256 CSV: `{meta['dataset_sha256']}`.",
             f"Artifact UTC: {meta['generated_utc']}; seed {meta['seed']}.", "",
             f"{meta['dataset']['raw_rows']:,} dòng gốc → {meta['dataset']['unique_rows']:,} dòng theo 8 X + y; bỏ {meta['dataset']['exact_duplicates_removed']:,} dòng trùng; {meta['dataset']['conflicting_groups']} nhóm X có nhãn mâu thuẫn được giữ cùng tập.", "",
             "| Tập | Số dòng | Nhãn 1 | Nhãn 0 |", "|---|---:|---:|---:|"]
    for name, split in meta["splits"].items():
        lines.append(f"| {name} | {split['rows']} | {split['positive']} | {split['negative']} |")
    lines += ["", f"MLP `{meta['architecture']}`; best epoch {meta['best_epoch']}/{meta['epochs_run']}; batch {meta['batch_size']}; threshold {meta['threshold']}.", "",
              "| Mô hình | Accuracy | Precision | Recall | F1 | ROC-AUC | AP | Brier |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, result in ml["metrics"].items():
        values = [f"{100 * result[key]:.2f}%" for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]]
        lines.append(f"| {name} | " + " | ".join(values) + f" | {result['brier_score']:.5f} |")
    lines += ["", "Confusion matrix có thứ tự nhãn [0,1], hàng thật/cột dự đoán:", ""]
    for name, result in ml["metrics"].items():
        cm = result["confusion_matrix"]
        lines.append(f"- {name}: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}.")
    lines += ["", "Accuracy phải được đọc cùng Recall: majority baseline cho thấy ảnh hưởng lệch lớp. Đây là một split nhóm có seed cố định, chưa có external validation hoặc hiệu chuẩn. Không coi kết quả là độ chính xác y khoa; nguồn Kaggle chưa mô tả đủ provenance/label generation. Clinical notes, năm, địa điểm và race không nằm trong X.", "",
              "![Training loss](../../artifacts/ml/training_loss.png)",
              "![ROC](../../artifacts/ml/roc_curves.png)",
              "![Confusion matrices](../../artifacts/ml/confusion_matrices.png)", "",
              "Dữ liệu chi tiết: `artifacts/ml/model_comparison.csv`, `splits.csv`, `test_predictions.csv`, `training_history.csv`.", "",
              "## Q-learning so với Rule-based", "",
              f"Nguồn gốc Python được bảo toàn: hash trùng = **{rl_manifest['source_sha256'] == rl_manifest['original_sha256']}**. Train {rl_manifest['episodes']} episode × {rl_manifest['steps_per_episode']} lượt; evaluate {rl_manifest['evaluation_episodes']} episode, không khám phá/cập nhật Q.", "",
              "| Agent baseline | Reward/lượt đánh giá | Tỷ lệ giảm giá |", "|---|---:|---:|"]
    for row in read_csv(ARTIFACTS / "rl" / "baseline_evaluation.csv"):
        lines.append(f"| {row['agent']} | {float(row['mean_reward_per_step']):.4f} | {100*float(row['discount_action_rate']):.2f}% |")
    seeds = read_csv(ARTIFACTS / "rl" / "multi_seed_summary.csv")
    lines += ["", "| Agent, 5 seed | Mean reward/lượt | Sample SD |", "|---|---:|---:|"]
    for name in ["RL", "Rule-based"]:
        values = [float(row["mean_reward_per_step"]) for row in seeds if row["agent"] == name]
        lines.append(f"| {name} | {statistics.mean(values):.4f} | {statistics.stdev(values):.4f} |")
    lines += ["", "| Alpha | RL reward/lượt |", "|---|---:|"]
    for row in read_csv(ARTIFACTS / "rl" / "learning_rate_comparison.csv"):
        lines.append(f"| {row['alpha']} | {float(row['rl_mean_reward_per_step']):.4f} |")
    lines += ["", "| Exploration | Epsilon start/min/decay | RL reward/lượt |", "|---|---|---:|"]
    for row in read_csv(ARTIFACTS / "rl" / "exploration_comparison.csv"):
        lines.append(f"| {row['exploration_setting']} | {row['epsilon_start']} / {row['epsilon_min']} / {row['epsilon_decay']} | {float(row['rl_mean_reward_per_step']):.4f} |")
    lines += ["", "RL không vượt Rule một cách rõ ràng ở môi trường này; không có kiểm định chứng minh ưu thế. Alpha/epsilon chỉ so trên một training seed mỗi cấu hình. State quan sát gốc bỏ own_previous_action và time_remaining nên không khẳng định Markov đầy đủ/hội tụ tối ưu. Reward là lợi nhuận giả lập, không liên quan kết quả y tế.", "",
              "![RL training](../../artifacts/rl/episode_rewards.png)",
              "![Learning rate](../../artifacts/rl/learning_rate_comparison.png)",
              "![Exploration](../../artifacts/rl/exploration_comparison.png)", "",
              "## Đánh giá truy xuất tri thức", "",
              f"Chế độ: {retrieval['mode']}. {retrieval['supported_questions']} câu trong phạm vi, {retrieval['unsupported_questions']} câu ngoài phạm vi; threshold cosine={retrieval['minimum_score']}.", "",
              "| Chỉ số | Giá trị |", "|---|---:|",
              f"| Hit@1 | {100*retrieval['hit_at_1']:.2f}% |",
              f"| Hit@3 | {100*retrieval['hit_at_3']:.2f}% |",
              f"| Rejection accuracy | {100*retrieval['rejection_accuracy']:.2f}% |", "",
              retrieval["limitation"], "",
              "Kết quả từng câu kể cả câu sai: `artifacts/retrieval/question_results.csv`.", "",
              "## Workflow và kiểm thử", "",
              "| Kiểm tra workflow | Kết quả |", "|---|---|"]
    for check, passed in workflow["checks"].items():
        lines.append(f"| {check} | {'PASS' if passed else 'FAIL'} |")
    xml_path = ARTIFACTS / "test_results.xml"
    if xml_path.exists():
        suites = ET.parse(xml_path).getroot().iter("testsuite")
        counts = {key: 0 for key in ["tests", "failures", "errors", "skipped"]}
        for suite in suites:
            for key in counts:
                counts[key] += int(suite.attrib.get(key, 0))
        lines += ["", f"JUnit XML: {counts['tests']} tests; {counts['failures']} failures; {counts['errors']} errors; {counts['skipped']} skipped."]
    lines += ["", "Workflow dùng hồ sơ giả định để kiểm tra phần mềm, không đo độ chính xác y khoa. Unit/integration tests không thay thế kiểm tra giao diện trực quan. Xem `docs/04_VERIFICATION.md` cho phạm vi đã kiểm chứng.", "",
              "## Phiên bản môi trường model", "", *[f"- {name}: {version}" for name, version in meta["versions"].items()], ""]
    output = ROOT / "docs" / "report" / "03_RESULTS.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output)

if __name__ == "__main__":
    main()
