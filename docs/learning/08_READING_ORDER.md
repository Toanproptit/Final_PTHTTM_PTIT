# 08. Lộ trình đọc từng file từ dễ đến khó

Đọc theo thứ tự sau. Với mỗi module hãy tự trả lời “input là gì, output là gì, file nào gọi nó?” trước khi học chi tiết cú pháp.

| Thứ tự/file | Class/hàm chính và biến cần hiểu | Module liên quan | Ví dụ tự lần theo |
|---|---|---|---|
| 1. `schema.py` | PatientInput, QuestionInput, FEATURES, CATEGORY_VALUES, DEMO_PATIENT | Pydantic; API/data/agent cùng sử dụng | Vì sao thiếu bmi lỗi, còn smoking=No Info hợp lệ? |
| 2. `paths.py` | ROOT, RAW, ML, RL, KB, RUNTIME | pathlib; toàn project | RAW trỏ CSV Kaggle nào? |
| 3. `model.py` | DiabetesMLP.__init__, forward, network | torch.nn; train và Prediction | Đầu vào (1,15) ra (1,), đâu là logit? |
| 4. `data.py` | download_dataset, encode_features, load_dataset, split_dataset, make_preprocessor | pandas/sklearn/schema/paths | Vì sao 100000 còn 96146? Vì sao hash không chứa y? |
| 5. `train.py` | run_training, classification_metrics, create_plots; splits/arrays/targets/best_state/stale | data/model/torch/sklearn | Đi từ batch loss.backward đến checkpoint epoch tốt nhất |
| 6. `agents.py` | RuleBasedAgent.run; PredictionAgent.ready/load/run; ExplanationAgent.run | schema/data/model/joblib/torch | Nhập tuổi100 làm prediction bị bỏ qua ở đâu? |
| 7. `knowledge.py` | normalize, chunk_markdown, KnowledgeAgent.__init__/run; matrix/vectorizer/chunks/score | KB/sklearn; Orchestrator dùng | Từ HbA1c đến chunk testing:0 và link Vinmec |
| 8. `storage.py` | DemoStore.__init__/save/recent/clear; connection/path | sqlite3/contextlib | Vì sao `closing` cần thiết trên Windows? |
| 9. `orchestrator.py` | OrchestratorAgent.call/predict/ask; trace/run_id/result | agents/knowledge/storage | Input Explanation lấy từ biến nào của ba bước trước? |
| 10. `api.py` | create_app/lifespan; predict/ask/history/experiments/artifact | FastAPI/orchestrator/schema | POST không có model trả503, thiếu trường trả422 |
| 11. `web/app.js` | api/activate/buildForm/evidenceHTML/loadHistory/loadExperiments | API endpoints; HTML IDs | Vì sao JS không tính sigmoid hoặc Q-table báo cáo? |
| 12. `legacy/agent02_experiment.py` | Config, env, QLearningAgent, RuleBasedAgent, train/evaluate/4 experiments | numpy/matplotlib | Theo một step và tính terminal target bằng tay |
| 13. `rl_experiments.py` | run_experiments; source hash/manifest | Import trực tiếp legacy | Chứng minh file .tex được dùng bằng hash và import |
| 14. `evaluate.py` | evaluate_retrieval/evaluate_workflow; expected_document/checks | knowledge/orchestrator/temp SQLite | Câu nào top1 sai? Vì sao workflow dùng hồ sơ giả định? |
| 15. `tests/` | Kiểm tra leakage, frozen Q, nguồn, API, SQLite | Các module đã học | Cố ý dùng string bool và đọc lỗi422 |
| 16. `scripts/export_report.py` | Đọc artifact, sinh bảng Markdown | json/csv/statistics/XML | Đổi model thì bảng báo cáo lấy số mới từ đâu? |

## Chỗ thuật toán nên học kỹ

Trong `load_dataset`, bỏ trùng dựa trên đúng đầu vào được học; trong `split_dataset`, index inner là index cục bộ của remaining nên phải ánh xạ lại `remaining[train_local]`. Trong train, `copy.deepcopy(model.state_dict())` bảo toàn checkpoint tốt, nếu chỉ giữ tham chiếu có thể bị cập nhật tiếp. Trong Knowledge, query và documents dùng cùng vocabulary. Trong storage, `with connection` quản lý commit/rollback nhưng không tự đóng sqlite connection; `closing` giải quyết phần đóng.

## Cách học thực hành

Buổi 1: schema/data và thử đọc `splits.csv`; tự kiểm tra không giao nhóm. Buổi 2: MLP/loss và xem loss PNG, confusion matrix. Buổi 3: chạy một request, đọc trace và code điều phối. Buổi 4: QA và Q-learning; phân biệt supervised label với environment reward. Buổi 5: trình bày kịch bản demo, trả lời các câu trong DEFENSE_QA.

Không học thuộc giá trị score của một hồ sơ. Hãy hiểu đường tính và nguồn artifact; khi đổi seed/cấu hình hoặc dữ liệu, số liệu có thể đổi và cần chạy lại báo cáo. Các đoạn import và chi tiết CSS không quan trọng bằng hợp đồng, thuật toán, chống leakage và giới hạn nghiên cứu.
