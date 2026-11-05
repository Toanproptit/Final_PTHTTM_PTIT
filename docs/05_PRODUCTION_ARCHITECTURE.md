# Kiến trúc triển khai v2

## Thành phần

| Thành phần | Vai trò | Dữ liệu bền vững |
|---|---|---|
| React/Vite | Form hồ sơ, chat, knowledge, workflow trace, kết quả | Bundle tĩnh do FastAPI phục vụ |
| FastAPI | Validation API, điều phối agent, health và artifact API | Không giữ state trong RAM làm nguồn chính |
| LangGraph | State graph cho Validation, Safety, Prediction, Knowledge và Explanation | Routing decision và trace lưu trong audit DB |
| LangChain | `StructuredTool`, message protocol và vòng lặp gọi tool | Tool call/message lưu trong audit DB |
| MySQL 8.4 | System of record cho dữ liệu có cấu trúc và audit | Docker volume `mysql_data` |
| Neo4j 5.26 | Knowledge graph, provenance và vector retrieval | `neo4j_data`, `neo4j_logs` |
| OpenRouter | LLM tự chọn tool qua `ChatOpenRouter` | API key chỉ ở biến môi trường/backend |

## Luồng khởi động

`docker compose up --build -d` chờ MySQL và Neo4j healthy, sau đó backend tự chạy:

1. `alembic upgrade head` tạo/cập nhật schema MySQL.
2. `migrate_sqlite_to_mysql.py --if-present` kiểm tra lịch sử SQLite cũ; chỉ khi có bản ghi chưa migrate mới tạo bản sao timestamp rồi sao chép.
3. `import_dataset_mysql.py` kiểm tra hash, validate, deduplicate và nhập dataset theo batch; lần chạy sau trả về `already_imported`.
4. `import_knowledge_graph.py` tạo constraint, vector index, document/chunk/entity/fact và liên kết `SUPPORTED_BY`; lần chạy sau bỏ qua khi graph đã có fact.
5. Uvicorn mở cổng 8000.

## Điều phối và safety

`AgentState` chứa intent, validation, safety, prediction, knowledge, citation, routing decision, error và trace. LangGraph có các conditional edge để bỏ qua inference ngoài miền dữ liệu và chỉ gọi GraphRAG khi safety/nhãn yêu cầu bằng chứng. `AGENT_MAX_STEPS` chặn vòng lặp quá dài. Mọi tool LLM là LangChain `StructuredTool` trong allowlist; lỗi tool không được biến thành kết quả giả.

SafetyAgent độc lập với score của mô hình. Các quy tắc HbA1c/glucose chỉ phân luồng cảnh báo và luôn kèm điều kiện/nguồn. Hệ thống không chẩn đoán, kê đơn hoặc thay thế đánh giá của nhân viên y tế.

## Kiểm tra vận hành

```powershell
docker compose ps
docker compose logs -f backend
curl.exe http://127.0.0.1:8000/api/health
curl.exe http://127.0.0.1:8000/api/knowledge/graph
```

`/api/health` báo riêng database backend, Neo4j, prediction artifacts và chế độ orchestrator. `/api/knowledge/graph` chỉ trả một subgraph nhỏ có citation; endpoint trả 503 rõ ràng nếu Neo4j chưa sẵn sàng.

## Backup và restore

Tạo thư mục backup trước, rồi chạy từ root dự án:

```powershell
New-Item -ItemType Directory -Force runtime\backups
docker compose exec -T mysql mysqldump -uroot -p"$env:MYSQL_ROOT_PASSWORD" diabetes_agent | Set-Content -Encoding utf8 runtime\backups\mysql.sql
docker compose exec -T neo4j cypher-shell -u neo4j -p "$env:NEO4J_PASSWORD" "CALL apoc.export.cypher.all(null,{stream:true}) YIELD cypherStatements RETURN cypherStatements" | Set-Content -Encoding utf8 runtime\backups\neo4j.cypher
```

Neo4j Community image không bảo đảm có APOC export trong mọi cấu hình; phương án đáng tin cậy cho demo là dừng stack và sao lưu Docker volume. Restore là thao tác ghi đè: chỉ thực hiện sau khi kiểm tra file backup và chủ động xác nhận database đích.

MySQL restore từ file SQL đã kiểm tra:

```powershell
Get-Content -Raw runtime\backups\mysql.sql | docker compose exec -T mysql mysql -uroot -p"$env:MYSQL_ROOT_PASSWORD" diabetes_agent
```

## Phát triển không Docker

Nếu `DATABASE_URL` không được đặt, app dùng `runtime/demo.sqlite3` để tương thích local/test. Đây không phải cấu hình production. Neo4j thiếu credentials sẽ tắt GraphRAG có chủ đích và dùng TF-IDF fallback; trạng thái này xuất hiện trong health và retrieval trace.
