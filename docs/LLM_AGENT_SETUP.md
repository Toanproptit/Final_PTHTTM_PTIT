# Chạy demo LLM Agent với OpenRouter

Tab **LLM Agent** dùng LangChain `ChatOpenRouter` để model chọn giữa ba `StructuredTool`: tìm trong kho kiến thức tiểu đường, chạy state graph dự đoán LangGraph trên hồ sơ giả định, và đọc artifact thực nghiệm. Backend chỉ thực thi tool đã cho phép và trả trace để xem trong giao diện.

## Cấu hình

1. Sao chép `.env.example` thành `.env` ở thư mục gốc dự án.
2. Điền API key của bạn vào `OPENROUTER_API_KEY`. Không gửi key trong chat, không đưa vào JavaScript, và không commit file `.env`.
3. Chọn model có hỗ trợ tool calling trong OpenRouter và ghi đúng mã model vào `OPENROUTER_MODEL` (ví dụ `openai/gpt-4.1-mini`).
4. Khởi động lại server để nạp cấu hình:

```powershell
Copy-Item .env.example .env
# Sửa .env bằng editor để thay key
.\.venv\Scripts\python.exe -m uvicorn diabetes_demo.api:app --host 127.0.0.1 --port 8000
```

Có thể đặt biến môi trường trong terminal thay cho `.env`:

```powershell
$env:OPENROUTER_API_KEY = "your-key"
$env:OPENROUTER_MODEL = "openai/gpt-4.1-mini"
.\.venv\Scripts\python.exe -m uvicorn diabetes_demo.api:app --host 127.0.0.1 --port 8000
```

Mở `http://127.0.0.1:8000`, vào tab **LLM Agent**. Giao diện hiển thị model và các tool đang bật; mỗi câu trả lời có thể mở phần **Tool calls** để xem tool, tham số và kết quả tóm tắt.

## Giới hạn

- LangChain chuẩn hóa message/tool call, còn model chỉ được chọn các `StructuredTool` nằm trong allowlist; model không được chạy shell, tùy ý truy cập file, hoặc gọi hàm bất kỳ.
- Mỗi yêu cầu có tối đa 5 lượt gọi tool. API key chỉ được đọc ở backend từ biến môi trường hoặc `.env`.
- Tool dự đoán dùng hồ sơ giả định và nhãn dataset Kaggle. Điểm số không phải chẩn đoán hoặc nguy cơ mắc bệnh trong tương lai.
- Các request tới OpenRouter có thể phát sinh chi phí theo model và tài khoản của bạn.
