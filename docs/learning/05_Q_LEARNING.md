# 05. Q-learning và phần kế thừa AGENT02

## Code gốc thực sự được sử dụng ở đâu?

`legacy/agent02_experiment.py` là bản sao **nguyên byte** của `agent02_code.tex`. `diabetes_demo/rl_experiments.py` import các hàm từ file này để chạy. Test so hash hai file; `artifacts/rl/manifest.json` lưu hash, seed và cấu hình. Demo thương mại `index.html` là bản JavaScript minh họa riêng, không tạo các CSV báo cáo Python.

| Phần gốc | Cách dùng hiện tại |
|---|---|
| `Config` | Giữ defaults 5000 episode, 20 lượt, alpha=.1, gamma=.9 |
| `ECommerceEnvironment` | Giữ nguyên mô hình nhu cầu/giảm giá/lợi nhuận |
| `QLearningAgent` | Giữ nguyên Q-table, epsilon-greedy, update terminal |
| `RuleBasedAgent` | Giữ nguyên luật đối thủ |
| `train_rl_against_rule`, `evaluate_pair` | Gọi trực tiếp, train/evaluate riêng |
| Baseline, alpha, exploration, multi-seed | Chạy trực tiếp và đưa artifact thật lên web |
| Thay đổi ở wrapper | Đường dẫn output, Matplotlib Agg, manifest truy nguyên |
| Luật dữ liệu tiểu đường | Tạo class khác trong `diabetes_demo/agents.py`, chỉ kế thừa tư tưởng luật minh bạch |

Không gán profit reward thành “sức khỏe người bệnh”. Module RL độc lập vì môi trường cũ không mô tả tương tác y tế phù hợp.

## State, action, reward và transition

State quan sát cho RL là `(demand, competitor_previous_action)`. Mỗi biến 0/1, có 4 trạng thái quan sát. Action 0 = không giảm giá, 1 = giảm giá 20%. Q-table có shape `(2,2,2)`, tổng 8 giá trị.

Nếu demand thấp, n=10; cao, n=20. Xác suất bán hàng:

```text
p = clip(0.30 + 0.25*demand + 0.15*own_action - 0.10*other_action, 0.05, 0.95)
sales ~ Binomial(n, p)
effective_price = 10 * (1 - 0.2*own_action)
reward = sales * (effective_price - 5)
```

Giảm giá tăng xác suất bán nhưng giảm lãi mỗi đơn từ 5 xuống 3. Hai bên chọn trước khi `env.step` chạy; không thấy hành động hiện tại của nhau. Sau step, hành động được lưu làm previous_actions và nhu cầu mới chọn ngẫu nhiên độc lập. Mỗi episode reset previous_actions về 0 và demand ngẫu nhiên.

Mỗi bên lấy số đơn binomial riêng trên cùng n; mô phỏng không phân bổ độc quyền một tập khách hàng chung. Đây là giả định minh họa, không phải mô hình thị trường thực tế đầy đủ.

## Q-table mang ý nghĩa gì?

`Q(s,a)` ước lượng tổng reward chiết khấu nếu chọn a tại s rồi tiếp tục theo giá trị tốt nhất ước lượng. Nó không chỉ lưu reward lần gần nhất.

```text
target = reward                              nếu terminal
target = reward + gamma * max_a' Q(next_s,a') nếu chưa terminal
Q(s,a) ← Q(s,a) + alpha * (target - Q(s,a))
```

Ví dụ thuật toán, không phải kết quả thí nghiệm: Q cũ=10, reward=5, maxQnext=20, alpha=.1, gamma=.9 → target=23 → Q mới=11.3. Nếu terminal thì target=5 → Q mới=9.5. Test trong `tests/test_rl.py` kiểm tra riêng hai trường hợp để tránh lỗi cộng reward tương lai ở cuối episode.

Alpha điều khiển mức cập nhật theo sai lệch; gamma giảm trọng số reward xa. Epsilon-greedy chọn random action với xác suất epsilon, còn lại chọn Q cao nhất. Khi hòa, code dùng random tie-breaking. Random exploration vẫn có thể chọn đúng action greedy; epsilon không phải tỷ lệ action “sai”.

## Huấn luyện và đánh giá

Mỗi episode: `epsilon=max(epsilon_min, epsilon_start*epsilon_decay**episode)`. Agent RL có RNG riêng, seed=environment seed+10000. Rule không học và chỉ giảm khi demand cao + RL đã giảm ở lượt trước. Sau mỗi bước, RL cập nhật Q; cộng reward của cả hai bên.

Evaluation dùng environment seed khác, `explore=False` và không gọi `learn`. Q-table được giữ nguyên; chỉ đo reward/action counts. 500 episode × 20 lượt = 10000 lượt đánh giá baseline. Multi-seed dùng 11/22/33/44/55, báo mean và sample standard deviation (`ddof=1`). So alpha và exploration trong code gốc dùng một training seed mỗi cấu hình; không đủ chứng minh cấu hình tốt nhất trên mọi seed.

## Giới hạn state cần biết khi bảo vệ

State RL gốc **không chứa hành động trước của chính RL**, trong khi luật hiện tại của đối thủ lại phụ thuộc giá trị đó. Hai lịch sử có cùng state quan sát RL nhưng dẫn đến hành động đối thủ khác nhau; vì thế state quan sát có thể không thỏa Markov đầy đủ. State cũng không chứa số lượt còn lại trong horizon hữu hạn. Không được viện dẫn bảo đảm hội tụ tối ưu của tabular Q-learning như thể mọi giả định đã được đáp ứng.

Hướng cải tiến hợp lý là thêm own_previous_action và time_remaining vào state, rồi đánh giá lại; chưa thay đổi trong module bảo toàn mã gốc. Bảng Q chỉ có 8 ô nên cũng không đại diện cho deep reinforcement learning.

Lần chạy thật cho thấy reward RL/rule gần nhau. Chính sách giữ giá có thể gần luật gốc trong các trạng thái gặp khi evaluate; trạng thái ít gặp/không reachable có thể có greedy action khác. “RL có học” không đồng nghĩa “RL luôn thắng”. Hãy chỉ CSV/Q-table, so sánh nhiều seed và thảo luận giả định môi trường.
