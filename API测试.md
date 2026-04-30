```py
import os
from mem0 import MemoryClient

client = MemoryClient(
    api_key="m0-t3Lj5Ks8WWpKyVXHtwFhXcXvbHqUIGWAWgWDhRUu"
)
messages = [
    {"role": "user", "content": "Hi, I'm Alex. I'm a vegetarian and I'm allergic to nuts."},
    {"role": "assistant", "content": "Hello Alex! Noted that you're vegetarian and allergic to nuts."}
]

client.add(messages, user_id="alex")

query = "What can I cook for dinner tonight?"

results = client.search(query, filters={"user_id": "alex"})
print(results)
```

output：
{'results': [{'id': 'a77cad16-2cf2-4aea-b440-234ffba07720', 'memory': "User's name is Alex, follows a vegetarian diet, and has a nut allergy", 'user_id': 'alex', 'agent_id': None, 'app_id': None, 'run_id': None, 'score': 0.1262, 'metadata': {}, 'categories': ['personal_details', 'food', 'user_preferences', 'health'], 'created_at': '2026-04-25T02:43:10+00:00', 'updated_at': '2026-04-25T02:43:15.552994+00:00'}]}

进程已结束，退出代码为 0



调用 `client.add(messages, user_id="alex")` 时，Mem0 云端会利用大模型自动从对话中提取关键事实（比如“Alex 是素食主义者”“Alex 对坚果过敏”），经过去重后将它们转化为向量并存入托管向量库，完成记忆的持久化。

后续执行 `client.search(query, filters={"user_id": "alex"})` 时，云端会把查询也向量化，然后在 `user_id="alex"` 的记忆范围内进行语义相似度搜索，同时结合关键词匹配与实体增强等多路信号对候选记忆进行综合评分与排序，最终返回最相关的结果。

所有数据都存储在 Mem0 平台托管的云端基础设施中，包括原始文本、嵌入向量以及元数据，

![image-20260430163447814](.\assets\image-20260430163447814.png)



![image-20260430164842812](.\assets\image-20260430164842812.png)



extraction demo



```py
import os
import time
from mem0 import MemoryClient

# 初始化客户端
client = MemoryClient(
    api_key="m0-t3Lj5Ks8WWpKyVXHtwFhXcXvbHqUIGWAWgWDhRUu"
)

# 添加记忆
messages = [
    {"role": "user", "content": "Hi, I'm Alex. I'm a vegetarian and I'm allergic to nuts."},
    {"role": "assistant", "content": "Hello Alex! Noted that you're vegetarian and allergic to nuts."}
]

print("=== 添加记忆 ===")
result = client.add(messages, user_id="alex")
print(f"add 返回: {result}")
print(f"event_id: {result.get('event_id')}")

# 等待后台任务完成
print("\n等待后台处理完成...")
time.sleep(3)

# 搜索所有记忆
print("\n=== 搜索所有记忆 ===")
all_results = client.get_all(filters={"user_id": "alex"})
print("get_all 结果:")
for mem in all_results.get("results", []):
    print(f"  - ID: {mem['id']}")
    print(f"    memory: {mem['memory']}")
    print(f"    categories: {mem.get('categories', [])}")
    print(f"    metadata: {mem.get('metadata', {})}")
    print()

# 用搜索来获取实体关联信息
print("\n=== 搜索 'Alex vegetarian nuts' ===")
results = client.search("Alex vegetarian nuts", filters={"user_id": "alex"})
print("search 结果:")
for mem in results.get("results", []):
    print(f"  - ID: {mem['id']}")
    print(f"    memory: {mem['memory']}")
    print(f"    score: {mem['score']}")
    print(f"    categories: {mem.get('categories', [])}")
    print()
```

=== add 返回结果 ===
{'message': 'Memory processing has been queued for background execution', 'status': 'PENDING', 'event_id': '1ffa2030-5d1e-420f-b59f-620838acfa8a'}

=== search 返回结果 ===
{'results': [{'id': 'a77cad16-2cf2-4aea-b440-234ffba07720', 'memory': "User's name is Alex, follows a vegetarian diet, and has a nut allergy", 'user_id': 'alex', 'agent_id': None, 'app_id': None, 'run_id': None, 'score': 0.1262, 'metadata': {}, 'categories': ['personal_details', 'food', 'user_preferences', 'health'], 'created_at': '2026-04-25T02:43:10+00:00', 'updated_at': '2026-04-25T02:43:15.552994+00:00'}]}

进程已结束，退出代码为 0

