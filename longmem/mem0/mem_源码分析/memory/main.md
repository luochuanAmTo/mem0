### add()  search() 入口



```py
def add(
    self,
    messages,
    *,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    infer: bool = True,
    memory_type: Optional[str] = None,
    prompt: Optional[str] = None,
):
  

    processed_metadata, effective_filters = _build_filters_and_metadata(
        user_id=user_id,
        agent_id=agent_id,
        run_id=run_id,
        input_metadata=metadata,
    )

    if memory_type is not None and memory_type != MemoryType.PROCEDURAL.value:
        raise Mem0ValidationError(
            message=f"Invalid 'memory_type'. Please pass {MemoryType.PROCEDURAL.value} to create procedural memories.",
            error_code="VALIDATION_002",
            details={"provided_type": memory_type, "valid_type": MemoryType.PROCEDURAL.value},
            suggestion=f"Use '{MemoryType.PROCEDURAL.value}' to create procedural memories."
        )

    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    elif isinstance(messages, dict):
        messages = [messages]

    elif not isinstance(messages, list):
        raise Mem0ValidationError(
            message="messages must be str, dict, or list[dict]",
            error_code="VALIDATION_003",
            details={"provided_type": type(messages).__name__, "valid_types": ["str", "dict", "list[dict]"]},
            suggestion="Convert your input to a string, dictionary, or list of dictionaries."
        )

    if agent_id is not None and memory_type == MemoryType.PROCEDURAL.value:
        results = self._create_procedural_memory(messages, metadata=processed_metadata, prompt=prompt)
        return results

    if self.config.llm.config.get("enable_vision"):
        messages = parse_vision_messages(messages, self.llm, self.config.llm.config.get("vision_details"))
    else:
        messages = parse_vision_messages(messages)

    vector_store_result = self._add_to_vector_store(messages, processed_metadata, effective_filters, infer, prompt=prompt)
    return {"results": vector_store_result}
```

 

- **messages** (str 或 List[Dict[str, str]]): 要处理并存储的消息内容或消息列表（例如 `[{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]`）。
- **user_id** (str, 可选): 创建该记忆的用户 ID。默认为 None。
- **agent_id** (str, 可选): 创建该记忆的智能体 ID。默认为 None。
- **run_id** (str, 可选): 创建该记忆的运行 ID。默认为 None。
- **metadata** (dict, 可选): 与记忆一起存储的元数据。默认为 None。可以附加自定义信息
- **infer** (bool, 可选): 如果为 True（默认值），则使用大语言模型（LLM）从 'messages' 中提取关键事实，并决定是添加、更新还是删除相关记忆。如果为 False，则直接将 'messages' 作为原始记忆添加。
- **memory_type** (str, 可选): 指定记忆的类型。目前，只有 `MemoryType.PROCEDURAL.value`（"procedural_memory"）被明确用于创建程序性记忆（通常需要 'agent_id'）。其他情况下，记忆被视为通用的对话/事实记忆。
- **prompt** (str, 可选): 用于创建记忆的提示词。默认为 None。

**返回：**

- dict: 一个包含记忆添加操作结果的字典，通常在 "results" 键下包含一个受影响的记忆项列表（已添加、已更新）。例如： `{"results": [{"id": "...", "memory": "...", "event": "ADD"}]}`

**可能引发的异常：**

- **Mem0ValidationError**: 如果输入验证失败（例如无效的 memory_type、messages 格式等）。
- **VectorStoreError**: 如果向量存储操作失败。
- **EmbeddingError**: 如果嵌入生成失败。
- **LLMError**: 如果大语言模型（LLM）操作失败。
- **DatabaseError**: 如果数据库操作失败。

返回：

```
{
  "results": [
    {
      "id": "UUID格式的记忆ID",
      "memory": "提取出的记忆内容",      
      "event": "ADD"
    },
    {
      "id": "另一条记忆的ID",
      "memory": "另一条记忆内容",
      "event": "ADD"
    }
  ]
}
```

memory，实际存储的记忆文本。如果 `infer=True`，这里是大模型提炼过的事实；如果 `infer=False`，这里就是原始消息内容。



```py
def search(
    self,
    query: str,
    *,
    top_k: int = 20,
    filters: Optional[Dict[str, Any]] = None,
    threshold: float = 0.1,
    rerank: bool = False,
    **kwargs,
):
   
    # Reject top-level entity params - must use filters instead
    _reject_top_level_entity_params(kwargs, "search")

    # Validate search parameters (before applying defaults)
    _validate_search_params(threshold=threshold, top_k=top_k)

    # Validate and trim entity IDs in filters
    effective_filters = filters.copy() if filters else {}
    if "user_id" in effective_filters:
        effective_filters["user_id"] = _validate_and_trim_entity_id(
            effective_filters["user_id"], "user_id"
        )
    if "agent_id" in effective_filters:
        effective_filters["agent_id"] = _validate_and_trim_entity_id(
            effective_filters["agent_id"], "agent_id"
        )
    if "run_id" in effective_filters:
        effective_filters["run_id"] = _validate_and_trim_entity_id(
            effective_filters["run_id"], "run_id"
        )
    if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
        raise ValueError(
            "filters must contain at least one of: user_id, agent_id, run_id. "
            "Example: filters={'user_id': 'u1'}"
        )

    limit = top_k

    # Apply enhanced metadata filtering if advanced operators are detected
    if self._has_advanced_operators(effective_filters):
        processed_filters = self._process_metadata_filters(effective_filters)
        # Remove logical/operator keys that have been reprocessed
        for logical_key in ("AND", "OR", "NOT"):
            effective_filters.pop(logical_key, None)
        for fk in list(effective_filters.keys()):
            if fk not in ("AND", "OR", "NOT", "user_id", "agent_id", "run_id") and isinstance(effective_filters.get(fk), dict):
                effective_filters.pop(fk, None)
        effective_filters.update(processed_filters)

    keys, encoded_ids = process_telemetry_filters(effective_filters)
    capture_event(
        "mem0.search",
        self,
        {
            "limit": limit,
            "version": self.api_version,
            "keys": keys,
            "encoded_ids": encoded_ids,
            "sync_type": "sync",
            "threshold": threshold,
            "advanced_filters": bool(filters and self._has_advanced_operators(filters)),
        },
    )

    original_memories = self._search_vector_store(query, effective_filters, limit, threshold)

    # Apply reranking if enabled and reranker is available
    if rerank and self.reranker and original_memories:
        try:
            reranked_memories = self.reranker.rerank(query, original_memories, limit)
            original_memories = reranked_memories
        except Exception as e:
            logger.warning(f"Reranking failed, using original results: {e}")

    return {"results": original_memories}
```

**query** (str): 要搜索的查询内容。

**top_k** (int, 可选): 返回结果的最大数量。默认为 20。

**filters** (dict): 包含实体 ID 和可选元数据过滤条件的字典。必须至少包含 `user_id`、`agent_id`、`run_id` 中的一个。，这限定了在谁的记忆里搜索。

示例：`filters={"user_id": "u1", "agent_id": "a1"}`

`top_k`: 一个**整数**，控制最多返回几条记忆

`threshold`: 一个 **0 到 1 之间的小数**，作为相关性门槛。只有相似度得分高过这个值的记忆才会被返回。默认为 0.1。

`search` 方法始终返回一个**字典**

```
{
  "results": [
    {
      "id": "一条UUID格式的记忆ID",
      "memory": "具体的记忆内容字符串，比如'张三喜欢吃辣的菜'",
      "score": 0.92,
      "created_at": "2024-05-20T10:00:00Z",
      "updated_at": "2024-05-20T10:00:00Z",
      "user_id": "zhangsan",
      "actor_id": "张三",
      "role": "user"
    },
}
```



DEMO：

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

# 显式设置 infer=True（这也是默认行为）
result = client.add(messages, user_id="alex", infer=True)
print("=== add 返回结果 ===")
print(result)

query = "What can I cook for dinner tonight?"
results = client.search(query, filters={"user_id": "alex"})
print("\n=== search 返回结果 ===")
print(results)
```

D:\Apps\Anaconda\envs\cognee\python.exe F:\pythonproject\cooognee\mem0_test\test2.py 
=== add 返回结果 ===
{'message': 'Memory processing has been queued for background execution', 'status': 'PENDING', 'event_id': '8d3d88ee-c769-439a-869f-e89918d61083'}

=== search 返回结果 ===
{'results': [{'id': 'a77cad16-2cf2-4aea-b440-234ffba07720', 'memory': "User's name is Alex, follows a vegetarian diet, and has a nut allergy", 'user_id': 'alex', 'agent_id': None, 'app_id': None, 'run_id': None, 'score': 0.1262, 'metadata': {}, 'categories': ['personal_details', 'food', 'user_preferences', 'health'], 'created_at': '2026-04-25T02:43:10+00:00', 'updated_at': '2026-04-25T02:43:15.552994+00:00'}]}

进程已结束，退出代码为 0

**云端是异步处理的**。调用 `add()` 后，Mem0 云端**不会立即**完成记忆提取和存储，而是把这个任务放进队列里，在后台慢慢处理。所以返回的状态是 `"PENDING"`（待处理）。

### memory extraction如何从对话中提取事实 

```
def _should_use_agent_memory_extraction(self, messages, metadata):
    
    # Check if agent_id is present in metadata
    has_agent_id = metadata.get("agent_id") is not None

    # Check if there are assistant role messages
    has_assistant_messages = any(msg.get("role") == "assistant" for msg in messages)

    # Use agent memory extraction if agent_id is present and there are assistant messages
    return has_agent_id and has_assistant_messages
```

根据以下逻辑判断是否使用智能体记忆提取：

- 如果存在 agent_id **且**消息中包含助手（assistant）角色 -> 返回 True
- 其他情况 -> 返回 False

**参数：**

- **messages**: 消息字典列表
- **metadata**: 包含 user_id、agent_id 等信息的元数据

**返回：**

- bool: 如果应使用智能体记忆提取则返回 True，如果应使用用户记忆提取则返回 False



```py
def _add_to_vector_store(self, messages, metadata, filters, infer, prompt=None):
    if not infer:
        returned_memories = []
        for message_dict in messages:
            if (
                not isinstance(message_dict, dict)
                or message_dict.get("role") is None
                or message_dict.get("content") is None
            ):
                logger.warning(f"Skipping invalid message format: {message_dict}")
                continue

            if message_dict["role"] == "system":
                continue

            per_msg_meta = deepcopy(metadata)
            per_msg_meta["role"] = message_dict["role"]

            actor_name = message_dict.get("name")
            if actor_name:
                per_msg_meta["actor_id"] = actor_name

            msg_content = message_dict["content"]
            msg_embeddings = self.embedding_model.embed(msg_content, "add")
            mem_id = self._create_memory(msg_content, {msg_content: msg_embeddings}, per_msg_meta)

            returned_memories.append(
                {
                    "id": mem_id,
                    "memory": msg_content,
                    "event": "ADD",
                    "actor_id": actor_name if actor_name else None,
                    "role": message_dict["role"],
                }
            )
        return returned_memories
```

当 `infer=False` 时，不调用大模型，直接把每条消息当作一条独立记忆存储：

1. **逐条遍历消息**：循环处理 `messages` 列表中的每一条消息。
2. **跳过无效消息**：如果消息不是字典、缺少 `role` 或 `content` 字段，记录警告并跳过。
3. **跳过系统消息**：`role="system"` 的消息是给 AI 的指令，不需要被记住，直接跳过。
4. **构建元数据**：深拷贝传入的 `metadata`，并把当前消息的 `role`（user/assistant）写入元数据。
5. **提取说话人**：如果消息里有 `name` 字段（比如 `"name": "Alex"`），就把它记录为 `actor_id`，表示“这句话是谁说的”。
6. **异步嵌入**：用 `asyncio.to_thread` 把消息内容扔到线程池里生成嵌入向量，避免阻塞事件循环。
7. **创建记忆**：调用 `_create_memory` 把消息内容和嵌入向量一起存入向量库和历史表。
8. **收集结果**：把每条成功存储的记忆的 ID、内容、事件类型、说话人、角色汇总到列表里，最后返回。

**1. `messages`（消息列表）**
一个列表，里面是对话消息字典。例如：

```
[
  {"role": "system", "content": "你是一个有用的助手"},
  {"role": "user", "content": "我叫张三，是个程序员", "name": "张三"},
  {"role": "assistant", "content": "好的张三，记住了"}
]
```



**2. `metadata`（元数据字典）**
包含身份信息的字典。例如：

```
{"user_id": "zhangsan"}
```



**3. `filters`（过滤条件）**
同样包含身份信息的字典，在这个分支里暂时没用到，但必须传入。例如：

```
{"user_id": "zhangsan"}
```

**4. `infer`（布尔值）**
这里是 `False`。

**5. `prompt`（可选）**
这里为 `None`。

返回一个**列表**，里面是逐条存储的原始消息记忆（系统消息被跳过）：**`memory` 字段存的就是原始消息内容**：没有经过大模型提炼，直接原样存储。

text

```
[
  {
    "id": "a1b2c3d4-...",
    "memory": "我叫张三，是个程序员",
    "event": "ADD",
    "actor_id": "张三",
    "role": "user"
  },
  {
    "id": "e5f6g7h8-...",
    "memory": "好的张三，记住了",
    "event": "ADD",
    "actor_id": null,
    "role": "assistant"
  }
]
```

**Phase 2**

**把已有记忆、新对话、历史消息打包**。

##### Phase 3：批量嵌入把 Phase 2 提取到的所有记忆文本一次性批量生成嵌入向量。

##### Phase**7a. 全局去重**- **批量嵌入**-**批量搜索实体库**-**分拣：更新还是新增**-**批量插入新实体**

提取规则在utils/entity_extractions.py

对每条输入文本，系统按顺序提取四类实体：

| 类型         | 标签       | 例子                         | 提取逻辑                                                     |
| :----------- | :--------- | :--------------------------- | :----------------------------------------------------------- |
| **专有名词** | `PROPER`   | "OpenAI"、"San Francisco"    | 找到连续大写开头的词序列（跳过句首），去掉末尾的虚词         |
| **引用文本** | `QUOTED`   | 引号中的 "machine learning"  | 正则匹配双引号和特定位置的单引号内容                         |
| **复合名词** | `COMPOUND` | "machine learning algorithm" | 从名词短语中提取修饰词+核心名词的组合，过滤掉太通用的形容词和核心词 |
| **名词回退** | `NOUN`     | "team"、"effort"             | 当复合词修饰语只是环境描述词时，只保留核心名词               |