创建MemoryManager实例

未提供schema 则使用这种未定义的灵活方式，也可以使用自定义的schema

```py
class Memory(BaseModel):
    """Call this tool once for each new memory you want to record. Use multi-tool calling to record multiple new memories."""

    content: str = Field(
        description="The memory as a well-written, standalone episode/fact/note/preference/etc."
        " Refer to the user's instructions for more information the preferred memory organization."
    )#Field 是 Pydantic 提供的字段配置函数，用于为字段添加元数据（如 description），这些描述会被转换为提示词的一部分，指导 LLM 如何生成该字段的内容，从而提高提取的准确性。
```

self.instructions = instructions

```py
_MEMORY_INSTRUCTIONS = """You are a long-term memory manager maintaining a core store of semantic, procedural, and episodic memory. These memories power a life-long learning agent's core predictive model.

What should the agent learn from this interaction about the user, itself, or how it should act? Reflect on the input trajectory and current memories (if any).

1. **Extract & Contextualize**  
   - Identify essential facts, relationships, preferences, reasoning procedures, and context
   - Caveat uncertain or suppositional information with confidence levels (p(x)) and reasoning
   - Quote supporting information when necessary

2. **Compare & Update**  
   - Attend to novel information that deviates from existing memories and expectations.
   - Consolidate and compress redundant memories to maintain information-density; strengthen based on reliability and recency; maximize SNR by avoiding idle words.
   - Remove incorrect or redundant memories while maintaining internal consistency

3. **Synthesize & Reason**  
   - What can you conclude about the user, agent ("I"), or environment using deduction, induction, and abduction?
   - What patterns, relationships, and principles emerge about optimal responses?
   - What generalizations can you make?
   - Qualify conclusions with probabilistic confidence and justification

As the agent, record memory content exactly as you'd want to recall it when predicting how to act or respond. 
Prioritize retention of surprising (pattern deviation) and persistent (frequently reinforced) information, ensuring nothing worth remembering is forgotten and nothing false is remembered. Prefer dense, complete memories over overlapping ones."""
```



```
async def ainvoke(
    self,
    input: MemoryState,
    config: typing.Optional[RunnableConfig] = None,
    **kwargs: typing.Any,
) -> list[ExtractedMemory]:
```

MemoryState继承自 MessagesState，主要包含三个字段：messages必填，对话历史列表，每条包含 role 和 content、existing（可选，已有记忆列表，格式为 [(id, 记忆对象), ...]）、max_steps（可选，默认 1，控制 LLM 最多迭代几轮）。示例：

```py
await manager.ainvoke({"messages": [{"role": "user", "content": "我喜欢猫"}],
                       "existing": [("mem1", Preference(key="animal", value="cat"))], 						   "max_steps": 2})
```

config包含callbacks：回调函数（如日志、追踪、监控）

tags：标签（用于分组/筛选运行记录）

metadata：元数据（附加信息，如用户 ID、会话 ID）

max_concurrency：最大并发数

recursion_limit：递归限制等

```py
max_steps = input.get("max_steps")
if max_steps is None:
    max_steps = 1   #max_steps可选，默认 1，
```

```py
messages = input["messages"]
existing = input.get("existing")  #读取对话和记忆
```

```py
prepared_messages = self._prepare_messages(messages, max_steps)
    
```

 _prepare_messages

```py
def _prepare_messages(
    self, messages: list[AnyMessage], max_steps: int = 1
) -> list[dict]:
    id_ = str(uuid.uuid4())
    session = (
        f"\n\n<session_{id_}>\n{utils.get_conversation(messages)}\n</session_{id_}>"
    )
    if max_steps > 1:
        session = f"{session}\n\nYou have a maximum of {max_steps - 1} attempts"
        " to form and consolidate memories from this session."
    return [
        {"role": "system", "content": "You are a memory subroutine for an AI."},
        {
            "role": "user",
            "content": (
                f"{self.instructions}\n\nEnrich, prune, and organize memories based on any new information. "
                f"If an existing memory is incorrect or outdated, update it based on the new information. "
                f"All operations must be done in single parallel multi-tool call."
                f" Avoid duplicate extractions. {session}"
            ),
        },
    ]
```

 _prepare_messages加工后的输出：

```py

[
    {
        "role": "system",
        "content": "You are a memory subroutine for an AI."
    },
    {
        "role": "user",
        "content": (
            "你是一个记忆管理助手，负责从对话中提取重要信息（instraction）\n\n"
            "Enrich, prune, and organize memories based on any new information. "
            "If an existing memory is incorrect or outdated, update it based on the new information. "
            "All operations must be done in single parallel multi-tool call. "
            "Avoid duplicate extractions. "
            "\n\n<session_abc123-def456-7890>\n"
            "user: 我喜欢在晚上用暗色模式看书\n"
            "assistant: 好的，我记住了，你偏好暗色模式\n"
            "user: 对了，我最近在读《三体》\n"
            "assistant: 《三体》是部很棒的科幻小说\n"
            "</session_abc123-def456-7890>\n\n"
            "You have a maximum of 1 attempts to form and consolidate memories from this session."
        )
    }
]
```

```py
prepared_existing = self._prepare_existing(existing)
```

```py
    def _prepare_existing(
        self,
        existing: typing.Optional[
            typing.Union[
                list[str], list[tuple[str, BaseModel]], list[tuple[str, str, dict]]
            ]
        ],
    ) -> list[tuple[str, str, typing.Any]]:
        if existing is None:
            return []
        if all(isinstance(ex, str) for ex in existing):
            MemoryModel = self.schemas[0]
            return [
                (str(uuid.uuid4()), "Memory", MemoryModel(content=ex))
                for ex in existing
            ] #如果是纯字符串列表直接返回返回转换后的三元组列表，【UUID ，记忆类型Memory，将文本放入 Memory(content=ex) 对象中】 如
            '''输入: existing = ["用户喜欢暗色模式", "用户是程序员"]
               输出: [
                    ("uuid-1", "Memory", Memory(content="用户喜欢暗色模式")),
                    ("uuid-2", "Memory", Memory(content="用户是程序员"))
]'''
            
            
            
        result = []
        for e in existing:
            if isinstance(e, (tuple, list)) and len(e) == 3:
                result.append(tuple(e))  #已经是 (id, schema_name, memory_object) 格式直接转换为元组并添加到结果中
            else:
                # Assume a two-element tuple: (id, value)
                id_, value = e[0], e[1]
                kind = (
                    value.__repr_name__() if isinstance(value, BaseModel) else "__any__"
                )
                result.append((id_, kind, value))
        return result
    '''输出统一转换为 (id, schema_name, memory_object) 的三元组列表.'''
```

```py
external_ids = {mem_id for mem_id, _, _ in prepared_existing}
```

从标准化后的外部已有记忆中提取所有记忆ID存入external_ids

```py
extractor = create_extractor(
    self.model,                    # 大语言模型实例
    tools=list(self.schemas),      # 注册的工具列表
    enable_inserts=self.enable_inserts,   # 是否允许插入新记忆
      	'''	Memory 模型 → InsertMemory 工具
			UserPreference 模型 → InsertUserPreference 工具
			TodoItem 模型 → InsertTodoItem 工具'''
    enable_updates=self.enable_updates,   # 是否允许更新已有记忆
    enable_deletes=self.enable_deletes,   # 是否允许删除记忆
    existing_schema_policy=False,         # 不对已存在的记忆进行schema强制校验
)
```

创建提取器,

```py
payload = {"messages": prepared_messages, "existing": prepared_existing}
#整合输入数据
```

```py
for i in range(max_steps):  #每轮 LLM 可以返回一批工具调用（插入/更新/删除记忆）
    if i == 1:      #当进入第二轮新创建提取器，在原有工具基础上增加 Done 工具这样设计让 LLM 在第一轮必须处理，后续轮次可以选择结束
        extractor = create_extractor(
            self.model,
            tools=list(self.schemas) + [Done],
            enable_inserts=self.enable_inserts,
            enable_updates=self.enable_updates,
            enable_deletes=self.enable_deletes,
            existing_schema_policy=False,
        )
    response = extractor.invoke(payload, config=config)  #调用 LLM
    		
    
    		
    		'''response = {
                        "responses": [           # 工具调用的参数对象列表
                            UserPreference(key="theme", value="dark"),
                            RemoveDoc(json_doc_id="mem_002")
                        ],
                        "response_metadata": [  # 每个调用的元数据
                            {"json_doc_id": "mem_001"},  # 更新操作需要指定 ID
                            {}                            # 删除操作不需要额外元数据
                        ],
                        "messages": [           # 完整的消息历史
                            {"role": "user", "content": "..."},
                            {"role": "assistant", "content": "...", "tool_calls": [...]},
                            # ... 可能还有工具响应消息
                        ]
                        }'''
    
   
```



```py
 	is_done = False     #标记本轮是否遇到 Done 工具
    step_results: dict[str, BaseModel] = {} #临时存储本轮产生的记忆（用于后续合并）
    for r, rmeta in zip(response["responses"], response["response_metadata"]):
        #r:工具调用的参数对象（如 UserPreference(...)、RemoveDoc(...)、Done）
        #rmeta：元数据字典包含 json_doc_id 等信息
        if hasattr(r, "__repr_name__") and r.__repr_name__() == "Done":
            is_done = True
            continue #如果是 Done，设置 is_done = True 并跳过本次循环（不将其作为记忆存储）
        mem_id = (
            r.json_doc_id  
            if (
                hasattr(r, "__repr_name__") and r.__repr_name__() == "RemoveDoc"
            )  #如果 r 是 RemoveDoc 对象直接使用 r.json_doc_id（要删除的记忆 ID）
            else rmeta.get("json_doc_id", str(uuid.uuid4())) #为每个记忆操作确定唯一的 ID。从元数据中获取 json_doc_id（更新时需要指定要更新的记忆 ID）果没有（插入新记忆），则生成新的 UUID
        )
        step_results[mem_id] = r #存储本轮结果
    	results.update(step_results) #合并到全局结果
        for mem_id, _, mem in prepared_existing:
        	if mem_id not in results:
            	results[mem_id] = mem  #如果它的 ID 不在 results 中（意味着本轮没有任何操作影响它）则将其原样添加到 results 中（保留原值）如果不补齐，这些记忆就会丢失

        ai_msg = response["messages"][-1]  #response["messages"]：完整的消息历史列表ai_msg 包含 tool_calls 字段，记录了模型请求调用的所有工具
        if is_done or not ai_msg.tool_calls:
            break  #LLM 调用了 Done 工具，主动表示完成，AI 消息中没有工具调用，说明不需要任何记忆操作
        if i < max_steps - 1:
            actions = [
                (
                    "updated"
                    if rmeta.get("json_doc_id")
                    else (
                        "deleted"
                        if hasattr(r, "__repr_name__")
                        and r.__repr_name__() == "RemoveDoc"
                        else "inserted"
                    )
                )
                #判断逻辑（从内到外）：如果 rmeta.get("json_doc_id") 存在 → "updated"（更新已有记忆）否则，如果 r 是 RemoveDoc 对象 → "deleted"，否则 → "inserted"（插入新记忆）
                for r, rmeta in zip(
                    response["responses"], response["response_metadata"]
                )
            ]
            prepared_messages = (
                prepared_messages
                + [response["messages"][-1]] #+ [response["messages"][-1]]：添加本轮 AI 的消息（包含工具调用请求）
                + [
                    {
                        "role": "tool",
                        "content": f"Memory {rid} {action}.",
                        "tool_call_id": tc["id"],
                    }   #+ [工具响应消息列表]：为每个工具调用生成一条响应消息
                    for tc, ((rid, _), action) in zip(
                        ai_msg.tool_calls, zip(list(step_results.items()), actions)
                    )
                ]
                # zip(list(step_results.items()), actions)
                # step_results.items() = [("mem_001", obj1), ("mem_002", obj2), ...]
                # actions = ["updated", "deleted", ...]
                # 配对后：[(("mem_001", obj1), "updated"), (("mem_002", obj2), "deleted")]

                # zip(ai_msg.tool_calls, 上面的配对)
                # ai_msg.tool_calls = [{"id": "call_1", ...}, {"id": "call_2", ...}]
                # 最终配对：[({"id": "call_1"}, (("mem_001", obj1), "updated")), ...]		
                
                
            )
            # For the next iteration payload, drop all removal objects.
            payload = {
                "messages": prepared_messages, #扩展后的完整对话历史
                "existing": self._filter_response( #当前所有记忆，但经过过滤
                    list(results.items()), external_ids, exclude_removals=True
                ),
            }

    # For the final response, include removals only if they refer to an external memory.
    return self._filter_response(
        list(results.items()), external_ids, exclude_removals=False
    )
    
```



示例：

```py
class Memory(BaseModel):
    content: str

class UserPreference(BaseModel):
    key: str
    value: str

schemas = [Memory, UserPreference]
max_steps = 2

#已有记忆
existing = [
    ("mem_001", UserPreference(key="theme", value="light")),
    ("mem_002", Memory(content="用户住在上海"))
]

# 对话
messages = [
    {"role": "user", "content": "我改变主意了，现在喜欢暗色主题。另外，我最近搬家到了北京。"},
    {"role": "assistant", "content": "好的，已更新您的偏好和位置信息。"}
]
```

#### 第一轮**payload 内容**

```py
payload = {
    "messages": [系统消息, 用户消息(含对话)],
    "existing": [
        ("mem_001", "UserPreference", UserPreference(key="theme", value="light")),
        ("mem_002", "Memory", Memory(content="用户住在上海"))
    ]
}
```

**LLM 响应**（推测）：

```py
response = {
    "responses": [
        UserPreference(key="theme", value="dark"),      # 更新主题
        RemoveDoc(json_doc_id="mem_002"),               # 删除旧地址
        Memory(content="用户住在北京")                   # 插入新地址
    ],
    "response_metadata": [
        {"json_doc_id": "mem_001"},  # 更新 mem_001
        {},                           # 删除操作
        {}                            # 插入新记忆
    ],
    "messages": [...]  # 省略
}
```

#### for r, rmeta in zip(response["responses"], response["response_metadata"]):

```py
is_done = False
step_results = {}

# 第1个工具：更新
mem_id = "mem_001"  # 从 rmeta 获取
step_results["mem_001"] = UserPreference(key="theme", value="dark")

# 第2个工具：删除
mem_id = "mem_002"  # 从 r.json_doc_id 获取
step_results["mem_002"] = RemoveDoc(json_doc_id="mem_002")

# 第3个工具：插入
mem_id = "uuid-123"  # 新生成
step_results["uuid-123"] = Memory(content="用户住在北京")

# 合并到全局
results.update(step_results)
# results = {
#     "mem_001": UserPreference(key="theme", value="dark"),
#     "mem_002": RemoveDoc(json_doc_id="mem_002"),
#     "uuid-123": Memory(content="用户住在北京")
# }
```

#### 补齐未被操作的外部记忆

```py
for mem_id, _, mem in prepared_existing:
    if mem_id not in results:
        results[mem_id] = mem

# prepared_existing 中有 mem_001 和 mem_002
# mem_001 已在 results 中（被更新了）
# mem_002 已在 results 中（被删除了）
# 无需补齐
```



```py
ai_msg = response["messages"][-1]  # 获取 AI 消息
# is_done = False（没有 Done 工具）
# ai_msg.tool_calls 有内容（3个工具调用）
if is_done or not ai_msg.tool_calls:  # False
    break  # 不执行，继续
```

#### 准备下一轮

```py
构建 actions 列表
actions = [
    "updated",   # 工具1: rmeta有json_doc_id
    "deleted",   # 工具2: 是 RemoveDoc
    "inserted"   # 工具3: 其他情况
]
```



#### 扩展对话历史

```py
prepared_messages = (
    prepared_messages  # 原有消息
    + [response["messages"][-1]]  # 添加 AI 的工具调用消息
    + [  # 添加工具响应消息
        {
            "role": "tool",
            "content": "Memory mem_001 updated.",
            "tool_call_id": "call_abc"  # 对应AI消息中的工具调用ID
        },
        {
            "role": "tool",
            "content": "Memory mem_002 deleted.",
            "tool_call_id": "call_def"
        },
        {
            "role": "tool",
            "content": "Memory 550e8400... inserted.",
            "tool_call_id": "call_ghi"
        }
    ]
)
```

#### 更新下一轮的 payload

```py
payload = {
    "messages": prepared_messages,  # 扩展后的对话历史
    "existing": self._filter_response(
        list(results.items()), 
        external_ids={"mem_001", "mem_002"}, 
        exclude_removals=True  # 排除删除标记
    )
}

# _filter_response 过滤后，existing 为：
# [
#     ("mem_001", "UserPreference", UserPreference(key="theme", value="dark")),
#     ("550e8400...", "Memory", Memory(content="用户住在北京"))
# ]
# 注意：mem_002 的 RemoveDoc 被排除了
```





#### 第二轮（i=1， max_steps=2）重新创建 extractor，加入 `Done` 工具。

```
# 新的 payload 会包含第一轮的工具执行结果
payload = {
    "messages": [
        ...原有消息...,
        {"role": "assistant", "content": "...", "tool_calls": [...]},
        {"role": "tool", "content": "Memory mem_001 updated."},
        {"role": "tool", "content": "Memory mem_002 deleted."},
        {"role": "tool", "content": "Memory uuid-123 inserted."}
    ],
    "existing": [
        ("mem_001", "UserPreference", UserPreference(key="theme", value="dark")),
        ("uuid-123", "Memory", Memory(content="用户住在北京"))
        # 注意：mem_002 被排除了（exclude_removals=True）
    ]
}
```

**LLM 可能响应**：

```
response = {
    "responses": [Done()],  # 不需要更多操作，主动结束
    "response_metadata": [{}],
    "messages": [...]
}
```

```
# 遇到 Done
is_done = True
step_results = {}  # 没有其他工具
results.update(step_results)  # 无变化

# 补齐外部记忆
# mem_001 和 mem_002 已在 results 中

# 检查终止
ai_msg = response["messages"][-1]
is_done = True → break
```

### 最终返回

```
return _filter_response(
    [
        ("mem_001", UserPref(key="theme", value="dark")),
        ("mem_002", RemoveDoc(json_doc_id="mem_002")),
        ("uuid-123", Memory(content="用户住北京"))
    ],
    external_ids={"mem_001", "mem_002"},
    exclude_removals=False  # 保留删除操作
)
# 结果：返回所有三个记忆，mem_002 作为删除标记

输出：[
    ExtractedMemory(id="mem_001", content=UserPref(key="theme", value="dark")),
    ExtractedMemory(id="mem_002", content=RemoveDoc(json_doc_id="mem_002")),  # 表示要删除
    ExtractedMemory(id="uuid-123", content=Memory(content="用户住北京"))
]
```


