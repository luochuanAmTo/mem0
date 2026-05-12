extraction.py 本质是让LLM学会像人一样整理长期记忆，MemoryManager.ainvoke()，这是整个 extraction 的大脑。

是一个AI记忆反思循环，例如

messages:
\- 我喜欢机器人
\- 最近开始喜欢四足机器人

existing:
\- 用户喜欢机器人

LLM收到 prompt类似于：



```
已有记忆：
- 用户喜欢机器人

新对话：
- 最近开始喜欢四足机器人

请判断：
- 是否新增记忆
- 是否更新旧记忆
- 是否删除记忆
```

#### 一、Class  MemoryManager是一个 Runnable，它的核心任务是分析对话和现有记忆，然后输出结构化的新记忆列表,如通过 .invoke() 或 .ainvoke()方法调用

```py
async def ainvoke(
    self,
    input: MemoryState,
    config: typing.Optional[RunnableConfig] = None,
    **kwargs: typing.Any,
) -> list[ExtractedMemory]:
```

异步记忆分析入口输入：MemoryState如 

```py
{
    "messages": [...],
    "existing": [...],
    "max_steps": 3
}
```

```py
 max_steps = input.get("max_steps")
        if max_steps is None:
            max_steps = 1  # 最大反思轮数
        messages = input["messages"]   #用户聊天。用户：我喜欢机器人
        existing = input.get("existing")  #已有长期记忆。
        prepared_messages = self._prepare_messages(messages, max_steps)#构造 Prompt
        prepared_existing = self._prepare_existing(existing) #统一记忆格式
        # Track external memory IDs (those passed in from outside)
        external_ids = {mem_id for mem_id, _, _ in prepared_existing} #保存已有记忆ID集合
```

#### 创建 extractor

```py
extractor = create_extractor(
    self.model,
    tools=list(self.schemas),   #给LLM绑定Tools 强制Tool Calling，强制结构化输出
    enable_inserts=self.enable_inserts,
    enable_updates=self.enable_updates,
    enable_deletes=self.enable_deletes,
    existing_schema_policy=False,
)  #创建结构化LLM提取器把LLM变成可调用记忆工具的AI


```

#### 初始化 payload

```py
payload = {
    "messages": prepared_messages,
    "existing": prepared_existing
}
```

第一轮给LLM的输入相当于：

```
聊天：
...

已有记忆：
...
```

```py
response = await extractor.ainvoke(payload, config=config)
#真正调用LLM，messages+existing memories，输出tool calls
例如：create memory
	update memory
	delete memory
	Done
```

解析LLM输出遍历：所有 Tool Calls 如create memory，update memory，RemoveDoc，Done

```py
step_results[mem_id] = r
#保存本轮结果
results.update(step_results)#合并到最终记忆表
```

循环进行：

1. 看聊天
2. 看旧记忆
3. 判断重要性
4. 判断冲突
5. create/update/delete
6. 反思
7. 再修正
8. 输出最终记忆状态

​     

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
        
最终 Prompt
[
  {
    "role": "system",
    "content": "You are a memory subroutine for an AI."
  },

  {
    "role": "user",
    "content": """
Extract & Contextualize...
Compare & Update...
Synthesize & Reason...

Enrich, prune, and organize memories...

<session_xxx>
User: 我喜欢机器人
Assistant: 好的
</session_xxx>
"""
  }
]
```

​    

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
            ]
        result = []
        for e in existing:
            if isinstance(e, (tuple, list)) and len(e) == 3:
                result.append(tuple(e))
            else:
                # Assume a two-element tuple: (id, value)
                id_, value = e[0], e[1]
                kind = (
                    value.__repr_name__() if isinstance(value, BaseModel) else "__any__"
                )
                result.append((id_, kind, value))
        return result
    
   将已有记忆标准化格式如：
[
    (
        "随机uuid1",
        "Memory",
        Memory(content="喜欢机器人")
    ),

    (
        "随机uuid2",
        "Memory",
        Memory(content="喜欢Python")
    )
]
```

```py
def create_memory_manager(
    model: str | BaseChatModel,
    /,
    *,
    schemas: typing.Sequence[S] = (Memory,),
    instructions: str = _MEMORY_INSTRUCTIONS,
    enable_inserts: bool = True,
    enable_updates: bool = True,
    enable_deletes: bool = False,
) -> Runnable[MemoryState, list[ExtractedMemory]]:

    return MemoryManager(
        model,
        schemas=schemas,
        instructions=instructions,
        enable_inserts=enable_inserts,
        enable_updates=enable_updates,
        enable_deletes=enable_deletes,
    )
```

把 LLM、Schema、Prompt、权限配置，封装成一个可执行的记忆提取器对象

```py
def create_memory_searcher(
    model: str | BaseChatModel,
    prompt: str = "Search for distinct memories relevant to different aspects of the provided context.",
    *,
    namespace: tuple[str, ...] = ("memories", "{langgraph_user_id}"),
) -> Runnable[MessagesState, typing.Awaitable[list[SearchItem]]]:
    
    template = ChatPromptTemplate.from_messages(
        [
            ("system", prompt),
            ("placeholder", "{messages}"),
            ("user", "\n\nSearch for memories relevant to the above context."),
        ]
    )

    # Initialize model and search tool
    model_instance = (
        model if isinstance(model, BaseChatModel) else init_chat_model(model)
    )
    search_tool = create_search_memory_tool(
        namespace=namespace, response_format="content_and_artifact"
    )
    query_gen = model_instance.bind_tools([search_tool], tool_choice="search_memory")
```

这个函数是先让模型根据对话生成搜索查询，再去记忆库里检索相关记忆并排序返回



#### 二、MemoryStoreManager类是一个自动管理长期记忆生命周期的总控制器，整合了搜索旧记忆，写回数据库，namespace 管理，query 生成，默认记忆，多阶段记忆整理。

```py
 def invoke(
        self,
        input: MemoryStoreManagerInput,
        config: typing.Optional[RunnableConfig] = None,
        **kwargs: typing.Any,
    ) -> list[dict]:
        store = self.store
        namespace = self.namespace(config)
        convo = utils.get_conversation(input["messages"])

        with get_executor_for_config(config) as executor:
            if self.query_gen:
                convo = utils.get_conversation(input["messages"])
                query_text = (
                    f"Use parallel tool calling to search for distinct memories relevant to this conversation.:\n\n"
                    f"<convo>\n{convo}\n</convo>."
                )
                query_req = self.query_gen.invoke(query_text, config=config)
                search_results_futs = [
                    executor.submit(
                        store.search,
                        namespace,
                        **({**tc["args"], "limit": self.query_limit}),
                    )
                    for tc in query_req.tool_calls
                ]
            else:
                # Search over "query_limit" timespans starting from the most recent
                queries = utils.get_dialated_windows(
                    input["messages"], self.query_limit // 4
                )
                search_results_lists = [
                    store.search(namespace, query=query) for query in queries
                ]
                search_results_futs = [
                    executor.submit(
                        store.search,
                        namespace,
                        query=query,
                        limit=self.query_limit,
                    )
                    for query in queries
                ]

        search_results_lists = [fut.result() for fut in search_results_futs]
        store_map = self._sort_results(search_results_lists, self.query_limit)
        if not store_map and self.default_factory is not None:
            config = ensure_config(config)
            default = self.default_factory(config)
            coerced = self._coerce_default(default, self.schemas)
            dumped = {
                "kind": coerced.__repr_name__(),
                "content": coerced.model_dump(mode="json"),
            }
            store.put(
                namespace,
                key="default",
                value=dumped,
            )
            now = datetime.datetime.now(datetime.timezone.utc)
            store_map = self._sort_results(
                [
                    [
                        SearchItem(
                            namespace, "default", dumped, created_at=now, updated_at=now
                        )
                    ]
                ],
                self.query_limit,
            )
        store_based = [
            (sid, item.value["kind"], item.value["content"])
            for sid, item in store_map.items()
        ]
        ephemeral: list[tuple[str, str, dict]] = []
        removed_ids: set[str] = set()

        enriched = self.memory_manager.invoke(
            {
                "messages": input["messages"],
                "existing": store_based,
                "max_steps": input.get("max_steps"),
            },
            config=config,
        )
        store_based, ephemeral, removed = self._apply_manager_output(
            enriched, store_based, store_map, ephemeral
        )
        removed_ids.update(removed)

        for phase in self.phases:
            phase_manager = self._build_phase_manager(phase)
            phase_messages = (
                input["messages"] if phase.get("include_messages", False) else []
            )
            phase_input = {
                "messages": phase_messages,
                "existing": store_based + ephemeral,
            }
            phase_enriched = phase_manager.invoke(phase_input, config=config)
            store_based, ephemeral, removed = self._apply_manager_output(
                phase_enriched, store_based, store_map, ephemeral
            )
            removed_ids.update(removed)

        final_mem = store_based + ephemeral
        final_puts = []
        for sid, kind, content in final_mem:
            if sid in removed_ids:
                continue
            if sid in store_map:
                old_art = store_map[sid]
                if old_art.value["kind"] != kind or old_art.value["content"] != content:
                    final_puts.append(
                        {
                            "namespace": old_art.namespace,
                            "key": old_art.key,
                            "value": {"kind": kind, "content": content},
                        }
                    )
            else:
                final_puts.append(
                    {
                        "namespace": namespace,
                        "key": sid,
                        "value": {"kind": kind, "content": content},
                    }
                )

        final_deletes = []
        for sid in removed_ids:
            if sid in store_map:
                art = store_map[sid]
                final_deletes.append((art.namespace, art.key))

        with get_executor_for_config(config) as executor:
            for put in final_puts:
                executor.submit(store.put, **put)
            for ns, key in final_deletes:
                executor.submit(store.delete, ns, key)

        return final_puts

```

输入：

```
{
   "messages": [...],
   "max_steps": 3
}
```

输出：

```
[
   {
      "namespace": ...,
      "key": ...,
      "value": ...
   }
]
```

store = self.store，获取 BaseStore。

convo = utils.get_conversation(input["messages"])把 message list 转成纯文本对话。

with get_executor_for_config(config) as executor:搜索已有记忆

search_results_lists = [fut.result() for fut in search_results_futs]收集搜索结果

store_map = self._sort_results(search_results_lists, self.query_limit)排序去重

enriched = self.memory_manager.invoke(）调用 MemoryManager

store_based, ephemeral, removed = self._apply_manager_output(）合并 manager 输出
