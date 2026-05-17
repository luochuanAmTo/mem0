### 增加print输出观察数据流

```py
async def ainvoke(
    self,
    input: MemoryState,
    config: typing.Optional[RunnableConfig] = None,
    **kwargs: typing.Any,
) -> list[ExtractedMemory]:
    max_steps = input.get("max_steps")
    if max_steps is None:
        max_steps = 1
    messages = input["messages"]
    existing = input.get("existing")
    prepared_messages = self._prepare_messages(messages, max_steps)
    prepared_existing = self._prepare_existing(existing)
    # Track external memory IDs (those passed in from outside)
    external_ids = {mem_id for mem_id, _, _ in prepared_existing}

    extractor = create_extractor(
        self.model,
        tools=list(self.schemas),
        enable_inserts=self.enable_inserts,
        enable_updates=self.enable_updates,
        enable_deletes=self.enable_deletes,
        existing_schema_policy=False,
    )
    # initial payload uses the full prepared_existing list
    payload = {"messages": prepared_messages, "existing": prepared_existing}

    print("payload:")
    print(payload)

    # Use a dict to record the latest update for each memory id.
    results: dict[str, BaseModel] = {}

    for i in range(max_steps):
        if i == 1:
            extractor = create_extractor(
                self.model,
                tools=list(self.schemas) + [Done],
                enable_inserts=self.enable_inserts,
                enable_updates=self.enable_updates,
                enable_deletes=self.enable_deletes,
                existing_schema_policy=False,
            )
        response = await extractor.ainvoke(payload, config=config)
        print('*'*50)
        print(response)
        print('*' * 50)
        is_done = False
        step_results = {}
        for r, rmeta in zip(response["responses"], response["response_metadata"]):
            if hasattr(r, "__repr_name__") and r.__repr_name__() == "Done":
                is_done = True
                continue
            mem_id = (
                r.json_doc_id
                if hasattr(r, "__repr_name__") and r.__repr_name__() == "RemoveDoc"
                else rmeta.get("json_doc_id", str(uuid.uuid4()))
            )
            step_results[mem_id] = r
        results.update(step_results)
        print("results:")
        print(results)
        for mem_id, _, mem in prepared_existing:
            if mem_id not in results:
                results[mem_id] = mem

        ai_msg = response["messages"][-1]
        if is_done or not ai_msg.tool_calls:
            break
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
                for r, rmeta in zip(
                    response["responses"], response["response_metadata"]
                )
            ]
            prepared_messages = (
                prepared_messages
                + [response["messages"][-1]]
                + [
                    {
                        "role": "tool",
                        "content": f"Memory {rid} {action}.",
                        "tool_call_id": tc["id"],
                    }
                    for tc, ((rid, _), action) in zip(
                        ai_msg.tool_calls, zip(list(step_results.items()), actions)
                    )
                ]
            )
            print("prepared_messages:")
            print(prepared_messages)
            # For the next iteration payload, drop all removal objects.
            payload = {
                "messages": prepared_messages,
                "existing": self._filter_response(
                    list(results.items()), external_ids, exclude_removals=True
                ),
            }
            print("payload:")
            print(payload)

    # For the final response, include removals only if they refer to an external memory.
    return self._filter_response(
        list(results.items()), external_ids, exclude_removals=False
    )
    
```

#### output:

```

payload:
{'messages': [{'role': 'system', 'content': 'You are a memory subroutine for an AI.'}, {'role': 'user', 'content': 'You are a long-term memory manager maintaining a core store of semantic, procedural, and episodic memory. These memories power a life-long learning agent\'s core predictive model.\n\nWhat should the agent learn from this interaction about the user, itself, or how it should act? Reflect on the input trajectory and current memories (if any).\n\n1. **Extract & Contextualize**  \n   - Identify essential facts, relationships, preferences, reasoning procedures, and context\n   - Caveat uncertain or suppositional information with confidence levels (p(x)) and reasoning\n   - Quote supporting information when necessary\n\n2. **Compare & Update**  \n   - Attend to novel information that deviates from existing memories and expectations.\n   - Consolidate and compress redundant memories to maintain information-density; strengthen based on reliability and recency; maximize SNR by avoiding idle words.\n   - Remove incorrect or redundant memories while maintaining internal consistency\n\n3. **Synthesize & Reason**  \n   - What can you conclude about the user, agent ("I"), or environment using deduction, induction, and abduction?\n   - What patterns, relationships, and principles emerge about optimal responses?\n   - What generalizations can you make?\n   - Qualify conclusions with probabilistic confidence and justification\n\nAs the agent, record memory content exactly as you\'d want to recall it when predicting how to act or respond. \nPrioritize retention of surprising (pattern deviation) and persistent (frequently reinforced) information, ensuring nothing worth remembering is forgotten and nothing false is remembered. Prefer dense, complete memories over overlapping ones.\n\nEnrich, prune, and organize memories based on any new information. If an existing memory is incorrect or outdated, update it based on the new information. All operations must be done in single parallel multi-tool call. Avoid duplicate extractions. \n\n<session_885d1d04-20f2-4490-811a-b716755f046a>\n================================ Human Message =================================\n\n我改变主意了，现在喜欢暗色主题。另外，我最近搬家到了北京。\n\n================================== Ai Message ==================================\n\n好的，已更新您的偏好和位置信息。\n</session_885d1d04-20f2-4490-811a-b716755f046a>\n\nYou have a maximum of 1 attempts'}], 'existing': [('mem_001', 'UserPreference', UserPreference(key='theme', value='light')), ('mem_002', 'Memory', Memory(content='用户住在上海'))]}
**************************************************
{'messages': [AIMessage(content='', additional_kwargs={'updated_docs': {'11a03bf6-dc12-4fb2-bf9f-ddf1d5024386': 'mem_001', '975fb3f2-26fb-4e56-aa4d-4ad5b096db72': 'mem_002'}}, response_metadata={}, id='0e1559ab-3eff-4933-8395-cc6e605a6b9c', tool_calls=[{'name': 'UserPreference', 'args': {'key': 'theme', 'value': 'dark'}, 'id': '11a03bf6-dc12-4fb2-bf9f-ddf1d5024386', 'type': 'tool_call'}, {'name': 'Memory', 'args': {'content': '用户住在北京'}, 'id': '975fb3f2-26fb-4e56-aa4d-4ad5b096db72', 'type': 'tool_call'}], invalid_tool_calls=[])], 'responses': [UserPreference(key='theme', value='dark'), Memory(content='用户住在北京')], 'response_metadata': [{'id': '11a03bf6-dc12-4fb2-bf9f-ddf1d5024386', 'json_doc_id': 'mem_001'}, {'id': '975fb3f2-26fb-4e56-aa4d-4ad5b096db72', 'json_doc_id': 'mem_002'}], 'attempts': 1}
**************************************************
results:
{'mem_001': UserPreference(key='theme', value='dark'), 'mem_002': Memory(content='用户住在北京')}
prepared_messages:
[{'role': 'system', 'content': 'You are a memory subroutine for an AI.'}, {'role': 'user', 'content': 'You are a long-term memory manager maintaining a core store of semantic, procedural, and episodic memory. These memories power a life-long learning agent\'s core predictive model.\n\nWhat should the agent learn from this interaction about the user, itself, or how it should act? Reflect on the input trajectory and current memories (if any).\n\n1. **Extract & Contextualize**  \n   - Identify essential facts, relationships, preferences, reasoning procedures, and context\n   - Caveat uncertain or suppositional information with confidence levels (p(x)) and reasoning\n   - Quote supporting information when necessary\n\n2. **Compare & Update**  \n   - Attend to novel information that deviates from existing memories and expectations.\n   - Consolidate and compress redundant memories to maintain information-density; strengthen based on reliability and recency; maximize SNR by avoiding idle words.\n   - Remove incorrect or redundant memories while maintaining internal consistency\n\n3. **Synthesize & Reason**  \n   - What can you conclude about the user, agent ("I"), or environment using deduction, induction, and abduction?\n   - What patterns, relationships, and principles emerge about optimal responses?\n   - What generalizations can you make?\n   - Qualify conclusions with probabilistic confidence and justification\n\nAs the agent, record memory content exactly as you\'d want to recall it when predicting how to act or respond. \nPrioritize retention of surprising (pattern deviation) and persistent (frequently reinforced) information, ensuring nothing worth remembering is forgotten and nothing false is remembered. Prefer dense, complete memories over overlapping ones.\n\nEnrich, prune, and organize memories based on any new information. If an existing memory is incorrect or outdated, update it based on the new information. All operations must be done in single parallel multi-tool call. Avoid duplicate extractions. \n\n<session_885d1d04-20f2-4490-811a-b716755f046a>\n================================ Human Message =================================\n\n我改变主意了，现在喜欢暗色主题。另外，我最近搬家到了北京。\n\n================================== Ai Message ==================================\n\n好的，已更新您的偏好和位置信息。\n</session_885d1d04-20f2-4490-811a-b716755f046a>\n\nYou have a maximum of 1 attempts'}, AIMessage(content='', additional_kwargs={'updated_docs': {'11a03bf6-dc12-4fb2-bf9f-ddf1d5024386': 'mem_001', '975fb3f2-26fb-4e56-aa4d-4ad5b096db72': 'mem_002'}}, response_metadata={}, id='0e1559ab-3eff-4933-8395-cc6e605a6b9c', tool_calls=[{'name': 'UserPreference', 'args': {'key': 'theme', 'value': 'dark'}, 'id': '11a03bf6-dc12-4fb2-bf9f-ddf1d5024386', 'type': 'tool_call'}, {'name': 'Memory', 'args': {'content': '用户住在北京'}, 'id': '975fb3f2-26fb-4e56-aa4d-4ad5b096db72', 'type': 'tool_call'}], invalid_tool_calls=[]), {'role': 'tool', 'content': 'Memory mem_001 updated.', 'tool_call_id': '11a03bf6-dc12-4fb2-bf9f-ddf1d5024386'}, {'role': 'tool', 'content': 'Memory mem_002 updated.', 'tool_call_id': '975fb3f2-26fb-4e56-aa4d-4ad5b096db72'}]
payload:
{'messages': [{'role': 'system', 'content': 'You are a memory subroutine for an AI.'}, {'role': 'user', 'content': 'You are a long-term memory manager maintaining a core store of semantic, procedural, and episodic memory. These memories power a life-long learning agent\'s core predictive model.\n\nWhat should the agent learn from this interaction about the user, itself, or how it should act? Reflect on the input trajectory and current memories (if any).\n\n1. **Extract & Contextualize**  \n   - Identify essential facts, relationships, preferences, reasoning procedures, and context\n   - Caveat uncertain or suppositional information with confidence levels (p(x)) and reasoning\n   - Quote supporting information when necessary\n\n2. **Compare & Update**  \n   - Attend to novel information that deviates from existing memories and expectations.\n   - Consolidate and compress redundant memories to maintain information-density; strengthen based on reliability and recency; maximize SNR by avoiding idle words.\n   - Remove incorrect or redundant memories while maintaining internal consistency\n\n3. **Synthesize & Reason**  \n   - What can you conclude about the user, agent ("I"), or environment using deduction, induction, and abduction?\n   - What patterns, relationships, and principles emerge about optimal responses?\n   - What generalizations can you make?\n   - Qualify conclusions with probabilistic confidence and justification\n\nAs the agent, record memory content exactly as you\'d want to recall it when predicting how to act or respond. \nPrioritize retention of surprising (pattern deviation) and persistent (frequently reinforced) information, ensuring nothing worth remembering is forgotten and nothing false is remembered. Prefer dense, complete memories over overlapping ones.\n\nEnrich, prune, and organize memories based on any new information. If an existing memory is incorrect or outdated, update it based on the new information. All operations must be done in single parallel multi-tool call. Avoid duplicate extractions. \n\n<session_885d1d04-20f2-4490-811a-b716755f046a>\n================================ Human Message =================================\n\n我改变主意了，现在喜欢暗色主题。另外，我最近搬家到了北京。\n\n================================== Ai Message ==================================\n\n好的，已更新您的偏好和位置信息。\n</session_885d1d04-20f2-4490-811a-b716755f046a>\n\nYou have a maximum of 1 attempts'}, AIMessage(content='', additional_kwargs={'updated_docs': {'11a03bf6-dc12-4fb2-bf9f-ddf1d5024386': 'mem_001', '975fb3f2-26fb-4e56-aa4d-4ad5b096db72': 'mem_002'}}, response_metadata={}, id='0e1559ab-3eff-4933-8395-cc6e605a6b9c', tool_calls=[{'name': 'UserPreference', 'args': {'key': 'theme', 'value': 'dark'}, 'id': '11a03bf6-dc12-4fb2-bf9f-ddf1d5024386', 'type': 'tool_call'}, {'name': 'Memory', 'args': {'content': '用户住在北京'}, 'id': '975fb3f2-26fb-4e56-aa4d-4ad5b096db72', 'type': 'tool_call'}], invalid_tool_calls=[]), {'role': 'tool', 'content': 'Memory mem_001 updated.', 'tool_call_id': '11a03bf6-dc12-4fb2-bf9f-ddf1d5024386'}, {'role': 'tool', 'content': 'Memory mem_002 updated.', 'tool_call_id': '975fb3f2-26fb-4e56-aa4d-4ad5b096db72'}], 'existing': [ExtractedMemory(id='mem_001', content=UserPreference(key='theme', value='dark')), ExtractedMemory(id='mem_002', content=Memory(content='用户住在北京'))]}
**************************************************
{'messages': [AIMessage(content='', additional_kwargs={'updated_docs': {}}, response_metadata={}, id='5289bd19-5be0-4efa-a715-fe22ef0f8756', tool_calls=[{'name': 'Done', 'args': {}, 'id': 'a74f2466-7631-433b-93b2-49982057cb32', 'type': 'tool_call'}], invalid_tool_calls=[])], 'responses': [Done()], 'response_metadata': [{'id': 'a74f2466-7631-433b-93b2-49982057cb32'}], 'attempts': 1}
**************************************************
results:
{'mem_001': UserPreference(key='theme', value='dark'), 'mem_002': Memory(content='用户住在北京')}

========== 最终记忆 ==========

ExtractedMemory(id='mem_001', content=UserPreference(key='theme', value='dark'))

ExtractedMemory(id='mem_002', content=Memory(content='用户住在北京'))


进程已结束，退出代码为 0

```

#### 输出解析

#### 第一轮开始前的 Payload

```py
{
    'messages': [
        {'role': 'system', 'content': 'You are a memory subroutine for an AI.'},
        {'role': 'user', 'content': '...长指令...\n\n<session_xxx>\n对话内容\n</session_xxx>\n\nYou have a maximum of 1 attempts'}
    ], 
    'existing': [
        ('mem_001', 'UserPreference', UserPreference(key='theme', value='light')),
        ('mem_002', 'Memory', Memory(content='用户住在上海'))
    ]
}
```

#### 第一轮 Response

```
{
    'messages': [AIMessage(...)],  # AI 的工具调用消息
    'responses': [
        UserPreference(key='theme', value='dark'),  # 工具1
        Memory(content='用户住在北京')              # 工具2
    ],
    'response_metadata': [
        {'id': '11a03bf6...', 'json_doc_id': 'mem_001'},  # 工具1元数据
        {'id': '975fb3f2...', 'json_doc_id': 'mem_002'}   # 工具2元数据
    ],
    'attempts': 1
}
```

#### 第一轮处理后的 results

```py
{
    'mem_001': UserPreference(key='theme', value='dark'),
    'mem_002': Memory(content='用户住在北京')
}
```

#### 第二轮准备前的 prepared_messages

```py
[
    # 原有的系统消息和用户指令（保持不变）
    {'role': 'system', 'content': 'You are a memory subroutine...'},
    {'role': 'user', 'content': '...长指令...'},
    
    # 新增：第一轮 AI 的工具调用消息
    AIMessage(content='', tool_calls=[...]),
    
    # 新增：工具响应消息（2条）
    {'role': 'tool', 'content': 'Memory mem_001 updated.', 'tool_call_id': '11a03bf6...'},
    {'role': 'tool', 'content': 'Memory mem_002 updated.', 'tool_call_id': '975fb3f2...'}
]
```

#### 第二轮 Payload

```py
{
    'messages': [
        # 原有的系统消息和用户指令
        {'role': 'system', ...},
        {'role': 'user', ...},
        
        # 第一轮的 AI 消息（工具调用）
        AIMessage(content='', tool_calls=[...]),
        
        # 第一轮的工具响应
        {'role': 'tool', 'content': 'Memory mem_001 updated.', ...},
        {'role': 'tool', 'content': 'Memory mem_002 updated.', ...}
    ],
    'existing': [
        ExtractedMemory(id='mem_001', content=UserPreference(key='theme', value='dark')),
        ExtractedMemory(id='mem_002', content=Memory(content='用户住在北京'))
    ]
}
```

#### 第二轮 Response

```py
{
    'messages': [AIMessage(tool_calls=[{'name': 'Done', ...}])],
    'responses': [Done()],
    'response_metadata': [{'id': 'a74f2466...'}],
    'attempts': 1
}
```

#### 第二轮处理后的 results

```py
{
    'mem_001': UserPreference(key='theme', value='dark'),
    'mem_002': Memory(content='用户住在北京')
}
```

#### 最终输出

```py
[
    ExtractedMemory(id='mem_001', content=UserPreference(key='theme', value='dark')),
    ExtractedMemory(id='mem_002', content=Memory(content='用户住在北京'))
]
```

