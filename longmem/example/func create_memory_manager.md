一、create_memory_manager给它一段和AI的聊天记录，它会从中提炼出需要记住的关键信息，责理解对话并把信息变成记忆

1.非结构化

```py
import asyncio
from langmem import create_memory_manager

manager = create_memory_manager(
    "ollama:qwen3:8b"
)  

                    '''def create_memory_manager(
                        model: str | BaseChatModel,      # 必选：模型名称或实例
                        /,                                # 位置参数分隔符
                        *,                                # 后面的参数必须使用关键字传递
                        schemas: typing.Sequence[S] = (Memory,),  # 记忆的schema定义
                          使用方法class PreferenceMemory(BaseModel):
                                        category: str
                                        preference: str
                                        context: str

                                    async def main():
                                        manager = create_memory_manager(
                                            "ollama:qwen3:8b",
                                            schemas=[PreferenceMemory],
                                        )
                        instructions: str = _MEMORY_INSTRUCTIONS, # 系统指令
                        enable_inserts: bool = True,      # 是否允许插入新记忆
                        enable_updates: bool = True,      # 是否允许更新现有记忆
                        enable_deletes: bool = False,     # 是否允许删除记忆
                    ) -> Runnable[MemoryState, list[ExtractedMemory]]:  # 返回类型注解'''


conversation = [
    {
        "role": "user",
        "content": "I prefer dark mode in all my apps"
    },
    {
        "role": "assistant",
        "content": "I'll remember that preference"
    },
]
async def main():
    # 提取记忆
    memories = await manager(conversation)
        #调用MemoryManager.__call__()，__call__() 内部又会调用await self.ainvoke(...)，
        #async def ainvoke(
               ...
            #messages = input["messages"]
        	#existing = input.get("existing")
        	#prepared_messages = self._prepare_messages(messages, max_steps)
        	#prepared_existing = self._prepare_existing(existing)
        #)
        '''extractor = create_extractor(
            self.model,
            tools=list(self.schemas),
            enable_inserts=self.enable_inserts,
            enable_updates=self.enable_updates,
            enable_deletes=self.enable_deletes,
            existing_schema_policy=False,
            
            
            payload = {"messages": prepared_messages, "existing": prepared_existing}包含当前对话消息和当前已知的记忆列表
            response = await extractor.ainvoke(payload, config=config)
              （response包含列表，每个元素是模型生成的工具调用参数，response_metadata列表，每个元素是与上述工具调用相关的元数据json_doc_id，messages列表，包含本次调用后完整的消息历史）
            {'messages': [AIMessage(content='', additional_kwargs={'updated_docs': {}}, response_metadata={}, id='38075a4e-00b0-4ea1-a1ee-4643a4296bc9', tool_calls=[{'name': 'PreferenceMemory', 'args': {'category': 'preference', 'context': 'UI settings', 'preference': 'light mode'}, 'id': '1c7266cc-f8bb-4e50-b950-8bbf23b38638', 'type': 'tool_call'}], invalid_tool_calls=[])], 'responses': [PreferenceMemory(category='preference', preference='light mode', context='UI settings')], 'response_metadata': [{'id': '1c7266cc-f8bb-4e50-b950-8bbf23b38638'}], 'attempts': 1}


)'''
    print(memories)
    print(memories[0][1])

asyncio.run(main())


```

```
output：
[ExtractedMemory(id='44daf91e-c07a-42f8-b23b-75d83537c5e0', content=Memory(content='{"type": "preference", "description": "User prefers dark mode in all applications. Confirmed explicitly with confidence 1.0. No exceptions or qualifications provided."\n\n**Context**: User stated "I prefer dark mode in all my apps" during session f1b2663a-0aee-4aa6-a764-9661a82583e8. AI acknowledged by stating "I\'ll remember that preference". No conflicting memories exist. This preference should inform interface rendering decisions across all platforms."'))]
content='{"type": "preference", "description": "User prefers dark mode in all applications. Confirmed explicitly with confidence 1.0. No exceptions or qualifications provided."\n\n**Context**: User stated "I prefer dark mode in all my apps" during session f1b2663a-0aee-4aa6-a764-9661a82583e8. AI acknowledged by stating "I\'ll remember that preference". No conflicting memories exist. This preference should inform interface rendering decisions across all platforms."'

```

2.结构化

```py
import asyncio
from pydantic import BaseModel
from langmem import create_memory_manager
class PreferenceMemory(BaseModel):
    """Store the user's preference"""
    category: str
    preference: str
    context: str
manager = create_memory_manager(
    "ollama:qwen3:8b",
    schemas=[PreferenceMemory]
)
conversation = [
    {
        "role": "user",
        "content": "I prefer dark mode in all my apps"
    },
    {
        "role": "assistant",
        "content": "I'll remember that preference"
    }
]

async def main():
    memories = await manager(conversation)
    print(memories)
    print("\n========== Structured Memory ==========\n")
    # memories[0]
    # -> (id, PreferenceMemory(...))
    memory_obj = memories[0][1]
    print(memory_obj)
    print("\n========== Fields ==========\n")
    print("category:", memory_obj.category)
    print("preference:", memory_obj.preference)
    print("context:", memory_obj.context)
asyncio.run(main())
```

```
output：
[ExtractedMemory(id='f860921e-1c01-4f47-9094-2970909f8e7c', content=PreferenceMemory(category='preference', preference='dark mode in all apps', context='interface settings'))]

========== Structured Memory ==========

category='preference' preference='dark mode in all apps' context='interface settings'

========== Fields ==========

category: preference
preference: dark mode in all apps
context: interface settings
```

第一个输出：ExtractedMemory(
    id='...',
    content=Memory(content='...')
)LangMem 没有使用自定义的 PreferenceMemory，而是退回到了默认 schema，对应源码：self.schemas = schemas if schemas is not None else (Memory,)，LLM 自己生成的一段 JSON 风格字符串，模型没有真正走 Tool Calling

第二个输出：

PreferenceMemory(
    category='preference',
    preference='dark mode in all apps',
    context='interface settings'
)

LLM 真正调用了 Pydantic Tool Schema

##### create_memory_manager的三种工作模式：

1.利用现有记忆：更新记忆

```py
import asyncio
from pydantic import BaseModel
from langmem import create_memory_manager
class PreferenceMemory(BaseModel):
    category: str
    preference: str
    context: str

async def main():
    manager = create_memory_manager(
        "ollama:qwen3:8b",
        schemas=[PreferenceMemory],
    )
    memories = [
        (
            "memory_1",
            PreferenceMemory(
                category="preference",
                preference="dark mode in all apps",
                context="interface settings",
            ),
        )
    ]
    conversation = [
        {
            "role": "user",
            "content": "Actually I changed my mind, dark mode hurts my eyes",
        },
        {
            "role": "assistant",
            "content": "I'll update your preference",
        },
    ]
    updated_memories = await manager.ainvoke(
        {
            "messages": conversation,
            "existing": memories,
        }
    )
    print("\n========== Updated Memories ==========\n")
    print(updated_memories)
asyncio.run(main())
```

```
output:

========== Updated Memories ==========

[ExtractedMemory(id='memory_1', content=PreferenceMemory(category='preference', preference='light mode in all apps', context='interface settings'))]
```

2.仅插入记忆：禁止更新删除enable_updates=False,enable_deletes=False,

```py
import asyncio
from pydantic import BaseModel
from langmem import create_memory_manager
class PreferenceMemory(BaseModel):
    category: str
    preference: str
    context: str

async def main():
    manager = create_memory_manager(
        "ollama:qwen3:8b",
        schemas=[PreferenceMemory],
        enable_updates=False,
        enable_deletes=False,
    )
    memories = [
        (
            "memory_1",
            PreferenceMemory(
                category="preference",
                preference="dark mode",
                context="UI settings",
            ),
        )
    ]
    conversation = [
        {
            "role": "user",
            "content": "Actually I changed my mind, light mode is the best mode",
        },
        {
            "role": "assistant",
            "content": "I'll update your preference",
        },
    ]

    updated_memories = await manager.ainvoke(
        {
            "messages": conversation,
            "existing": memories,
        }
    )
    print("\n========== Insert Only ==========\n")
    print(updated_memories)
asyncio.run(main())
```



```
output:

========== Insert Only ==========

[
ExtractedMemory(id='e851457e-76e1-4514-8840-a69a7c1f9d48', content=PreferenceMemory(category='preference', preference='light mode', context='UI settings')), 
ExtractedMemory(id='memory_1', content=PreferenceMemory(category='preference', preference='dark mode', context='UI settings'))
]
可以看出不能改也不能删旧记忆所以旧的 'memory_1'（dark mode）必须原封不动地保留下来。
必须新加一条：因为用户现在明确表达了完全相反的新偏好（light mode）
```

3.提供多个最大提取和合成步骤：

```
import asyncio
from pydantic import BaseModel
from langmem import create_memory_manager


class PreferenceMemory(BaseModel):
    category: str
    preference: str
    context: str


async def main():
    manager = create_memory_manager(
        "ollama:qwen3:8b",
        schemas=[PreferenceMemory],
    )

    conversation = [
        {
            "role": "user",
            "content": "I prefer dark mode in all my apps",
        },
        {
            "role": "assistant",
            "content": "I'll remember that preference",
        },
    ]

    max_steps = 3

    memories = await manager.ainvoke(
        {
            "messages": conversation,
            "max_steps": max_steps,
        }
    )

    print("\n========== Multi-step Extraction ==========\n")
    print(memories)


asyncio.run(main())
```

```
output：
========== Multi-step Extraction ==========

[ExtractedMemory(id='c224340c-9380-4f4d-95dd-fbf5953e86ac', content=PreferenceMemory(category='preference', preference='dark mode across all applications', context="user's app settings"))]

```
