knowledge_tools 是 Lang Mem 的“长期记忆工具系统”将增删改查包装成LLM 可调用 Tool

供 Agent 自主使用。LLM Tool Schema 是一种结构化的 JSON 描述格式，用来告诉大语言模型"你可以调用哪些外部工具、每个工具能做什么、需要什么参数"。就是工具的函数签名说明书。当在代码里定义一个 Python 函数，框架会自动把函数名、参数类型、默认值和文字描述转换成 LLM 能看懂的 JSON 格式。LLM 拿到这个说明书后，会自己判断当前对话该不该调用工具、该填什么参数，然后输出一个标准化的函数调用请求，而不是直接回复文本。



两个核心

```py
create_manage_memory_tool()
create_search_memory_tool()
```

整体架构LLM--Tool Calling--manage_memory() search_memory()--Lang Graph Base Store--Vector Memory

##### 1.create_manage_memory_tool（）

创建一个用于管理持久化记忆的工具，支持创建、更新、删除操作。LLM 通过 Tool 来操纵长期记忆而不是手动存，生成的 Tool 类似于

```json
manage_memory(
    content="用户喜欢机器人",
    action="create"
)
manage_memory(
    id="memory-id",
    content="用户现在喜欢MPC",
    action="update"
)
...
```

##### 2.create_search_memory_tool()

这是Memory Retrieval Tool允许 Agent 主动搜索长期记忆

```py
memories = await store.asearch(
    namespace,
    query=query,
)
```

是向量检索，Agent 可以自主决定是否搜索 memory把 Python 函数转成 LLM Tool Schema，例如将def search_memory(query: str)转化为

```json
{
  "name": "search_memory",
  "parameters": {
    "query": "string"
  }
}
```

Memory 是 Agent 自主行为不是外挂数据库。

```py
def create_search_memory_tool(
    namespace: tuple[str, ...] | str,
    *,
    instructions: str = _MEMORY_SEARCH_INSTRUCTIONS,
    store: BaseStore | None = None,
    response_format: typing.Literal["content", "content_and_artifact"] = "content",
    name: str = "search_memory",
):
   #核心参数 instructions 定义了何时应该调用这个工具的指导语，这是实现自主决定是否搜索的关键。
    namespacer = utils.NamespaceTemplate(namespace)
    initial_store = store

    async def asearch_memory(
        query: str,
        *,
        limit: int = 10,
        offset: int = 0,
        filter: typing.Optional[dict] = None,
    ):
        store = _get_store(initial_store)
        namespace = namespacer()
        memories = await store.asearch(
            namespace,
            query=query,
            filter=filter,
            limit=limit,
            offset=offset,
        )
        if response_format == "content_and_artifact":
            return utils.dumps([m.dict() for m in memories]), memories
        return utils.dumps([m.dict() for m in memories])

    def search_memory(
        query: str,
        *,
        limit: int = 10,
        offset: int = 0,
        filter: typing.Optional[dict] = None,
    ):
        store = _get_store(initial_store)
        namespace = namespacer()
        memories = store.search(
            namespace,
            query=query,
            filter=filter,
            limit=limit,
            offset=offset,
        )
        if response_format == "content_and_artifact":
            return utils.dumps([m.dict() for m in memories]), memories
        return utils.dumps([m.dict() for m in memories])

    description = """Search your long-term memories for information relevant to your current context. {instructions}""".format(
        instructions=instructions
    )
    
    #构建工具的 description 字符串，将 instructions 嵌入其中。这个 description 会被放入 LLM 的 Tool Schema 的 description 字段。LLM 在每次决策时都会读取这个 description，根据其中的指导自主判断当前对话是否需要搜索记忆。

    return StructuredTool.from_function(
        search_memory,
        asearch_memory,
        name=name,
        description=description,
        response_format=response_format,
    )


def _get_store(initial_store: BaseStore | None = None) -> BaseStore:
    try:
        if initial_store is not None:
            store = initial_store
        else:
            store = get_store()
        return store
    except RuntimeError as e:
        raise errors.ConfigurationError("Could not get store") from e

```

description 会告诉 LLM例如当用户提到过去话题时，应该搜索长期记忆。LLM 看到用户发起类似你还记得我吗？的问题时就会思考：这个问题需要历史记忆--调用 search_memory，LLM不会直接调用而是输出{
  "tool": "search_memory",
  "args": {
    "query": "用户偏好"
  }
}

框架拿到 name: "search_memory 后，在自己的工具注册表里找到StructuredTool.from_function 注册的同名工具对象：然后框架执行 Tool，进入search_memory(query)，search_memory 函数本身不负责搜索，真正搜索发生在BaseStore即实例InMemoryStore.search()，

BaseStore:实现了：query embedding，memory embedding， similarity search（cosine similarity）功能，以及返回返回最像的 memory

https://github.com/langchain-ai/langgraph/blob/main/libs/checkpoint/langgraph/store/base/__init__.py?utm_source=chatgpt.com

```py
 return StructuredTool.from_function(
        search_memory,
        asearch_memory,
        name=name,
        description=description,
        response_format=response_format,
    )
```

`StructuredTool.from_function` 接收的 Python 函数后，自动提取元信息生成 JSON Schema：

**输入：**

```
def search_memory(
    query: str,
    *,
    limit: int = 10,
    offset: int = 0,
    filter: dict | None = None,
):
```

**最终生成的 Schema：**

```
{
  "type": "function",
  "function": {
    "name": "search_memory",
    "description": "Search your long-term memories for information relevant to your current context. 当用户问及之前讨论过的话题时应该调用...",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "Search query to match against memories"
        },
        "limit": {
          "type": "integer",
          "default": 10
        },
        "offset": {
          "type": "integer", 
          "default": 0
        },
        "filter": {
          "type": "object",
          "nullable": true
        }
      },
      "required": ["query"]
    }
  }
}
```

当创建 agent 时把工具传进去：

```
agent = create_react_agent(
    "anthropic:claude-3-5-sonnet-latest",
    tools=[search_tool],  # ← 这个 search_tool 是 StructuredTool 实例
    store=store,
)
```

LLM API 收到的请求体中就包含了完整的工具列表：

```
{
  "model": "claude-3-5-sonnet-latest",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant..."},
    {"role": "user", "content": "我之前说过我喜欢什么编程语言？"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "search_memory",
        "description": "Search your long-term memories...",
        "parameters": {...}
      }
    },
    {
      "type": "function",  
      "function": {
        "name": "manage_memory",
        "description": "Create, update, or delete memories...",
        "parameters": {...}
      }
    }
  ]
}
```