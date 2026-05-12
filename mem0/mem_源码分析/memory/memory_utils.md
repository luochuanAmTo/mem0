| 功能类别                | 包含的主要函数                                               | 核心作用                                                     |
| :---------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| **1. LLM 交互与提示词** | `get_fact_retrieval_messages` `get_fact_retrieval_messages_legacy` `ensure_json_instruction` | 为不同场景（用户/智能体）组装发送给大模型的提示词，并**确保模型按要求返回JSON格式**。 |
| **2. 事实与实体清洗**   | `normalize_facts` `format_entities` `remove_spaces_from_entities` `sanitize_relationship_for_cypher` | **归一化**大模型返回的各种非标准格式（如事实有时是字符串，有时是对象），并将实体关系数据**格式化为图数据库可用的标准形式**。 |
| **3. 消息与格式解析**   | `parse_messages` `parse_vision_messages` `get_image_description` | 将 Mem0 内部的多模态消息列表转换为纯文本摘要（特别是把图片转成文字描述），以便后续记忆提取。 |
| **4. 内容提取与安全**   | `remove_code_blocks` `extract_json` `process_telemetry_filters` | 从混合文本中**提取纯净的JSON**，剔除代码块标记等干扰，并对遥测数据中的敏感ID进行**哈希脱敏**。 |



**`get_fact_retrieval_messages(message, is_agent_memory) -> tuple`**记忆提取的**入口分派器**根据 `is_agent_memory` 标志，选择不同的系统提示词（`AGENT_MEMORY_EXTRACTION_PROMPT` 或 `USER_MEMORY_EXTRACTION_PROMPT`）

```py

def get_fact_retrieval_messages(message, is_agent_memory=False):
    """Get fact retrieval messages based on the memory type.
    
    Args:
        message: The message content to extract facts from
        is_agent_memory: If True, use agent memory extraction prompt, else use user memory extraction prompt
        
    Returns:
        tuple: (system_prompt, user_prompt)
    """
    if is_agent_memory:
        return AGENT_MEMORY_EXTRACTION_PROMPT, f"Input:\n{message}"
    else:
        return USER_MEMORY_EXTRACTION_PROMPT, f"Input:\n{message}"
```

**`ensure_json_instruction(system_prompt, user_prompt) -> tuple`**OpenAI要求当`response_format`设为`json_object`时，提示词中必须包含单词 `json`。该函数会检查，如果用户自定义提示词中遗漏了此词，就自动追加一条JSON格式指令。

```Python
def ensure_json_instruction(system_prompt, user_prompt):
    """Ensure the word 'json' appears in the prompts when using json_object response format.

    OpenAI's API requires the word 'json' to appear in the messages when
    response_format is set to {"type": "json_object"}. When users provide a
    custom_instructions that doesn't include 'json', this causes a
    400 error. This function appends a JSON format instruction to the system
    prompt if 'json' is not already present in either prompt.

    Args:
        system_prompt: The system prompt string
        user_prompt: The user prompt string

    Returns:
        tuple: (system_prompt, user_prompt) with JSON instruction added if needed
    """
    combined = (system_prompt + user_prompt).lower()
    if "json" not in combined:
        system_prompt += (
            "\n\nYou must return your response in valid JSON format "
            "with a 'facts' key containing an array of strings."
        )
    return system_prompt, user_prompt


```

**`normalize_facts(raw_facts) -> list小模型可能返回 [{"fact": "..."}, {"text": "..."}] 这样的对象数组，而非简单的字符串列表。该函数能智能提取值并归一化为统一的字符串列表。隔离了LLM的异构输出，让下游代码只需处理干净、确定的字符串列表。`**

```
def normalize_facts(raw_facts):
    """Normalize LLM-extracted facts to a list of strings.

    Smaller LLMs (e.g. llama3.1:8b) sometimes return facts as objects
    like {"fact": "..."} or {"text": "..."} instead of plain strings.
    This mirrors the TypeScript FactRetrievalSchema validation.
    """
    if not raw_facts:
        return []
    normalized = []
    for item in raw_facts:
        if isinstance(item, str):
            fact = item
        elif isinstance(item, dict):
            fact = item.get("fact") or item.get("text")
            if fact is None:
                logger.warning("Unexpected fact shape from LLM, skipping: %s", item)
                continue
        else:
            fact = str(item)
        if fact:
            normalized.append(fact)
    return normalized
```

**`parse_vision_messages(messages, llm, vision_details) -> list遍历消息列表，识别出所有包含图像的内容（支持列表或多图），然后调用视觉模型将其统一转换为文本描述。`****实现了多模态到纯文本的降维**。经它处理后，后续的记忆提取流程就无需关心原始输入是图还是文，只需处理统一的文本摘要。