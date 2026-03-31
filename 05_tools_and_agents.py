from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
import importlib

setup_module = importlib.import_module("00_multi_model_setup")
get_model = setup_module.get_model

@tool
def multiply(a: int, b: int) -> int:
    """计算两个数字的乘积并返回"""
    return a * b

@tool
def get_weather(location: str) -> str:
    """获取指定城实的天气预报。如果不知道城市，就不应该调用。"""
    if "北京" in location:
        return "晴，20度，适合出门"
    else:
        return "多云，可能有小雨"

def main():
    llm = get_model("openai")

    # 1. 将工具绑定到 LLM 上
    tools = [multiply, get_weather]
    # bind_tools 是调用模型原生 Tool-Calling 能力的关键。这比早期 ReAct Prompt 解析要准得多。
    llm_with_tools = llm.bind_tools(tools)

    # 2. 编写 Agent 用的 Prompt
    # 注意：Agent 必定需要有 placeholder 来容纳 {agent_scratchpad}（它存放中间思考草稿）
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个拥有多种辅助工具的智能助理。遇到不会的问题或需要计算的功能，请优先调用工具。不要撒谎！"),
        ("user", "{input}"),
        ("placeholder", "{agent_scratchpad}"), 
    ])

    # 3. 创建预设引擎 (创建原生的基于模型 Tool-Calling API 支持的 Agent)
    agent = create_tool_calling_agent(llm, tools, prompt)

    # 4. 把 agent 和 真正的 python 逻辑（用来执行工具的执行器）打包在一起 -> AgentExecutor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    print("=== 开始运行 LangChain Tools & Agent ===")
    
    # 示例1：调用乘法工具
    print("\n[测试1: 乘法]")
    res1 = agent_executor.invoke({"input": "12乘以15等于多少？"})
    # 会看到 verbose 打印 LLM 先调用了 multiply 然后返回了 180
    print("[最终结果]:", res1['output'])
    
    # 示例2：调用天气工具
    print("\n[测试2: 天气]")
    res2 = agent_executor.invoke({"input": "北京今天天气怎么样？"})
    print("[最终结果]:", res2['output'])


if __name__ == "__main__":
    main()
