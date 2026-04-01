import importlib
from typing import Annotated, Literal, Sequence, TypedDict
import operator

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

setup_module = importlib.import_module("00_multi_model_setup")
get_model = setup_module.get_model

# ======================================================================
# 对应课程：7.1 Single-Agent 存在的局限
# 为什么不用一个 Agent 搞定所有事？
# 1. 角色冲突：如果它既要写代码，又要写测试，还要写文档，提示词（Prompt）会非常长且难以维护。
# 2. Token 爆炸：几万行的聊天记录会让单体模型产生严重幻觉（Hallucination）。
#
# 对应课程：8.1 Supervisor 架构介绍与基本构建原理
# Supervisor (主管) 模式是企业级多智能体最常见的架构：
# 它类似于一个包工头。Supervisor 不干具体活，它只负责看一眼总任务，
# 然后把活派给底下的 Worker（如 Coder、Researcher）。Worker 干完活后，
# 把结果汇报给 Supervisor，Supervisor 检查没问题后，再决定是结束还是派给下一个 Worker。
# ======================================================================

# 1. 定义全局状态 (Global State)
# Supervisor 架构的特点是：所有子节点都读写同一个黑板（State）。
# operator.add 表示消息会被追加，而不是覆盖。
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str  # 记录 Supervisor 决定下一个该轮到谁执行

# 2. 定义 Supervisor 节点
def supervisor_node(state: AgentState) -> dict:
    """
    主管节点：它唯一的任务就是读取黑板上的历史对话，然后决定下一步该怎么办。
    它不使用普通的输出，而是使用 with_structured_output 强制输出枚举值。
    """
    print("\n[Supervisor 思考中...] 正在审查任务进度...")
    llm = get_model("openai", temperature=0)

    # 定义可以路由的节点名称
    members = ["Researcher", "Coder"]
    
    # 强制模型输出的结构：只能是 members 之一，或者是 FINISH
    class Router(BaseModel):
        next: Literal[*members, "FINISH"] = Field(
            description="下一个执行的工人，或者如果任务已经完成，输出 FINISH。"
        )
    
    system_prompt = (
        "你是这个团队的主管。你的团队有以下成员：{members}。"
        "根据用户的请求和当前进度，指定下一个应该行动的工人。"
        "Researcher 负责上网搜集资料和整理思路。"
        "Coder 负责写代码。"
        "如果任务已经完全完成，请回复 FINISH。"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "根据以上对话，接下来该谁干活了？")
    ]).partial(members=", ".join(members))

    # 构建并调用强制路由链
    supervisor_chain = prompt | llm.with_structured_output(Router)
    
    # 获取主管的决定
    response = supervisor_chain.invoke(state)
    print(f"  -> 主管决定派发给: {response.next}")
    
    # 将决定写回状态黑板
    return {"next": response.next}

# 3. 定义 Worker 节点 (这里为了演示简单，我们用普通的 LLM 调用模拟)
def researcher_node(state: AgentState) -> dict:
    """研究员节点：负责搜集资料。"""
    print("\n[Worker - Researcher 正在干活...]")
    llm = get_model("openai")
    
    # 给工人带上专属帽子
    messages = [
        ("system", "你是一名高级研究员。请用一句话简要总结你需要搜集的技术概念。不要废话。")
    ] + state["messages"]
    
    response = llm.invoke(messages)
    # 把结果写回黑板，并在前面打上自己的名字标签，方便主管知道是谁说的
    return {"messages": [HumanMessage(content=f"[Researcher 的汇报]: {response.content}")]}

def coder_node(state: AgentState) -> dict:
    """程序员节点：负责写代码。"""
    print("\n[Worker - Coder 正在干活...]")
    llm = get_model("openai")
    
    messages = [
        ("system", "你是一名高级工程师。请根据 Researcher 提供的资料，用一行 Python 代码实现核心逻辑。")
    ] + state["messages"]
    
    response = llm.invoke(messages)
    return {"messages": [HumanMessage(content=f"[Coder 的汇报]: {response.content}")]}


def main():
    print("=== 开始运行 对应 8.2 案例的 Multi-Agent Supervisor 架构演示 ===")
    
    # 4. 构建主管图 (Graph)
    workflow = StateGraph(AgentState)

    # 添加所有的节点
    workflow.add_node("Supervisor", supervisor_node)
    workflow.add_node("Researcher", researcher_node)
    workflow.add_node("Coder", coder_node)

    # 5. 定义路由逻辑
    # 所有的 Worker 干完活后，边全部指回 Supervisor（这就是典型的星型网络拓扑）
    workflow.add_edge("Researcher", "Supervisor")
    workflow.add_edge("Coder", "Supervisor")

    # Supervisor 的边是条件路由：它根据 state["next"] 的值决定走向
    workflow.add_conditional_edges(
        "Supervisor",
        lambda x: x["next"],
        {
            "Researcher": "Researcher",
            "Coder": "Coder",
            "FINISH": END
        }
    )

    # 整个图的入口必须是主管
    workflow.add_edge(START, "Supervisor")

    # 编译成可执行的图
    app = workflow.compile()

    # 6. 测试多智能体协作
    user_task = "我要写一个快速排序算法，你先帮我查查它的核心原理，然后再写代码。"
    print(f"\n[用户布置任务]: {user_task}")
    
    # 使用 stream_mode="updates" 观察每个节点产生的新状态
    for s in app.stream(
        {"messages": [HumanMessage(content=user_task)]},
        # 设置递归上限，防止 Agent 们陷入死循环一直互相推诿
        config={"recursion_limit": 10}
    ):
        if "__end__" not in s:
            # 打印当前是哪个节点产出了数据
            print(f"\n[系统日志] 当前执行节点: {list(s.keys())[0]} 完成了工作。")

if __name__ == "__main__":
    main()
