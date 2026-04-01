from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
import importlib

setup_module = importlib.import_module("00_multi_model_setup")
get_model = setup_module.get_model

class RouterState(TypedDict):
    question: str
    classification: str
    response: str


def classification_node(state: RouterState) -> RouterState:
    """
    调用大模型判断用户的问题是关于什么类型的。
    这里用非常生硬的Prompt限制它只输出特定的单词： 'math' 或 'general'。
    """
    print(">>> classification_node 正在工作...")
    llm = get_model("openai", temperature=0)
    prompt = f"请把以下问题分类为两类之一返回：'math'(数学相关), 'general'(通用或其他无聊问题)。只能输出这俩词！\n用户问题: {state['question']}"
    
    result = llm.invoke(prompt)
    classification = result.content.strip().lower()
    return {"classification": classification}


def math_node(state: RouterState) -> RouterState:
    print(">>> 进入专门处理数学的分支...")
    llm = get_model("openai", temperature=0)
    # 给模型一点具体的系统指令角色
    res = llm.invoke([
        ("system", "你是一个数学专家，请用数学严谨的语言解释。"),
        ("user", state["question"])
    ])
    return {"response": res.content}


def general_node(state: RouterState) -> RouterState:
    print(">>> 进入处理闲聊的通用分支...")
    llm = get_model("openai", temperature=0.7)
    res = llm.invoke([
        ("system", "你是一个幽默的陪聊。回答要风趣。"),
        ("user", state["question"])
    ])
    return {"response": res.content}


# 定义路由于边的逻辑
def route_question(state: RouterState) -> Literal["math_node", "general_node"]:
    """通过读取 state 中的字段，返回接下来要执行的下个节点的名称"""
    if "math" in state["classification"]:
        return "math_node"
    return "general_node"


def main():
    print("=== 开始运行 LangGraph 条件路由器并测试分支控制法 ===")
    workflow = StateGraph(RouterState)

    workflow.add_node("classifier", classification_node)
    workflow.add_node("math_node", math_node)
    workflow.add_node("general_node", general_node)

    # 开始节点走向分类器
    workflow.add_edge(START, "classifier")

    # 条件边：从分类器出来后，根据 route_question 函数返回的字符串走向不同节点
    # 【LangGraph 最新知识点】：
    # 1. 在最新版本的 LangGraph 中，如果你的路由函数 `route_question` 返回的字符串
    #    刚好就是要跳转的节点的名称，那么你可以省略第三个参数（映射字典）。
    # 2. 这种按名称直接路由的约定俗成（Convention over Configuration）大大简化了代码。
    workflow.add_conditional_edges(
        "classifier",       # 判断起点
        route_question      # 用于判断的逻辑方法
    )

    # 将两个子处理节点的出口都汇聚到 END
    workflow.add_edge("math_node", END)
    workflow.add_edge("general_node", END)

    app = workflow.compile()

    print("\n[测试1：数学问题]")
    res1 = app.invoke({"question": "欧拉公式是什么？为什么它这么美？"})
    print("分支判定为:", res1["classification"])
    print("AI 最终回复:", res1["response"])

    print("\n[测试2：闲聊问题]")
    res2 = app.invoke({"question": "今天天气不错，我们去干嘛？"})
    print("分支判定为:", res2["classification"])
    print("AI 最终回复:", res2["response"])

if __name__ == "__main__":
    main()
