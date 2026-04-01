from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class MyState(TypedDict):
    """
    定义图的状态：图在各个节点传递时，拥有一份共享全局数据变量
    【LangGraph 最新知识点】：
    1. `TypedDict` 是定义 State 的最基础方式。默认情况下，每个节点返回的字典会**覆盖** (Overwrite) 当前状态中对应的键。
    2. 如果你需要累加（比如把新消息 append 到消息列表中），需要使用 `typing.Annotated[list, operator.add]` (通常写作 `Annotated[list, add]`) 来声明该字段的更新合并策略。
    """
    input_text: str
    counter: int
    uppercase_text: str
    appended_text: str


def node_1_uppercase(state: MyState) -> MyState:
    """在这个节点，我们把文本转大写。"""
    # 打印看执行轨迹
    print(">>> 正在执行 node_1_uppercase ...")
    val = state["input_text"]
    return {"uppercase_text": val.upper()}


def node_2_append(state: MyState) -> MyState:
    """在这个节点，我们在末尾追加感叹号，并计数加一。"""
    print(">>> 正在执行 node_2_append ...")
    current_val = state.get("uppercase_text", "")
    current_counter = state.get("counter", 0)
    return {
        "appended_text": current_val + "!!!",
        "counter": current_counter + 1
    }


def main():
    print("=== 开始运行 LangGraph 基础流程（无大模型纯逻辑流） ===")
    # 1. 实例化图构造器
    workflow = StateGraph(MyState)

    # 2. 为图中添加节点（Node）
    workflow.add_node("step_uppercase", node_1_uppercase)
    workflow.add_node("step_append", node_2_append)

    # 3. 定义边（Edges），也就代表从哪里走到哪里
    # START -> step_uppercase -> step_append -> END
    workflow.add_edge(START, "step_uppercase")
    workflow.add_edge("step_uppercase", "step_append")
    workflow.add_edge("step_append", END)

    # 4. 编译这个 Graph，使之变成一个 Runnable
    app = workflow.compile()

    # 5. 调用它。就像 Langchain 的 LCEL 一样，我们可以 invoke
    initial_state = {
        "input_text": "hello langgraph",
        "counter": 0
    }
    print("初始状态:", initial_state)
    
    final_state = app.invoke(initial_state)
    
    print("\n[运行完成]")
    print("最终状态:", final_state)

if __name__ == "__main__":
    main()
