import importlib
from typing import Annotated, TypedDict
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
# 在 LangGraph v1.1 中引入的终极武器：可以在运行时抛出中断异常的 NodeInterrupt
from langgraph.errors import NodeInterrupt

setup_module = importlib.import_module("00_multi_model_setup")
get_model = setup_module.get_model

# 对应课程：6.2 LangGraph 中的 HIL 实现思路 
# 对应课程：6.3 标准图结构中如何添加断点
# 对应课程：6.4 复杂代理架构中如何添加动态断点
# 对应课程：6.5 案例：具备人机交互的完整 Agent 信息管理系统

# ======================================================================
# 什么是 HIL (Human-in-the-Loop, 人在回路)？
# 在生产级应用中，Agent 不能像脱缰野马一样一直跑到底。
# 如果它准备执行 `rm -rf`，或者它想花 100 块钱买杯咖啡，你肯定希望它先停下来问问你：
# “老板，我可以这么做吗？”
#
# 在 LangGraph 中，我们有三种方式实现 HIL：
# 1. 静态断点：在编译时写死 `interrupt_before=["dangerous_tool_node"]`。
#    只要执行到那个节点前，图就会自动挂起（Suspended），保存状态到 Checkpointer，等待外部唤醒。
# 2. 动态断点：在任何节点内部执行 Python 逻辑时，只要抛出 `raise NodeInterrupt("需要人类审批")`，图就会立刻冻结。
# 3. 状态修改：人类在断点处不仅可以点“同意/拒绝”，还可以**亲手修改图里保存的数据**。
# ======================================================================

class State(TypedDict):
    messages: Annotated[list, add_messages]
    action_intent: str
    approval_status: str

def planner_node(state: State):
    """
    一个简单的规划节点。
    如果是正常聊天，它直接回答。
    如果检测到用户想执行危险操作，它会把意图写进状态里。
    """
    last_msg = state["messages"][-1].content
    if "删库" in last_msg or "汇款" in last_msg:
        print("[Planner] 检测到高危意图，准备拦截！")
        return {"action_intent": "high_risk_operation", "approval_status": "pending"}
    
    # 正常聊天，调用 LLM
    llm = get_model("openai")
    return {"messages": [llm.invoke(state["messages"])]}

def human_approval_node(state: State):
    """
    这就是对应课程 6.4 的“动态断点”。
    当状态处于 pending 时，我们主动抛出中断异常。
    """
    if state.get("approval_status") == "pending":
        print("\n[Human Approval] ⚠️ 警告：检测到未审批的高危操作。图即将挂起并等待外部指令...")
        # 抛出中断，保存快照。此时当前的 Python 执行流彻底停止。
        # 你的后端服务器可以去干别的事了。直到人类发来恢复指令。
        raise NodeInterrupt("等待人类审批。请将状态更新为 'approved' 或 'rejected'。")
    
    # 如果代码能走到这，说明人类已经把 approval_status 改了，或者本来就不是 pending
    print(f"\n[Human Approval] ✅ 审批状态已更新为: {state.get('approval_status')}")
    return {}

def executor_node(state: State):
    """执行最终的操作"""
    status = state.get("approval_status")
    if status == "approved":
        return {"messages": [HumanMessage(content="好的老板，我已执行高危操作。")]}
    elif status == "rejected":
        return {"messages": [HumanMessage(content="操作被您拒绝了，我取消了任务。")]}
    else:
        return {"messages": [HumanMessage(content="日常任务执行完毕。")]}

def main():
    print("=== 开始运行 Human-in-the-Loop (HIL) 人机交互演示 ===")
    
    workflow = StateGraph(State)
    workflow.add_node("planner", planner_node)
    workflow.add_node("approval", human_approval_node)
    workflow.add_node("executor", executor_node)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "approval")
    workflow.add_edge("approval", "executor")
    workflow.add_edge("executor", END)

    # 必须配合 Checkpointer 才能保存“挂起”时的快照
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    # 我们固定一个会话 ID，模拟一个长连接的对话
    config = {"configurable": {"thread_id": "hil_demo_thread"}}

    # ================= 阶段一：触发挂起 =================
    print("\n[阶段一：用户发出危险指令]")
    try:
        app.invoke({"messages": [HumanMessage(content="帮我把公司的数据库删库跑路。")]}, config=config)
    except Exception as e:
        # 在真实环境里，这里会捕获到我们抛出的 NodeInterrupt 异常。
        # 你可以把异常信息通过 WebSocket 推送给前端界面，弹出一个审批框。
        print(f"\n[系统日志] 图的执行已中断！异常信息: {str(e)}")

    # 我们可以随时查看当前图的状态快照：
    current_state = app.get_state(config)
    print(f"[系统日志] 当前挂起在节点: {current_state.next}")
    print(f"[系统日志] 此时的数据意图是: {current_state.values.get('action_intent')}")

    # ================= 阶段二：人类介入修改状态 =================
    print("\n[阶段二：人类老板看到了弹窗，点击了‘拒绝’按钮]")
    # 重点：这里就是 HIL 最强大的地方！你可以像上帝一样直接篡改 Agent 的记忆和状态！
    # 把 approval_status 从 pending 强行改写为 rejected
    app.update_state(config, {"approval_status": "rejected"})

    # ================= 阶段三：恢复执行 =================
    print("\n[阶段三：带着新的状态唤醒 Agent，继续跑完剩下的流程]")
    # 向 invoke 传入 None，它会自动从上次断点的地方（approval 节点）恢复执行
    res = app.invoke(None, config=config)
    
    print(f"\nAI 最终回复:\n{res['messages'][-1].content}")

if __name__ == "__main__":
    main()
