import importlib
from typing import Annotated, TypedDict
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

# 对应课程：7.2 Multi-Agent 架构分类及子图的通信模式
# 对应课程：7.3 父、子图状态中无共同键的通信方式
# 对应课程：8.3 GraphRAG 基本介绍与核心架构
# 对应课程：8.4 案例：Multi-Agent实现混合多知识库检索

# ======================================================================
# 【知识点 1：什么是子图 (Subgraph)？】
# 就像在编程里，如果一个函数太长，你会把它拆成几个小函数。
# 在 LangGraph 中，如果一个流程图（Agent）太复杂（比如既要查维基百科、又要查本地库、还要调取API），
# 我们可以把“检索本地库”单独写成一个完整的子图（Child Graph），然后把它当作一个节点（Node）
# 嵌入到主图（Parent Graph）里！这就是多智能体最优雅的解耦方式。
#
# 【知识点 2：父子图的通信】
# 子图有自己的 State 定义，父图有自己的 State 定义。它们可能字段名完全不一样！
# 解决办法：在父图中调用子图节点时，写一个转换函数（Mapper），把父图的参数提取出来塞给子图。
#
# 【知识点 3：什么是 GraphRAG？】
# 传统的 RAG（我们之前写的 FAISS）是向量检索：把文档切碎，算 Cosine 相似度。
# GraphRAG 则是构建知识图谱（Knowledge Graph）。
# 比如：实体（林客）-[属性：工作]->实体（程序员）-[关系：修复]->实体（十年祖传Bug）。
# 它更适合回答全局性的问题，比如“梳理一下这个项目中所有和网络请求相关的核心模块之间的调用链路”。
# 微软开源了强大的 GraphRAG 库，在企业级通常结合 Neo4j 图数据库使用。
# ======================================================================

# ==================== 构建子图 (Child Graph) ====================
# 假设这个子图是一个专门负责处理复杂 GraphRAG 检索的智能体。
class SubgraphState(TypedDict):
    search_query: str
    kg_results: list # 知识图谱检索结果

def kg_retrieval_node(state: SubgraphState):
    """模拟一个耗时的知识图谱（GraphRAG）检索过程"""
    query = state["search_query"]
    print(f"    [子图 - GraphRAG] 正在知识图谱中检索实体关系: '{query}'...")
    
    # 模拟从 Neo4j 中查到了复杂关系
    mock_result = f"实体 [{query}] --[相关联]--> [高频词汇：多智能体, 记忆, 子图]"
    return {"kg_results": [mock_result]}

def kg_synthesis_node(state: SubgraphState):
    """把图谱查到的碎片拼成一段话"""
    print(f"    [子图 - GraphRAG] 正在汇总图谱碎片...")
    res = "汇总：" + " | ".join(state["kg_results"])
    return {"kg_results": [res]}

# 编译子图
sub_workflow = StateGraph(SubgraphState)
sub_workflow.add_node("retrieve", kg_retrieval_node)
sub_workflow.add_node("synthesize", kg_synthesis_node)
sub_workflow.add_edge(START, "retrieve")
sub_workflow.add_edge("retrieve", "synthesize")
sub_workflow.add_edge("synthesize", END)
child_app = sub_workflow.compile()


# ==================== 构建父图 (Parent Graph) ====================
class ParentState(TypedDict):
    user_question: str
    final_answer: str

def parent_router_node(state: ParentState):
    print(f"\n[父图 - 主路由] 收到问题: {state['user_question']}")
    print("[父图 - 主路由] 决定调用 GraphRAG 子系统进行深度挖掘...")
    return {}

def call_subgraph_node(state: ParentState):
    """
    这就是对应 7.3 的“无共同键的通信方式”：
    父图的 State 里根本没有 `search_query`，子图的 State 里也没有 `user_question`。
    我们在这里手动做一个转换和桥接。
    """
    print("[父图 -> 子图] 正在跨层级通信，启动子图...")
    
    # 1. 父图状态提取 -> 子图状态输入
    sub_input = {"search_query": state["user_question"]}
    
    # 2. 调用子图 (invoke 子图就跟 invoke 一个普通的 Runnable 没有任何区别！)
    sub_result = child_app.invoke(sub_input)
    
    # 3. 子图状态输出 -> 父图状态更新
    final_answer = sub_result["kg_results"][0]
    print(f"[子图 -> 父图] 子图执行完毕，将结果送回黑板。")
    return {"final_answer": final_answer}

def main():
    print("=== 开始运行 Subgraph 子图通信与 GraphRAG 概念演示 ===")
    
    workflow = StateGraph(ParentState)
    workflow.add_node("router", parent_router_node)
    # 把封装好的通信节点作为普通的节点加进图里
    workflow.add_node("graph_rag_subsystem", call_subgraph_node)

    workflow.add_edge(START, "router")
    workflow.add_edge("router", "graph_rag_subsystem")
    workflow.add_edge("graph_rag_subsystem", END)

    parent_app = workflow.compile()

    # 运行主应用
    res = parent_app.invoke({"user_question": "LangGraph 的核心概念有哪些？"})
    
    print("\n[系统日志] 整个多代理嵌套流程跑完了！")
    print(f"最终输出给用户的答案:\n{res.get('final_answer')}")

if __name__ == "__main__":
    main()
