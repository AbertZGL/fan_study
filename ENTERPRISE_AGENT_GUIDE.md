# 从入门到企业级：开发大型 AI 编码助手（如 Claude Code / OpenHands）指南

你敏锐地察觉到了**“Demo 玩具”**和**“企业级生产力工具”**之间的巨大鸿沟。

之前我们学习的 LangChain / LangGraph 示例，展示了**大脑（LLM）**如何通过**神经（Graph/LCEL）**去调用**手脚（Tools）**。但这就像是造了一个会算加减法、会查天气的机器人。

如果你要开发像 **Claude Code**、**GitHub Copilot Workspace**、**OpenHands** (前 OpenDevin) 或 **SWE-agent** 这种级别的 AI 编码项目，单纯依靠基础的 ReAct 循环是远远不够的。你需要解决以下几个核心的工程难题，并补充相应的系统架构。

---

## 核心差距 1：安全且状态隔离的执行沙盒 (Sandbox)

**现状不足**：我们在 Demo 中让 AI 执行本地的 Python 函数（如 `heavy_computation`），这是极度危险的。如果 AI 生成了 `os.system("rm -rf /")`，你的电脑就完了。
**企业级方案**：AI 编码助手的底层必须是一个**隔离的沙盒环境**（通常是 Docker 容器、gVisor 或专门的沙盒服务如 E2B / Daytona）。AI 的一切文件读写、Bash 命令、依赖安装，都必须在沙盒内进行。

### 补充代码示例：实现一个基础的 Docker Sandbox 工具

```python
import docker
from langchain_core.tools import tool

class DockerSandbox:
    def __init__(self, image="python:3.10-slim"):
        self.client = docker.from_env()
        # 启动一个常驻的后台容器作为工作区
        self.container = self.client.containers.run(
            image, 
            command="tail -f /dev/null", 
            detach=True, 
            working_dir="/workspace"
        )

    def execute_command(self, cmd: str) -> str:
        """在沙盒内执行 Bash 命令并返回输出"""
        exit_code, output = self.container.exec_run(cmd)
        if exit_code != 0:
            return f"Error: {output.decode('utf-8')}"
        return output.decode('utf-8')

# 实例化沙盒
sandbox = DockerSandbox()

@tool
def run_bash_in_sandbox(command: str) -> str:
    """
    执行终端命令，例如 ls, cat, pip install 等。
    注意：这是真实执行，不是模拟！
    """
    return sandbox.execute_command(command)
```

---

## 核心差距 2：仓库级上下文感知 (Repo-level RAG & AST)

**现状不足**：基础 RAG（如 FAISS + TextSplitter）切分代码会破坏逻辑结构。一段代码被从中间切断，AI 就看不懂了。且把几万行代码塞进 Prompt 会导致 Context Window 爆炸。
**企业级方案**：
1. **基于 AST (抽象语法树) 的解析**：使用 `Tree-sitter` 解析代码库，提取出所有的类名、函数签名、Docstring，构建出一份“代码地图 (Code Map)”。
2. **语义搜索 + 符号搜索 (Symbol Search)**：AI 需要像 IDE 一样，拥有“跳转到定义 (Go to Definition)”和“查找所有引用 (Find References)”的工具。
3. **基于 LSIF / ctags 的静态分析工具**。

### 补充代码示例：为 AI 提供精确的 Ripgrep 搜索工具

```python
import subprocess
from langchain_core.tools import tool

@tool
def search_codebase(pattern: str, dir_path: str = ".") -> str:
    """
    使用 ripgrep (rg) 在整个代码库中进行高性能的正则搜索。
    当你想找到某个函数在哪里被调用，或者某个变量在哪里定义时使用。
    """
    try:
        # rg -n 显示行号, -C 2 显示上下2行上下文
        result = subprocess.run(
            ["rg", "-n", "-C", "2", pattern, dir_path],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.stdout if result.stdout else "未找到匹配项。"
    except Exception as e:
        return f"搜索失败: {str(e)}"
```

---

## 核心差距 3：精确的代码修改能力 (Patch & Diff Application)

**现状不足**：Demo 中 AI 每次回答都是把整段代码重新打印一遍。如果是 2000 行的文件，让 AI 重写一遍既慢又极易出错（经常把没改的代码写错）。
**企业级方案**：AI 必须学会**“打补丁 (Patch)”**。像 SWE-agent 这样的项目，会提供极其精细的文件编辑工具，比如：
- `view_file(path, start_line, end_line)`：只看某几行。
- `edit_file(path, search_string, replace_string)`：用精确的字符串替换，或者标准的 Unified Diff 格式来修改代码。

### 补充代码示例：基于块替换的代码编辑工具

```python
from langchain_core.tools import tool

@tool
def replace_code_block(file_path: str, old_code: str, new_code: str) -> str:
    """
    替换文件中的指定代码块。
    要求 old_code 必须与文件中的原始内容完全一致（包括缩进）。
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        if old_code not in content:
            return "错误：在文件中找不到完全匹配的 old_code。请先使用 view_file 工具确认内容。"
            
        updated_content = content.replace(old_code, new_code, 1)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(updated_content)
            
        return f"成功：{file_path} 已更新。"
    except Exception as e:
        return f"文件修改失败：{str(e)}"
```

---

## 核心差距 4：多智能体协作与长程规划 (Multi-Agent & Planning)

**现状不足**：一个单体 Agent 既要做计划，又要写代码，还要跑测试，容易陷入死循环或幻觉（Hallucination）。
**企业级方案**：采用**多智能体架构 (Multi-Agent Architecture)**。通常至少包含：
1. **Planner (规划师)**：不写代码，只看需求，拆解出 Step 1, Step 2, Step 3。
2. **Coder / Executor (执行者)**：只负责完成 Planner 交给的当前 Step，调用工具写代码。
3. **Reviewer (审查者)**：运行测试（`pytest`），如果报错，把报错信息喂回给 Coder。

### 补充代码示例：LangGraph 多智能体路由架构设计

```python
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage
import operator

# 定义包含多个角色的全局状态
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    plan: List[str]          # 当前的任务计划
    current_step: str        # 当前正在执行的步骤
    code_diff: str           # 生成的代码补丁
    test_results: str        # 测试运行结果

def planner_node(state: AgentState):
    """规划师节点：生成计划清单"""
    pass

def coder_node(state: AgentState):
    """程序员节点：调用 Bash, Git, EditFile 工具修改代码"""
    pass

def tester_node(state: AgentState):
    """测试节点：运行 npm test 或 pytest"""
    pass

def reviewer_router(state: AgentState) -> str:
    """条件路由：根据测试结果决定走向"""
    if "FAIL" in state["test_results"]:
        return "coder"  # 打回重写
    elif len(state["plan"]) > 0:
        return "planner" # 继续下一个计划
    else:
        return "end"     # 全部完成

# 构建企业级多智能体工作流
workflow = StateGraph(AgentState)
workflow.add_node("planner", planner_node)
workflow.add_node("coder", coder_node)
workflow.add_node("tester", tester_node)

workflow.add_edge(START, "planner")
workflow.add_edge("planner", "coder")
workflow.add_edge("coder", "tester")
workflow.add_conditional_edges("tester", reviewer_router, {
    "coder": "coder",
    "planner": "planner",
    "end": END
})
```

---

## 核心差距 5：人类介入 (Human-in-the-loop) 与回溯 (Time Travel)

**现状不足**：Demo 一按回车就跑到底。但在真实项目中，AI 可能会说：“我要重构整个鉴权模块，并 `git push -f`”。
**企业级方案**：
必须利用 LangGraph 的 `checkpointer` 与 `interrupt_before` 功能。在执行危险工具（修改文件、提交代码）前，强制暂停图的执行，将当前的 Diff 展示给用户（UI 层），用户点击 "Approve" 或输入修改意见后，Graph 再继续运行。

```python
# 编译时设置人类拦截点
app = workflow.compile(
    checkpointer=memory,
    interrupt_before=["tester"] # 在运行测试（或修改代码前）暂停，等待人类审批
)
```

---

## 核心差距 6：标准化的工具生态与上下文协议 (MCP & Skills)

**现状不足**：在之前的 Demo 中，我们通过写 Python 函数并加 `@tool` 装饰器来给大模型提供工具。但在企业级应用中：
1. 工具（如查数据库、查 Jira、查网页）往往是由其他团队使用不同语言开发的，无法直接 import Python 函数。
2. 随着功能增加，我们不可能把几百个工具一次性塞给大模型（会导致 Context Window 溢出且模型选择困难）。

**企业级方案**：
引入 **MCP (Model Context Protocol)** 和 **动态技能系统 (Skills System)**。

### 澄清：什么是 Tool？什么是 Skill？

在初学时，大家经常把 Tool（工具）和 Skill（技能）混为一谈。但在像 Trae / Claude Code 这样的大型系统中，它们有严格的层级划分：

*   **Tool (工具)**：是最底层的**执行单元**。它通常是一个具体的 Python 函数或 API 调用，比如 `extract_text_from_pdf` 或 `run_bash_command`。它只负责“干活”，没有独立意识。
*   **Skill (技能)**：是一个更高维度的**业务能力包 (Capability Package)**。一个 Skill 往往包含：
    1.  **元数据描述 (Metadata)**：比如 `SKILL.md`，用极其精炼的语言告诉 Agent 这个技能的作用和**触发时机**（何时该用它）。
    2.  **一个或多个 Tools**：为了完成这个技能，底层可能打包了多个 Tool。
    3.  **动态挂载机制 (Dynamic Discovery)**：Agent 不会在启动时硬编码 `tools=[t1, t2, t3]`，而是根据用户当前的任务意图（Intent），去 `.trae/skills/` 目录下**动态扫描并加载**相关的 Skill。

---

*   **MCP (Model Context Protocol)**：由 Anthropic (Claude 团队) 提出的开源协议。它像是一个“大模型届的 USB 接口”，允许你的 Agent 通过标准化的协议（基于 stdio 或 SSE）去连接本地或远程的独立工具服务，而不需要改动 Agent 本身的代码。
*   **Skills (技能挂载)**：将特定领域的知识和工具链打包成一个“技能插件”。Agent 在运行时，通过意图识别（Intent Detection）动态地只加载当前任务需要的 Skill。

### 补充代码示例：基于 MCP 协议接入远程工具服务

在这个示例中，我们将演示如何让 LangChain / LangGraph Agent 作为一个 **MCP Client**，去连接一个独立运行的 **MCP Server**。

```python
import asyncio
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain_core.tools import tool
import importlib

# 假设我们已经配好了多模型
setup_module = importlib.import_module("00_multi_model_setup")
get_model = setup_module.get_model

# =====================================================================
# 模拟 MCP Server 端的工具 (在真实世界中，这可能是一个用 Node.js 写的独立进程)
# =====================================================================
@tool
def mcp_fetch_jira_ticket(ticket_id: str) -> str:
    """[MCP Server 提供] 获取 Jira 工单的详细信息"""
    print(f"\n  [MCP Server] 正在通过 HTTP 请求拉取 Jira 工单 {ticket_id}...")
    return f"工单 {ticket_id} 描述：前端页面在移动端显示错位。状态：To Do"

# =====================================================================
# MCP Client 端 (你的主 Agent)
# =====================================================================
async def run_mcp_agent():
    print("=== 开始运行 MCP 架构演示 ===")
    llm = get_model("openai")
    
    # 1. 动态技能加载 (Skill Loading)
    # 在真实的大型系统中，这里不会硬编码。
    # 而是根据用户的输入 "帮我修一下 Jira 上的 BUG"，
    # 使用一个小型的意图识别模型，去检索并加载 "Jira Skill" 包。
    user_intent = "jira_debugging"
    
    active_tools = []
    if user_intent == "jira_debugging":
        print("[系统] 检测到 Jira 调试意图，正在通过 MCP 协议挂载 Jira 工具集...")
        # 实际开发中，这里会通过 MCP Client 连接到 MCP Server 获取工具 Schema
        active_tools.append(mcp_fetch_jira_ticket)
    
    # 2. 只有当前任务需要的工具被传给了大模型，极大地节省了 Token 并提高了准确率
    agent_executor = create_agent(
        model=llm, 
        tools=active_tools,
        system_prompt="你是一个高级研发工程师，你需要根据工单信息修复代码。"
    )

    # 3. 执行任务
    user_input = "请帮我看看 PROJ-123 这个工单是怎么回事？"
    print(f"\nUser: {user_input}")
    
    res = agent_executor.invoke({"messages": [HumanMessage(content=user_input)]})
    print(f"AI: {res['messages'][-1].content}")

if __name__ == "__main__":
    asyncio.run(run_mcp_agent())
```

### MCP 架构带来的革命性优势：
1. **解耦**：你的主 Agent 永远是用 Python/LangGraph 写的，但你的文件读取工具可以是用 Rust 写的（追求极速），你的数据库查询工具可以是用 TypeScript 写的。它们之间通过 MCP 协议通信。
2. **安全隔离**：你可以把危险的工具（如执行 Shell）放在一个隔离的 Docker 容器里运行 MCP Server，而主 Agent 跑在宿主机上，即使大模型被“提示词注入”攻击，也只能在沙盒里搞破坏。
3. **生态复用**：像 Cursor、Claude Code 等产品都已经支持 MCP。只要你写了一个好用的 MCP Server（比如专门用来操作 AWS 的工具），所有支持 MCP 的 AI 助手都可以无缝插拔使用它。

## 总结：如果你要造一个 OpenHands，你需要补充的技术栈：

1. **底层设施**：Docker SDK, gVisor, PTY (伪终端交互，用于获取命令行的实时输出)。
2. **代码理解**：Tree-sitter (AST 解析), LSIF, LSP (语言服务器协议，让 AI 获取代码补全和错误提示)。
3. **框架进阶**：深入掌握 LangGraph 的 Multi-Agent (如 Supervisor 模式, Hierarchical Agent 模式)。
4. **评测基准**：集成 SWE-bench，使用大量的真实 Github Issue 来评测你的 Agent 解决 Bug 的能力。
5. **前端交互**：类似 Cursor / Trae 的左侧对话框 + 右侧代码差异对比 (Diff View) 的 UI 界面。