import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

# =========================================
# 多模型接入示例：如何初始化不同厂商的大模型
# 【LangChain v0.2/v0.3 最新知识点】：
# 1. 架构解耦：以前所有的集成都在 `langchain` 或 `langchain-community` 中，
#    现在官方推荐使用独立的厂商包（Partner Packages），例如 `langchain-openai`, `langchain-anthropic`。
# 2. ChatModel 成为主流：早期的 `LLM` 类（纯文本补全）已逐渐被边缘化，
#    现在的核心是 `BaseChatModel`（基于角色的消息列表传递），例如 `ChatOpenAI`, `ChatAnthropic`。
# =========================================

def get_openai_model(temperature=0):
    """
    OpenAI 模型初始化示例
    依赖：pip install langchain-openai
    需要环境变量：OPENAI_API_KEY
    """
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model="gpt-4o-mini", temperature=temperature)


def get_deepseek_model(temperature=0):
    """
    DeepSeek 模型初始化示例 (通过 OpenAI 兼容接口)
    依赖：pip install langchain-openai
    需要环境变量：DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL (通常为 https://api.deepseek.com)
    """
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model="deepseek-chat", 
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        temperature=temperature
    )


def get_zhipu_model(temperature=0):
    """
    智谱 AI (GLM) 模型初始化示例 (通过 OpenAI 兼容接口)
    注意：智谱 V4 接口支持完全兼容 OpenAI 的接入方式
    需要环境变量：ZHIPU_API_KEY, ZHIPU_BASE_URL (通常为 https://open.bigmodel.cn/api/paas/v4)
    """
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model="glm-4",
        api_key=os.getenv("ZHIPU_API_KEY"),
        base_url=os.getenv("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
        temperature=temperature
    )


def get_anthropic_model(temperature=0):
    """
    Anthropic 模型初始化示例 (Claude 系列)
    依赖：pip install langchain-anthropic
    需要环境变量：ANTHROPIC_API_KEY
    """
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(
        model_name="claude-3-haiku-20240307",
        temperature=temperature
    )


# -------------------------------------------------------------
# 便捷获取当前活跃的模型实例（我们在后续教学案例里将引入这个函数）
# 你可以根据自己拥有的 API KEY，在这里切换 default_model 引擎
# -------------------------------------------------------------
def get_model(provider="openai", temperature=0):
    if provider == "openai":
        return get_openai_model(temperature)
    elif provider == "deepseek":
        return get_deepseek_model(temperature)
    elif provider == "zhipu":
        return get_zhipu_model(temperature)
    elif provider == "anthropic":
        return get_anthropic_model(temperature)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

if __name__ == "__main__":
    print("模型工厂配置完成。")
    print("你可以通过修改 'get_model()' 中的 provider 参数，一键切换后端模型！")
    get_model("openai")
