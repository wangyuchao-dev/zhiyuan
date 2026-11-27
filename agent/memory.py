"""记忆中枢：把对话历史、事实槽、增量摘要三份存储组合成对外的单一入口。

组合而非继承 —— ChatService 只跟 Memory 打交道，
以后加第四种存储时，上层一行都不用改。
build_context() 负责拼上下文，顺序是「最近消息 → 摘要 → 关键事实」
（事实放最后，规避 lost in the middle）。"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.chat_history_manager import ChatHistoryManager
from agent.facts_manager import FactManager
from agent.summary_manager import SummaryManager

SESSION_KEY = "session"


class Memory:
    """单个会话的记忆中枢：历史 / 事实槽 / 摘要，三层各自独立，对外一个入口"""

    # 摘要触发阈值：攒够这么多条消息才压一次。
    # 压一次 = 一次完整的模型调用，每轮都压等于成本和延迟都翻倍。
    SUMMARY_TRIGGER = 10

    def __init__(self, chat_session_id: str):

        key = f"{SESSION_KEY}：{chat_session_id}"

        self.history = ChatHistoryManager(key)
        self.facts = FactManager(key)
        self.summary = SummaryManager(key)

        # 摘要存在 SummaryManager 的模块级 store 里：
        # 这样 Memory 实例被重建时，摘要不会丢。
        self._summary = self.summary.get()  # 本次会话最新的摘要
        self._unsummaryed = 0  # 未摘要的对话条数

    # 触发阈值调用摘要的函数
    def _maybe_summary(self, llm):
        if self._unsummaryed < self.SUMMARY_TRIGGER:
            return
        # 用 ChatHistoryManager 的 get_recent_messages 取最近几条
        new_mgs = self.history.get_recent_messages(self._unsummaryed)
        self._summary = self.summary.generate_incremental_summary(self._summary, new_mgs, llm)
        self.summary.set(self._summary)  # 落 store，别只存在实例里
        self._unsummaryed = 0

    def add_user(self, text: str, llm):
        """写入用户这句话，并检查是否需要压摘要"""
        self.history.add_message(HumanMessage(content=text))
        self._unsummaryed += 1
        self._maybe_summary(llm)

    def add_ai(self, text: str, llm):
        """写入模型这条回复，并检查是否需要压摘要"""
        self.history.add_message(AIMessage(content=text))
        self._unsummaryed += 1
        self._maybe_summary(llm)

    # 存储关键事实：同key则覆盖、新增key则新增
    def update_facts(self, slots: dict):
        """把意图识别出来的 slots（和正则抽到的实体）写进事实槽，空值跳过"""
        if not slots:
            return
        for key, value in slots.items():
            if value:
                self.facts.put(key, value)  # 老师的方法叫 put，不是 set_fact

    # 核心方法：构建上下文的函数：【公式：上下文 = 最近消息 + 摘要 + 关键事实】
    # 顺序很关键：事实和摘要要放在【最后】，离用户这句话最近，模型才看得见。
    # 放最前面的话，后面跟着一串对话，模型注意力会被最近的对话带走，
    # 结果就是"事实槽里明明有订单号，客服还在问客户要订单号"。
    #
    # 角色选择：摘要和事实用 SystemMessage，不是 AIMessage。
    #   AIMessage 的语义是"AI 说过的话" —— 用它会连续出现三条 ai，
    #   模型会把系统注入的内容当成自己上一轮的发言，而且破坏了 human/ai 交替结构。
    #   SystemMessage 表示"系统注入的指令"，语义正确，严格校验的模型模板也不会报错。
    #   两块合成一条，避免出现多条连续 system。
    #
    # limit: 最近消息条数，由 QaConfig.history_limit 传入
    def build_context(self, limit: int = 10) -> list:
        """拼本轮上下文：最近 N 条消息 + 摘要 + 关键事实（事实放最后）"""
        msgs = []

        # 1、最近消息（先让它看到聊了什么）
        msgs.extend(self.history.get_recent_messages(limit))

        # 2、摘要 + 关键事实（合成一条 SystemMessage，放最后，离本轮问题最近）
        blocks = []
        if self._summary:
            blocks.append(f"【本次会话摘要】{self._summary}")
        facts = self.facts.get_all()
        if facts:
            t = " | ".join([f"{key}: {value}" for key, value in facts.items()])
            blocks.append(f"【当前已知事实，回答时直接用，不要再问客户】{t}")
        if blocks:
            msgs.append(SystemMessage(content="\n".join(blocks)))

        return msgs

    def clear_all(self) -> None:
        """
        清空本会话的全部记忆：历史 / 事实槽 / 摘要。

        对外只有这一个入口 —— Agent.reset_session 调它。
        此前这个方法缺失，导致 POST /api/reset 必然 AttributeError（500），
        前端「清空会话」按钮点了没有任何效果。
        """
        self.history.clear()
        self.facts.clear()
        self.summary.set("")
        self._summary = ""
        self._unsummaryed = 0
