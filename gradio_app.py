"""
CoreCoder Gradio Chat — 可直接嵌入现有的 Gradio 页面。

Usage:
    python gradio_app.py

特点:
    - 真实流式输出 (token 逐字显示, 非假流式)
    - 工具调用实时展示
    - 多用户会话隔离 (gr.State 传类而非实例)
    - 可作为 Tab 嵌入现有 Gradio 应用 (from gradio_app import app)

⚠  vLLM 必须用以下参数启动才支持工具调用:
    vllm serve <model> --enable-auto-tool-choice --tool-call-parser hermes
"""

import sys
import threading
import queue as tqueue
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr
from corecoder.agent import Agent
from corecoder.llm import LLM
from corecoder.config import Config


def _build_llm() -> LLM:
    cfg = Config.from_env()
    if not cfg.api_key:
        cfg.api_key = "vllm"
    return LLM(
        model=cfg.model,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )


class GradioAgent:
    """Per-user agent wrapper — gr.State(GradioAgent) creates one per session."""

    def __init__(self):
        self.agent: Agent | None = None

    def _ensure_agent(self):
        if self.agent is None:
            self.agent = Agent(llm=_build_llm())

    def chat_stream(self, message: str, history: list):
        """Generator that yields real-time updates as the agent works.

        Uses a threading.Queue to bridge the synchronous agent.chat() (running
        in a daemon thread) with Gradio's generator-based streaming. Each token
        and tool call is yielded to the UI as it happens.
        """
        self._ensure_agent()

        q: tqueue.Queue = tqueue.Queue()
        thread_done = threading.Event()

        def on_token(tok: str):
            q.put(("token", tok))

        def on_tool(name: str, args: dict):
            brief = ", ".join(f"{k}={str(v)[:30]}" for k, v in args.items())
            q.put(("tool", f"🔧 **{name}**({brief})"))

        def run_agent():
            try:
                response = self.agent.chat(message, on_token=on_token, on_tool=on_tool)
                q.put(("done", response))
            except Exception as e:
                q.put(("error", str(e)))
            finally:
                thread_done.set()

        thread = threading.Thread(target=run_agent, daemon=True)
        thread.start()

        # messages format: list of {role, content} dicts
        new_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": ""},
        ]
        yield new_history

        current_text = ""
        tool_log: list[str] = []

        while True:
            try:
                event_type, content = q.get(timeout=0.1)
            except tqueue.Empty:
                if thread_done.is_set() and q.empty():
                    break
                continue

            if event_type == "token":
                current_text += content
            elif event_type == "tool":
                tool_log.append(content)
            elif event_type == "done":
                if not current_text:
                    current_text = content
            elif event_type == "error":
                current_text = f"❌ 错误: {content}"

            display = current_text
            if tool_log:
                display += "\n\n---\n" + "\n".join(tool_log)
            new_history[-1]["content"] = display
            yield new_history

            if event_type in ("done", "error"):
                break

        thread.join(timeout=1)

    def reset(self):
        if self.agent:
            self.agent.reset()


def _create_app() -> gr.Blocks:
    css = """
    .tool-log { color: #6b8; font-family: monospace; font-size: 0.85em; }
    footer { visibility: hidden; }
    """
    with gr.Blocks(css=css, title="CoreCoder Chat") as app:
        gr.Markdown("## ⚡ CoreCoder Agent Chat")
        gr.Markdown(f"模型: **{_build_llm().model}** | 支持工具调用 + 流式输出")

        # type='messages' is the Gradio 5.x default and works on 4.x too
        chatbot = gr.Chatbot(label="对话", height=500, type="messages")
        msg_input = gr.Textbox(
            label="输入",
            placeholder="输入任务… (例如: 列出项目里的所有 TODO)",
            lines=2,
        )
        with gr.Row():
            send_btn = gr.Button("发送", variant="primary")
            clear_btn = gr.Button("清空上下文")

        # pass the CLASS, not an instance — Gradio calls GradioAgent() per
        # session so each user gets their own isolated agent
        agent_state = gr.State(GradioAgent)

        def on_send(message: str, history: list, agent: GradioAgent):
            if not message.strip():
                yield history
                return
            yield from agent.chat_stream(message, history)

        def on_clear(agent: GradioAgent):
            agent.reset()
            return [], ""

        send_btn.click(
            on_send,
            inputs=[msg_input, chatbot, agent_state],
            outputs=[chatbot],
        ).then(lambda: "", None, msg_input)

        msg_input.submit(
            on_send,
            inputs=[msg_input, chatbot, agent_state],
            outputs=[chatbot],
        ).then(lambda: "", None, msg_input)

        clear_btn.click(on_clear, inputs=[agent_state], outputs=[chatbot, msg_input])

    return app


app = _create_app()


def main():
    cfg = Config.from_env()
    print(f"""
══════════════════════════════════════════════
  CoreCoder Gradio Chat
  Model : {cfg.model}
  Base  : {cfg.base_url}
  Addr  : http://0.0.0.0:7860
══════════════════════════════════════════════
  ⚠ vLLM 工具调用必须用以下参数启动:
    vllm serve <model> --enable-auto-tool-choice --tool-call-parser hermes
  否则 agent 的工具永远不会触发
══════════════════════════════════════════════
""".strip())
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()
