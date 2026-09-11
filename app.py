"""Streamlit chat, side-by-side comparison, and evaluation dashboard."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from sql_agent.demo import create_demo
from sql_agent.history import load_profiles, load_runs, summarize
from sql_agent.run import rollout


st.set_page_config(page_title="AgenticRL SQL Lab", page_icon="🧪", layout="wide")
st.title("AgenticRL SQL Lab")
st.caption("Qwen3.5-4B · BIRD-style tools · Base / SFT / SFT + GRPO")


@st.cache_resource
def demo_tasks():
    task_file = create_demo("data/generated")
    return [json.loads(line) for line in task_file.read_text(encoding="utf-8").splitlines()]


@st.cache_data(ttl=5)
def history():
    return load_runs("outputs")


def args_for(profile_name, profile):
    return argparse.Namespace(
        base_url=profile["base_url"], model=profile["model"], profile=profile_name,
        api_key_env=profile.get("api_key_env", "SQL_AGENT_API_KEY"), max_calls=6, max_tokens=2048,
    )


def save_result(result):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = Path("outputs/ui") / (stamp + ".jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8")
    history.clear()
    return output


def render_result(result, show_score=True):
    if result.get("prediction"):
        st.code(result["prediction"], language="sql")
    else:
        st.error("模型没有调用 submit_sql。")
    query_result = result.get("query_result")
    if query_result:
        st.dataframe(pd.DataFrame(query_result["rows"], columns=query_result["columns"]), use_container_width=True)
    if result.get("query_error"):
        st.error(result["query_error"])
    score = result.get("score")
    cols = st.columns(3)
    if show_score and score is not None:
        cols[0].metric("执行正确", "是" if score.get("correct") else "否")
    else:
        cols[0].metric("执行正确", "无标签")
    cols[1].metric("探索调用", result.get("tool_calls", 0))
    cols[2].metric("耗时", "%.2fs" % result.get("elapsed_seconds", 0))
    with st.expander("完整工具轨迹"):
        for message in result.get("messages", []):
            if message.get("role") in {"assistant", "tool"}:
                st.json(message)


profiles = load_profiles()
tasks = demo_tasks()
labels = {profile["label"]: name for name, profile in profiles.items()}
chat_tab, compare_tab, dashboard_tab = st.tabs(["SQL 对话", "模型对比", "实验看板"])

with chat_tab:
    left, main = st.columns([1, 2])
    with left:
        label = st.selectbox("模型", list(labels), key="chat_model")
        selected_task = st.selectbox("示例数据库", tasks, format_func=lambda row: row["id"])
        with st.expander("数据库 schema"):
            from sql_agent.environment import SQLDatabase
            st.json(SQLDatabase(selected_task["database"]).schema())
    with main:
        question = st.chat_input("输入一个关于示例数据库的问题")
        if question:
            st.chat_message("user").write(question)
            task = {"id": "interactive", "question": question, "database": selected_task["database"]}
            try:
                with st.chat_message("assistant"), st.spinner("模型正在探索数据库"):
                    name = labels[label]
                    result = rollout(task, args_for(name, profiles[name]))
                    path = save_result(result)
                    render_result(result, show_score=False)
                    st.caption("记录：" + str(path))
            except Exception as exc:
                st.error("模型服务调用失败：" + str(exc))

with compare_tab:
    task = st.selectbox("冻结题目", tasks, format_func=lambda row: row["question"], key="compare_task")
    chosen = st.multiselect("对比模型", list(labels), default=list(labels))
    if st.button("运行独立对比", type="primary", disabled=not chosen):
        columns = st.columns(len(chosen))
        for column, label in zip(columns, chosen):
            with column:
                st.subheader(label)
                name = labels[label]
                try:
                    with st.spinner("运行中"):
                        result = rollout(task, args_for(name, profiles[name]))
                        save_result(result)
                    render_result(result)
                except Exception as exc:
                    st.error(str(exc))

with dashboard_tab:
    records = history()
    if not records:
        st.info("outputs/ 中还没有评测记录。运行批量评测或上方对话后刷新。")
    else:
        summary = summarize(records)
        frame = pd.DataFrame(summary)
        st.dataframe(frame, use_container_width=True)
        scored = frame.dropna(subset=["accuracy"])
        if not scored.empty:
            st.bar_chart(scored.set_index("profile")["accuracy"])
        st.subheader("任务记录")
        rows = []
        for row in records:
            rows.append({
                "profile": row.get("profile") or row.get("model"), "task_id": row.get("task_id"),
                "correct": (row.get("score") or {}).get("correct"), "tool_calls": row.get("tool_calls"),
                "elapsed_seconds": row.get("elapsed_seconds"), "prediction": row.get("prediction"),
                "source": "%s:%s" % (row.get("source_file"), row.get("source_line")),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
