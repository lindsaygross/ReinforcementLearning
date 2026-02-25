import random
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import streamlit as st

from config import (
    APP_TITLE,
    LOCAL_DATA_FILE,
    NUM_PREDICT,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    SUPABASE_KEY,
    SUPABASE_URL,
    TEMPERATURE_1,
    TEMPERATURE_2,
)
from database import DatabaseManager
from export import build_training_rows, compute_stats, to_jsonl
from llm import OllamaService


st.set_page_config(page_title=APP_TITLE, layout="wide")


@st.cache_resource
def get_db_manager() -> DatabaseManager:
    return DatabaseManager(
        supabase_url=SUPABASE_URL,
        supabase_key=SUPABASE_KEY,
        local_file=LOCAL_DATA_FILE,
    )


@st.cache_resource
def get_ollama_service() -> OllamaService:
    return OllamaService(
        host=OLLAMA_HOST,
        model=OLLAMA_MODEL,
        num_predict=NUM_PREDICT,
    )


def init_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    defaults = {
        "prompt_text": "",
        "generated": False,
        "current_prompt": "",
        "response_a": "",
        "response_b": "",
        "latency_a_ms": 0,
        "latency_b_ms": 0,
        "position_mapping": {},
        "generation_params": {},
        "flash_message": None,
        "clear_prompt_on_rerun": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_comparison(clear_prompt: bool = True) -> None:
    st.session_state.generated = False
    st.session_state.current_prompt = ""
    st.session_state.response_a = ""
    st.session_state.response_b = ""
    st.session_state.latency_a_ms = 0
    st.session_state.latency_b_ms = 0
    st.session_state.position_mapping = {}
    st.session_state.generation_params = {}
    if clear_prompt:
        # Avoid mutating a widget-backed key after widget instantiation in the same run.
        st.session_state.clear_prompt_on_rerun = True


def show_flash() -> None:
    flash: Optional[Dict[str, str]] = st.session_state.get("flash_message")
    if not flash:
        return

    level = flash.get("level", "info")
    text = flash.get("text", "")
    if level == "success":
        st.success(text)
    elif level == "warning":
        st.warning(text)
    else:
        st.info(text)

    st.session_state.flash_message = None


def store_preference(db: DatabaseManager, preference: str) -> None:
    if preference == "a":
        chosen = st.session_state.response_a
        rejected = st.session_state.response_b
    elif preference == "b":
        chosen = st.session_state.response_b
        rejected = st.session_state.response_a
    else:
        chosen = None
        rejected = None

    record = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt": st.session_state.current_prompt,
        "response_a": st.session_state.response_a,
        "response_b": st.session_state.response_b,
        "chosen": chosen,
        "rejected": rejected,
        "preference": preference,
        "model": OLLAMA_MODEL,
        "generation_params": st.session_state.generation_params,
        "response_a_latency_ms": st.session_state.latency_a_ms,
        "response_b_latency_ms": st.session_state.latency_b_ms,
        "session_id": st.session_state.session_id,
        "position_mapping": st.session_state.position_mapping,
    }

    _, message = db.insert_record(record)
    st.session_state.flash_message = {"level": "success", "text": f"Preference saved. {message}"}
    reset_comparison(clear_prompt=True)


def generate_pair(ollama_service: OllamaService, prompt: str) -> None:
    with st.spinner("Generating responses from Ollama..."):
        gen1_text, gen1_latency = ollama_service.generate_response(prompt, TEMPERATURE_1)
        gen2_text, gen2_latency = ollama_service.generate_response(prompt, TEMPERATURE_2)

    if random.random() < 0.5:
        response_a = gen1_text
        response_b = gen2_text
        latency_a = gen1_latency
        latency_b = gen2_latency
        temp_a = TEMPERATURE_1
        temp_b = TEMPERATURE_2
        mapping = {"a": "generation_1", "b": "generation_2"}
    else:
        response_a = gen2_text
        response_b = gen1_text
        latency_a = gen2_latency
        latency_b = gen1_latency
        temp_a = TEMPERATURE_2
        temp_b = TEMPERATURE_1
        mapping = {"a": "generation_2", "b": "generation_1"}

    st.session_state.generated = True
    st.session_state.current_prompt = prompt
    st.session_state.response_a = response_a
    st.session_state.response_b = response_b
    st.session_state.latency_a_ms = latency_a
    st.session_state.latency_b_ms = latency_b
    st.session_state.position_mapping = mapping
    st.session_state.generation_params = {
        "response_a": {"temperature": temp_a, "num_predict": NUM_PREDICT},
        "response_b": {"temperature": temp_b, "num_predict": NUM_PREDICT},
    }


def main() -> None:
    init_state()
    db = get_db_manager()
    ollama_service = get_ollama_service()

    # Safe point to clear prompt before creating the text area widget.
    if st.session_state.clear_prompt_on_rerun:
        st.session_state.prompt_text = ""
        st.session_state.clear_prompt_on_rerun = False

    st.title(APP_TITLE)
    st.info(
        "Enter a prompt, generate two model responses, then pick A, B, or Tie. "
        "Responses are locked after generation to avoid re-roll bias."
    )

    is_healthy, health_message = ollama_service.health_check()
    if not is_healthy:
        st.error(health_message)
        st.warning(
            "Install/start Ollama and pull the configured model before using this app."
        )
        st.code(
            """# Install Ollama from https://ollama.com
# Start Ollama (if needed)
ollama serve

# Pull the default model
ollama pull llama3.2"""
        )
        st.stop()

    show_flash()

    if db.supabase_ready:
        st.info("Supabase connected. Records will be stored in `preference_data`.")
    else:
        st.warning(
            "Supabase is not configured or unavailable. Falling back to local JSONL storage."
        )

    st.subheader("Comparison")
    st.text_area(
        "Prompt",
        key="prompt_text",
        height=140,
        placeholder="Ask a question or provide an instruction for the model...",
        disabled=st.session_state.generated,
    )

    btn_cols = st.columns([1, 1, 4])
    with btn_cols[0]:
        if st.button(
            "Generate Responses",
            disabled=st.session_state.generated or not st.session_state.prompt_text.strip(),
            use_container_width=True,
        ):
            try:
                generate_pair(ollama_service, st.session_state.prompt_text.strip())
                st.session_state.flash_message = {
                    "level": "info",
                    "text": "Responses generated. Select your preference below.",
                }
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to generate responses: {exc}")

    with btn_cols[1]:
        if st.button("New Comparison", use_container_width=True):
            reset_comparison(clear_prompt=True)
            st.session_state.flash_message = {
                "level": "info",
                "text": "Ready for a new comparison.",
            }
            st.rerun()

    if st.session_state.generated:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### Response A")
            st.write(st.session_state.response_a)
            st.caption(f"Latency: {st.session_state.latency_a_ms} ms")

        with col_b:
            st.markdown("### Response B")
            st.write(st.session_state.response_b)
            st.caption(f"Latency: {st.session_state.latency_b_ms} ms")

    st.subheader("Preference")
    pref_cols = st.columns(3)

    with pref_cols[0]:
        if st.button(
            "Prefer A",
            disabled=not st.session_state.generated,
            use_container_width=True,
        ):
            store_preference(db, "a")
            st.rerun()

    with pref_cols[1]:
        if st.button(
            "Prefer B",
            disabled=not st.session_state.generated,
            use_container_width=True,
        ):
            store_preference(db, "b")
            st.rerun()

    with pref_cols[2]:
        if st.button(
            "Tie",
            disabled=not st.session_state.generated,
            use_container_width=True,
        ):
            store_preference(db, "tie")
            st.rerun()

    records = db.get_all_records()
    stats = compute_stats(records)

    with st.sidebar:
        st.header("Dataset")
        st.caption(
            "Source: Supabase" if db.supabase_ready else f"Source: Local file ({LOCAL_DATA_FILE})"
        )

        st.metric("Total Comparisons", stats["total"])
        st.metric("Prefer A", stats["pref_a"])
        st.metric("Prefer B", stats["pref_b"])
        st.metric("Tie", stats["ties"])
        st.metric("Avg A Latency (ms)", stats["avg_a_latency_ms"])
        st.metric("Avg B Latency (ms)", stats["avg_b_latency_ms"])

        training_rows = build_training_rows(records)
        training_jsonl = to_jsonl(training_rows)
        full_jsonl = to_jsonl(records)

        st.download_button(
            "Export Training Data",
            data=training_jsonl,
            file_name="training_data.jsonl",
            mime="application/json",
            disabled=not training_rows,
            use_container_width=True,
        )
        st.download_button(
            "Export Full Data",
            data=full_jsonl,
            file_name="preference_data_full.jsonl",
            mime="application/json",
            disabled=not records,
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
