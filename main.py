import json
import time
from datetime import datetime, timezone
from pathlib import Path

from retrieval import retrieve_information
from answer import generate_answer
from critic import critique_answer

QUESTIONS_ROOT = Path("questions")


PRICING = {
    "retriever": {
        "input": 0.00,
        "output": 0.00
    },
    "generator": {
        # GPT-4o mini
        "input": 0.15,
        "output": 0.60
    },
    "critic": {
        # Perplexity Sonar
        "input": 1.00,
        "output": 1.00
    }
}

ENERGY_PER_TOKEN = 0.000000086   # kWh/token
CO2_PER_KWH = 0.445              # kg CO2 / kWh


def print_stats(all_stats, total_runtime):

    print("\n==============================")
    print("      SYSTEM STATISTICS")
    print("==============================")

    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    grand_total_cost = 0

    for stats in all_stats:

        input_tokens = stats["input_tokens"] or 0
        output_tokens = stats["output_tokens"] or 0
        tokens = stats["total_tokens"] or 0

        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        total_tokens += tokens

        pricing = PRICING[stats["agent"]]

        input_cost = (
            input_tokens / 1_000_000
        ) * pricing["input"]

        output_cost = (
            output_tokens / 1_000_000
        ) * pricing["output"]

        total_cost = input_cost + output_cost
        grand_total_cost += total_cost

        energy = tokens * ENERGY_PER_TOKEN
        co2 = energy * CO2_PER_KWH

        print(f"\n{stats['agent'].upper()}")
        print(f"Runtime: {stats['runtime']:.2f} seconds")
        print(f"Input Tokens: {input_tokens}")
        print(f"Output Tokens: {output_tokens}")
        print(f"Total Tokens: {tokens}")
        print(f"Total Cost: ${total_cost:.8f}")
        print(f"Estimated CO₂: {co2:.8f} kg")

    total_energy = total_tokens * ENERGY_PER_TOKEN
    total_co2 = total_energy * CO2_PER_KWH

    print("\n------------------------------")
    print("TOTAL SYSTEM")
    print("------------------------------")
    print(f"Runtime: {total_runtime:.2f} seconds")
    print(f"Input Tokens: {total_input_tokens}")
    print(f"Output Tokens: {total_output_tokens}")
    print(f"Total Tokens: {total_tokens}")
    print(f"Total Cost: ${grand_total_cost:.8f}")
    print(f"Estimated CO₂: {total_co2:.8f} kg")


def main():

    question = input("Enter your question: ")

    system_start = time.perf_counter()

    # Agent 1:

    print("\n[1/3] Running Retrieval Agent...")

    retrieval_result, retrieval_stats = retrieve_information(
        question,
        save_output=False
    )

    # Agent 2

    print("[2/3] Running Generator Agent...")

    answer, generator_stats = generate_answer(
        question,
        retrieval_result
    )

    # Agent 3

    print("[3/3] Running Critic Agent...")

    critique, critic_stats = critique_answer(
        question,
        answer
    )

    system_runtime = time.perf_counter() - system_start

    # final answer

    print("\n==============================")
    print("FINAL ANSWER")
    print("==============================")
    print(answer)

    # stats:

    all_stats = [
        retrieval_stats,
        generator_stats,
        critic_stats
    ]

    print_stats(all_stats, system_runtime)


def compute_stats(all_stats, total_runtime):
    """
    Mirrors print_stats() but returns structured data instead of printing.
    Every value matches what the CLI would print.
    """
    total_input_tokens  = 0
    total_output_tokens = 0
    total_tokens        = 0
    grand_total_cost    = 0

    agents = []

    for stats in all_stats:
        input_tokens  = stats["input_tokens"]  or 0
        output_tokens = stats["output_tokens"] or 0
        tokens        = stats["total_tokens"]  or 0

        total_input_tokens  += input_tokens
        total_output_tokens += output_tokens
        total_tokens        += tokens

        pricing    = PRICING[stats["agent"]]
        input_cost = (input_tokens  / 1_000_000) * pricing["input"]
        output_cost= (output_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost
        grand_total_cost += total_cost

        energy = tokens * ENERGY_PER_TOKEN
        co2    = energy * CO2_PER_KWH

        agents.append({
            "agent":         stats["agent"],
            "runtime":       stats["runtime"],
            "input_tokens":  input_tokens,
            "output_tokens": output_tokens,
            "total_tokens":  tokens,
            "cost":          total_cost,
            "co2":           co2,
        })

    total_energy = total_tokens * ENERGY_PER_TOKEN
    total_co2    = total_energy * CO2_PER_KWH

    system = {
        "runtime":       total_runtime,
        "input_tokens":  total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens":  total_tokens,
        "cost":          grand_total_cost,
        "co2":           total_co2,
    }

    return {"agents": agents, "system": system}


def save_run(folder: Path, question: str, answer: str, critique: str, stats: dict):
    """
    Save answer, critique, stats, and metadata to the run folder.
    Complements the files already written by retrieval.py.
    """
    folder = Path(folder)

    folder.joinpath("answer.txt").write_text(answer, encoding="utf-8")
    folder.joinpath("critique.txt").write_text(critique, encoding="utf-8")
    folder.joinpath("stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )

    # derive approval status from critique text
    import re
    status_match = re.search(r"Status:\s*(APPROVED|FLAGGED)", critique, re.IGNORECASE)
    status = status_match.group(1).upper() if status_match else "UNKNOWN"

    score_match = re.search(r"Accuracy Score:\s*(\d+)", critique, re.IGNORECASE)
    score = int(score_match.group(1)) if score_match else None

    run_id = folder.name  # e.g. "question_001"

    metadata = {
        "run_id":    run_id,
        "question":  question,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "chunks":    stats["system"]["total_tokens"],   # kept for compat; real chunk count in retrieved_context.json
        "status":    status,
        "score":     score,
        "cost":      stats["system"]["cost"],
        "co2":       stats["system"]["co2"],
        "runtime":   stats["system"]["runtime"],
    }

    folder.joinpath("metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def load_history():
    """Return a list of metadata dicts for all saved runs, newest first."""
    if not QUESTIONS_ROOT.exists():
        return []

    runs = []
    for folder in sorted(QUESTIONS_ROOT.iterdir(), reverse=True):
        meta_file = folder / "metadata.json"
        if meta_file.exists():
            try:
                runs.append(json.loads(meta_file.read_text(encoding="utf-8")))
            except Exception:
                pass
    return runs


def load_run(run_id: str) -> dict | None:
    """Load a single saved run by its run_id (e.g. 'question_001')."""
    folder = QUESTIONS_ROOT / run_id
    if not folder.exists():
        return None

    def read(name):
        f = folder / name
        return f.read_text(encoding="utf-8") if f.exists() else ""

    meta_file = folder / "metadata.json"
    metadata  = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}

    stats_file = folder / "stats.json"
    stats      = json.loads(stats_file.read_text(encoding="utf-8")) if stats_file.exists() else {}

    return {
        "run_id":   run_id,
        "question": read("question.txt"),
        "answer":   read("answer.txt"),
        "critique": read("critique.txt"),
        "stats":    stats,
        "metadata": metadata,
    }


def run_pipeline(question):
    """
    Generator for the web UI. Yields dicts with 'type' set to
    'status', 'complete', or 'error'.
    """
    system_start = time.perf_counter()

    try:
        yield {"type": "status", "message": "Searching Real-ESSI notes…"}
        retrieval_result, retrieval_stats = retrieve_information(
            question, save_output=True
        )

        yield {"type": "status", "message": "Generating answer…"}
        answer, generator_stats = generate_answer(question, retrieval_result)

        yield {"type": "status", "message": "Evaluating answer…"}
        critique, critic_stats = critique_answer(question, answer)

    except Exception as exc:
        yield {"type": "error", "message": str(exc)}
        return

    system_runtime = time.perf_counter() - system_start
    all_agent_stats = [retrieval_stats, generator_stats, critic_stats]
    stats           = compute_stats(all_agent_stats, system_runtime)

    # Save everything to disk
    saved_folder = retrieval_stats.get("saved_folder")
    run_id = None
    if saved_folder:
        save_run(Path(saved_folder), question, answer, critique, stats)
        run_id = Path(saved_folder).name

    yield {
        "type":     "complete",
        "answer":   answer,
        "critique": critique,
        "chunks":   retrieval_result["chunk_count"],
        "stats":    stats,
        "run_id":   run_id,
    }


if __name__ == "__main__":
    main().
