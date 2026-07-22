import time

from retrieval import retrieve_information
from answer import generate_answer
from critic import critique_answer


def print_stats(all_stats, total_runtime):

    print("\n==============================")
    print("SYSTEM STATISTICS")
    print("==============================")

    total_tokens = 0

    for stats in all_stats:
        print(f"\n{stats['agent'].upper()}")
        print(f"Runtime: {stats['runtime']:.2f} seconds")
        print(f"Input Tokens: {stats['input_tokens']}")
        print(f"Output Tokens: {stats['output_tokens']}")
        print(f"Total Tokens: {stats['total_tokens']}")

        if stats["total_tokens"] is not None:
            total_tokens += stats["total_tokens"]

    print("\n------------------------------")
    print("TOTAL SYSTEM")
    print("------------------------------")
    print(f"Total Runtime: {total_runtime:.2f} seconds")
    print(f"Total Tokens: {total_tokens}")


def main():

    question = input("Enter your question: ")

    start_time = time.perf_counter()


    # ==============================
    # Agent 1: Retrieval
    # ==============================

    print("\n[1/3] Running Retrieval Agent...")

    retrieval_result, retrieval_stats = retrieve_information(
        question,
        save_output=False
    )


    # ==============================
    # Agent 2: Generator
    # ==============================

    print("[2/3] Running Generator Agent...")

    answer, generator_stats = generate_answer(
        question,
        retrieval_result
    )


    # ==============================
    # Agent 3: Critic
    # ==============================

    print("[3/3] Running Critic Agent...")

    critique, critic_stats = critique_answer(
        question,
        answer
    )


    total_runtime = time.perf_counter() - start_time


    # ==============================
    # Final Output
    # ==============================

    print("\n=============================te=")
    print("FINAL ANSWER")
    print("==============================")
    print(answer)


    # ==============================
    # Statistics
    # ==============================

    all_stats = [
        retrieval_stats,
        generator_stats,
        critic_stats
    ]

    print_stats(
        all_stats,
        total_runtime
    )


if __name__ == "__main__":
    main()
