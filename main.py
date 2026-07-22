import time

from retrieval import retrieve_information
from answer import generate_answer
from critic import critique_answer


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


if __name__ == "__main__":
    main()
