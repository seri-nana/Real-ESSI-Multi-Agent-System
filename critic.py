import os
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

api_key = os.getenv("CRITIQUE_API_KEY")

if not api_key:
    raise ValueError(
        "CRITIQUE_API_KEY was not found. "
        "Make sure it is defined in your .env file."
    )


llm = ChatOpenAI(
    model="sonar",
    api_key=api_key,
    base_url="https://api.perplexity.ai",
    temperature=0
)


critic_prompt = ChatPromptTemplate.from_template(
"""
You are a critic agent for an engineering AI system.

Evaluate the answer using your engineering knowledge.

User Question:
{question}

Agent #2 Answer:
{answer}

Score the answer from 0–100.

Approval Rule:
- If the accuracy score is 90% or higher, return Status: APPROVED.
- If the accuracy score is below 90%, return Status: FLAGGED.

Return ONLY this format:

Accuracy Score: __%

Status: APPROVED or FLAGGED

Overall:
<2 short sentences>

Do not write anything else.
"""
)


def critique_answer(question: str, answer: str):

    final_prompt = critic_prompt.format(
        question=question,
        answer=answer
    )

    start_time = time.perf_counter()

    response = llm.invoke(final_prompt)

    runtime = time.perf_counter() - start_time


    # Extract token usage
    try:
        usage = response.response_metadata["token_usage"]

        input_tokens = usage["prompt_tokens"]
        output_tokens = usage["completion_tokens"]
        total_tokens = usage["total_tokens"]

    except Exception:
        input_tokens = None
        output_tokens = None
        total_tokens = None


    stats = {
        "agent": "critic",
        "runtime": runtime,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens
    }


    return response.content, stats
