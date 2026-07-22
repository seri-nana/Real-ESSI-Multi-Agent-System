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
Only use information that is peer reviewed

User Question:
{question}

Agent #2 Answer:
{answer}

Score the answer from 0–100.

Scoring rubric:

Technical accuracy: 0–40
Completeness: 0–30
Relevance: 0–20
Clarity: 0–10

Add the scores exactly.

Rules:
- Do not estimate.
- Use the same rubric every time.
- Give identical scores if the question and answer are identical.

Return:

Technical Accuracy: __/40
Completeness: __/30
Relevance: __/20
Clarity: __/10

Approval Rule:
- If the accuracy score is 90% or higher, return Status: APPROVED.
- If the accuracy score is below 90%, return Status: FLAGGED.

Return ONLY this format:

Accuracy Score: __/100

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
