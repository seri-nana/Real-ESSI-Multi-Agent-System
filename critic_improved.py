import os
import re
import time
from urllib import response
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

api_key = os.getenv("CRITIQUE_API_KEY")

if not api_key:
    raise ValueError(
        "CRITIQUE_API_KEY was not found."
    )


# Perplexity model with web access
llm = ChatOpenAI(
    model="sonar",
    api_key=api_key,
    base_url="https://api.perplexity.ai",
    temperature=0,
    top_p=0
)


# Step 1: Retrieve evidence


evidence_prompt = ChatPromptTemplate.from_template(
"""
You are a research assistant.

Find reliable sources that can verify the following answer.

Use scholarly sources, peer-reviewed papers, university pages,
or official academic sources whenever possible.

Question:
{question}

Answer to verify:
{answer}

Return:
1. Sources found
2. Key information from each source
3. Whether the sources support or contradict the answer
"""
)


# Step 2: Critic evaluation
critic_prompt = ChatPromptTemplate.from_template(
"""
You are an answer validation agent for a Retrieval-Augmented Generation (RAG) system.

Your task is to determine whether Agent #2's answer is supported by the evidence retrieved from reliable sources.

Your goal is NOT to check whether every word matches the evidence.
Your goal is to determine whether the answer communicates the correct overall meaning.

IMPORTANT RULES:

- Use ONLY the provided evidence.
- Do NOT use your own memory or outside knowledge.
- Compare meaning, not exact wording.
- Assume Agent #2 is correct unless the evidence clearly contradicts it.
- Do NOT heavily penalize:
    - synonyms
    - simplified explanations
    - different academic terminology
    - missing minor details
    - additional related details that do not conflict with evidence

- If the answer identifies the correct person and correctly describes their main research/work area, it should generally be considered supported.
- Additional details should only reduce confidence if they introduce a factual contradiction or a clearly incorrect statement.

Question:
{question}

Agent #2 Answer:
{answer}

Evidence Found:
{evidence}


SPECIAL CASE:

If Agent #2 provides no answer, refuses to answer, or states that information is unavailable:
Return:

Claim 1: SUPPORTED

Overall:
No factual claims were provided to evaluate.


EVALUATION PROCEDURE:

1. Identify the major factual claims in Agent #2's answer.
2. Group related statements into meaningful claims.
Do not split the answer into many small claims.
3. For each major claim, classify it as exactly one:

SUPPORTED:
The evidence confirms the claim or supports the same general meaning.

UNKNOWN:
The evidence does not contain enough information to verify the claim, but the claim is not contradicted.

CONTRADICTED:
The evidence directly conflicts with the claim.


IMPORTANT:
- UNKNOWN does NOT mean false.
- Do not assign low confidence simply because a detail was not found.
- Only use CONTRADICTED when the evidence clearly proves the answer is wrong.
- If the answer and evidence describe the same person and the same general work/research area, the answer should generally be considered supported.

FINAL EVALUATION PRINCIPLE:
The purpose of this critic is to detect incorrect answers, not punish additional detail.
If Agent #2 identifies:
1. the correct person,
2. the correct institution or field,
3. the correct general research/work area,
then the answer should be considered APPROVED unless there is a clear factual contradiction.
Minor differences in wording, level of detail, or included examples should not significantly change the evaluation.

Return ONLY:

Claim 1: SUPPORTED
Claim 2: UNKNOWN
Claim 3: CONTRADICTED

Overall:
<2 short sentences explaining whether the answer matches the evidence>
"""
)

def critique_answer(question: str, answer: str):

    start_time = time.perf_counter()


    # Get evidence first
    evidence_request = evidence_prompt.format(
        question=question,
        answer=answer
    )

    evidence_response = llm.invoke(
        evidence_request
    )

    evidence = evidence_response.content


    # Evaluate answer against evidence
    final_prompt = critic_prompt.format(
        question=question,
        answer=answer,
        evidence=evidence
    )

    response = llm.invoke(
        final_prompt
    )


 
    critique = response.content

    supported = len(re.findall(r":\s*SUPPORTED", critique, re.IGNORECASE))
    unknown = len(re.findall(r":\s*UNKNOWN", critique, re.IGNORECASE))
    contradicted = len(re.findall(r":\s*CONTRADICTED", critique, re.IGNORECASE))

    score = 100

    score -= unknown * 5
    score -= contradicted * 20

    score = max(score, 0)

    status = "APPROVED" if score >= 90 else "FLAGGED"

    overall_match = re.search(
    r"Overall:\s*(.*)",
    critique,
    re.DOTALL
)

    overall = overall_match.group(1).strip() if overall_match else ""

    final_output = f"""Accuracy Score: {score}%

    Status: {status}

    Overall:
{overall}
"""


    runtime = time.perf_counter() - start_time


    try:
        usage1 = evidence_response.response_metadata["token_usage"]
        usage2 = response.response_metadata["token_usage"]

        input_tokens = (
            usage1["prompt_tokens"]
            +
            usage2["prompt_tokens"]
        )

        output_tokens = (
            usage1["completion_tokens"]
            +
            usage2["completion_tokens"]
        )

        total_tokens = input_tokens + output_tokens

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


    return final_output, stats
