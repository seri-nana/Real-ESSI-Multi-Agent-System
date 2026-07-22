from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import time


load_dotenv()


llm = ChatOpenAI(
    model="gpt-4o-mini"
)


prompt = ChatPromptTemplate.from_template("""
Use ONLY the provided context to answer the user's question.

If the context does not contain enough information to answer the question,
say that the information is not available in the provided context.

Context:
{context}

Question:
{question}
""")


chain = prompt | llm


def generate_answer(question: str, retrieval_result: dict):

    start_time = time.perf_counter()


    context = "\n\n".join(
        chunk["text"]
        for chunk in retrieval_result["chunks"]
    )


    response = chain.invoke({
        "context": context,
        "question": question
    })


    runtime = time.perf_counter() - start_time


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
        "agent": "generator",
        "runtime": runtime,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens
    }


    return response.content, stats



if __name__ == "__main__":

    test_context = {
        "chunks": [
            {
                "text": (
                    "Dr. Jeremić is a professor in the Civil Engineering "
                    "Department at UC Davis. He researches earthquake "
                    "engineering and structural engineering."
                )
            }
        ]
    }


    question = input("Question: ")

    answer, stats = generate_answer(
        question,
        test_context
    )


    print("\nGenerated Answer:\n")
    print(answer)

    print("\nStats:")
    print(stats)
