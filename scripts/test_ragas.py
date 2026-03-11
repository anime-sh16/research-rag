# scratch_test.py (delete after)
import asyncio
import os

import instructor
from google import genai
from ragas.llms.base import InstructorLLM
from ragas.metrics.collections import Faithfulness

from src.config.config import settings

os.environ["GOOGLE_API_KEY"] = settings.google_api_key.get_secret_value()

client = genai.Client(api_key=settings.google_api_key.get_secret_value())

async_instructor = instructor.from_genai(
    client,
    mode=instructor.Mode.GENAI_STRUCTURED_OUTPUTS,
    use_async=True,
)
# llm = llm_factory(
#     model="gemini-2.5-flash-lite",
#     provider="google",
#     client=client,
# )
llm = InstructorLLM(
    client=async_instructor,
    model=settings.generation.model,
    provider="google",
)
print(f"is_async: {llm.is_async}")  # should print True

metric = Faithfulness(llm=llm)

result = asyncio.run(
    metric.ascore(
        user_input="What is 2+2?",
        response="2+2 equals 4.",
        retrieved_contexts=["Mathematics: 2+2 equals 4."],
    )
)
print(f"faithfulness score: {result}")
