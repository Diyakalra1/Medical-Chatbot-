from src.prompt import build_generation_prompt
from src.response_parser import parse_grounded_response


def build_context(documents):
    context_parts = []

    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "Medical Knowledge Base")
        page = document.metadata.get("page", "Unknown")

        context_parts.append(
            f"""
SOURCE {index}
Document: {source}
Page: {page}

{document.page_content}
"""
        )

    return "\n".join(context_parts)


def generate_answer(query, documents, client, model_name):
    context = build_context(documents)

    prompt = build_generation_prompt(query, context)

    response = client.models.generate_content(model=model_name,contents=prompt,)

    return parse_grounded_response(response.text)