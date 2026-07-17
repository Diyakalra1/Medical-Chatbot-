def parse_grounded_response(response_text):
    cleaned_response = response_text.strip()

    first_line = cleaned_response.splitlines()[0].strip().upper()

    if first_line == "GROUNDING_DECISION: SUPPORTED":
        grounding_decision = "SUPPORTED"
    elif first_line == "GROUNDING_DECISION: UNSUPPORTED":
        grounding_decision = "UNSUPPORTED"
    else:
        grounding_decision = "INVALID_RESPONSE"

    print("\n--- GROUNDING TRACE ---")
    print("Grounding Decision:", grounding_decision)

    if grounding_decision == "INVALID_RESPONSE":
        print("Reason: Model did not follow grounding response contract")
        return {
            "answer": (
                "I could not verify sufficient support from the retrieved "
                "medical evidence."
            ),
            "grounding_decision": grounding_decision,
        }

    if grounding_decision == "UNSUPPORTED":
        print("Reason: Retrieved context does not support the specific query or claim")
        return {
            "answer": (
                "The retrieved medical evidence does not contain sufficient "
                "information to support or address this specific claim or "
                "question. I cannot provide a grounded answer from the "
                "available medical knowledge base."
            ),
            "grounding_decision": grounding_decision,
        }

    print("Reason: Retrieved context supports the specific query")

    marker = "ANSWER:"
    marker_index = cleaned_response.find(marker)

    if marker_index == -1:
        return {
            "answer": (
                "I could not construct a grounded response from the "
                "retrieved medical evidence."
            ),
            "grounding_decision": "INVALID_RESPONSE",
        }

    answer = cleaned_response[marker_index + len(marker):].strip()

    return {
        "answer": answer,
        "grounding_decision": grounding_decision,
    }