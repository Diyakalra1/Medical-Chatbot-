def handle_conversation(query):
    query = query.strip().lower()

    if any(
        phrase in query
        for phrase in [
            "who are you",
            "what are you",
            "what do you do",
            "tell me about you",
            "what is medassist",
        ]
    ):
        return (
            "I'm MedAssist, a grounded health information assistant. "
            "I use a medical knowledge base to provide evidence-supported "
            "general health information."
        )

    if any(
        phrase in query
        for phrase in [
            "what can you do",
            "what can medassist do",
        ]
    ):
        return (
            "I can help explain medical conditions, symptoms, causes, "
            "and general health information using evidence retrieved from "
            "my medical knowledge base."
        )

    if any(
        phrase in query
        for phrase in [
            "thank",
            "thanks",
            "thank you",
            "thankyou",
        ]
    ):
        return "You're welcome. Feel free to ask another health-related question."

    return (
        "Hello! I'm MedAssist. I can help you understand general health "
        "and medical topics using a grounded medical knowledge base. "
        "What would you like to know?"
    )