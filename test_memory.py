from app.rag_chain import build_rag_chain, ask_question

print("Building RAG chain...")
chain = build_rag_chain()
print("Chain ready.\n")
print("=" * 60)

# This conversation only makes sense if memory is working.
# Each question depends on knowing what the previous one was about.

questions = [
    # Turn 1 — establish the topic
    "What is multi-head attention?",

    # Turn 2 — "it" refers to multi-head attention, needs memory
    "How many heads does it use in the paper?",

    # Turn 3 — "this" refers to the whole attention mechanism
    "Why is this better than single attention?",

    # Turn 4 — completely new topic, tests that memory doesn't bleed over
    "What optimizer was used for training?",

    # Turn 5 — "it" now refers to the optimizer
    "What were its hyperparameter values?",
]

for i, question in enumerate(questions):
    print(f"Turn {i+1}: {question}")
    result = ask_question(chain, question)
    print(f"Answer: {result['answer']}")
    pages = ["Page " + str(s["page"]) for s in result["sources"]]
    print(f"Sources: {pages}")
    print("-" * 60)
    