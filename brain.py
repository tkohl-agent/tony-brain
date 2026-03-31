import os
from openai import OpenAI
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

# ------------------------
# EMBEDDING
# ------------------------
def embed(text):
    res = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return res.data[0].embedding

# ------------------------
# MEMORY EXTRACTION
# ------------------------
def extract_memories(conversation):
    prompt = f"""
    Extract important long-term memories from this conversation.

    Use ONLY these types:
    - fact
    - preference
    - goal
    - project
    - relationship

    For each memory include:
    - type
    - content
    - importance (0.0 to 1.0)

    Importance rules:
    - 0.9–1.0 → core identity, major goals
    - 0.7–0.8 → strong preferences, active projects
    - 0.4–0.6 → useful but not critical
    - 0.1–0.3 → low signal

    Rules:
    - Keep memories short and atomic
    - No duplicates
    - No temporary info

    Return JSON list only.

    Conversation:
    {conversation}
    """

    res = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return eval(res.choices[0].message.content)

# ------------------------
# DUPLICATE CHECK
# ------------------------
def is_duplicate(content):
    query_embedding = embed(content)

    result = supabase.rpc("match_memories", {
        "query_embedding": query_embedding,
        "match_threshold": 0.90,
        "match_count": 1
    }).execute()

    return len(result.data) > 0

# ------------------------
# STORE MEMORY
# ------------------------
def store_memory(memory):
    if is_duplicate(memory["content"]):
        return

    vector = embed(memory["content"])

    supabase.table("memories").insert({
        "type": memory["type"],
        "content": memory["content"],
        "embedding": vector,
        "importance": memory.get("importance", 0.5)
    }).execute()

# ------------------------
# INGEST
# ------------------------
def ingest(conversation):
    memories = extract_memories(conversation)

    for m in memories:
        store_memory(m)

    supabase.rpc("decay_memory").execute()

# ------------------------
# RETRIEVE
# ------------------------
def get_memories(query):
    query_embedding = embed(query)

    result = supabase.rpc("match_memories", {
        "query_embedding": query_embedding,
        "match_threshold": 0.7,
        "match_count": 5
    }).execute()

    return result.data

# ------------------------
# REINFORCE
# ------------------------
def reinforce(memory_ids):
    supabase.table("memories") \
        .update({
            "importance": 1.0
        }) \
        .in_("id", memory_ids) \
        .execute()

# ------------------------
# BUILD CONTEXT
# ------------------------
def build_context(memories):
    grouped = {}

    for m in memories:
        grouped.setdefault(m["type"], []).append(m["content"])

    context = "\n[KNOWN USER CONTEXT]\n"

    for t, items in grouped.items():
        context += f"\n{t.upper()}:\n"
        for i in items:
            context += f"- {i}\n"

    return context

# ------------------------
# QUERY
# ------------------------
def query(user_input):
    memories = get_memories(user_input)

    memory_ids = [m["id"] for m in memories]
    reinforce(memory_ids)

    context = build_context(memories)

    print("\n--- CONTEXT ---")
    print(context)

    print("\n--- USER ---")
    print(user_input)
