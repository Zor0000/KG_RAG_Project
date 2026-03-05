from agent.copilot_agent import run_agent

query = "How do I connect Copilot Studio to Dataverse?"
result = run_agent(query)

print(result["intent"])
print(result["persona"])
print(result["rewritten_query"])

print("\nCONTEXT:")
for c in result["context"]:
    print("-", c[:120])
