import json

path = "QLoRA_mistral_7b_Gradio_Chatbot.ipynb"

with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

if "widgets" in nb.get("metadata", {}):
    print("Removing metadata.widgets")
    del nb["metadata"]["widgets"]
else:
    print("No metadata.widgets found")

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Done.")
