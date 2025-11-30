import json

path = "QLoRA_mistral_7b_Gradio_Chatbot.ipynb"

with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# 删除顶层 metadata.widgets
if "widgets" in nb.get("metadata", {}):
    print("Removing top-level metadata.widgets")
    del nb["metadata"]["widgets"]

# 删除每个 cell 里的 metadata.widgets
for i, cell in enumerate(nb.get("cells", [])):
    if "widgets" in cell.get("metadata", {}):
        print(f"Removing widgets from cell {i}")
        del cell["metadata"]["widgets"]

# 保存回文件
with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Done. All widget metadata removed.")
