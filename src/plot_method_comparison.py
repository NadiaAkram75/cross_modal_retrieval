import matplotlib.pyplot as plt

methods = ["Visual-only", "Semantic-only", "Hybrid"]
scores = [0.9398, 0.1276, 0.1276]

plt.figure(figsize=(6, 4))
plt.bar(methods, scores)
plt.ylabel("Mean semantic distance of top-5")
plt.title("Retrieval Method Comparison")
plt.tight_layout()
plt.savefig("method_comparison.png", dpi=200)

print("Saved: method_comparison.png")