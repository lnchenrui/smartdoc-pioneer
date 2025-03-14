from langchain_core.vectorstores import VectorStore
import inspect

print("VectorStore abstract methods:")
print(VectorStore.__abstractmethods__)

print("\nVectorStore methods:")
for name, method in inspect.getmembers(VectorStore, predicate=inspect.isfunction):
    print(f"{name}: {method}")

print("\nVectorStore class methods:")
for name, method in inspect.getmembers(VectorStore, predicate=inspect.ismethod):
    print(f"{name}: {method}")

print("\nVectorStore class attributes:")
for name, attr in inspect.getmembers(VectorStore):
    if not name.startswith('__') and not inspect.ismethod(attr) and not inspect.isfunction(attr):
        print(f"{name}: {attr}") 