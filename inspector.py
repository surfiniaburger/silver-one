import inspect
from google.genai.types import Part

print("="*30)
print("Inspecting Part class:")

# Print all callable members that don't start with an underscore
for name, member in inspect.getmembers(Part):
    if callable(member) and not name.startswith('_'):
        print(name)

print("\nSignature for Part.__init__:")
print(inspect.signature(Part.__init__))

print("="*30)

