from pyhctsa.utils import make_function_name_mappings
from pathlib import Path

# run the function to generate the mappings

if __name__ == "__main__":
    out = Path(__file__).parent.parent / "mappings" / "legacy_function_name_mappings.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    make_function_name_mappings(None, str(out))
