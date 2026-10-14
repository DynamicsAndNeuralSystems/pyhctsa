from setuptools import setup
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path


ext_path = Path(__file__).resolve().parent / "pyhctsa" / "_build" / "ext_modules.py"
spec = spec_from_file_location("extmods", ext_path)
extmods = module_from_spec(spec)
spec.loader.exec_module(extmods)

if __name__ == "__main__":
    setup(ext_modules=extmods.build_extensions())
