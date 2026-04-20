# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from pathlib import Path
import subprocess

project = 'pyhctsa'
copyright = '2026, Joshua B. Moore'
author = 'Joshua B. Moore'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc', 
    'sphinx_immaterial',
    'sphinx.ext.autosummary',
    'nbsphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx_design'
    ]

templates_path = ['_templates']
exclude_patterns = []

nbsphinx_execute = 'never'

autodoc_typehints = "description"
autodoc_typehints_format = "short"

napoleon_numpy_docstring = True
napoleon_google_docstring = False

# -- CSV generation ----------------------------------------------------------
def generate_csv(app):
    script = Path(__file__).parent / "scripts" / "generate_mappings.py"
    subprocess.run(["python", str(script)], check=True)

def setup(app):
    app.connect("builder-inited", generate_csv)

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_immaterial"
html_logo = "_static/apps_113dp_FFFFFF.svg"
html_css_files = ["custom.css"]
pygments_style = "friendly"
html_theme_options = {
    "features": [
        "navigation.expand",
        "navigation.sections",
        "navigation.top",
        "header.autohide",
    ],
    "font": False,
    "icon": {
        "repo": "fontawesome/brands/github",
    },
    "repo_name": "pyhctsa",
    "site_url": "https://dynamicsandneuralsystems.github.io/pyhctsa/",
    "repo_url": "https://github.com/DynamicsAndNeuralSystems/pyhctsa",
    "palette": [
        {
            "media": "(prefers-color-scheme: light)",
            "scheme": "default",
            "primary": "indigo",
            "accent": "indigo",
            "toggle": {
                "icon": "material/lightbulb-outline",
                "name": "Switch to dark mode",
            }
        },
        {
            "media": "(prefers-color-scheme: dark)",
            "scheme": "slate",
            "primary": "indigo",
            "accent": "indigo",
            "toggle": {
                "icon": "material/lightbulb",
                "name": "Switch to light mode",
            }
        },
    ],
}
html_static_path = ['_static']
