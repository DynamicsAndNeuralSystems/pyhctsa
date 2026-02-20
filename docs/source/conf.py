# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

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
    'sphinx.ext.napoleon'
    ]

templates_path = ['_templates']
exclude_patterns = []

nbsphinx_execute = 'never'

autodoc_typehints = "description"
autodoc_typehints_format = "short"

napoleon_numpy_docstring = True
napoleon_google_docstring = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_immaterial"
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
    "site_url": "https://your-site.com/",
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
