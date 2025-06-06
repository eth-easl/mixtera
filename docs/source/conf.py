import os
import sys

project = "Mixtera"
copyright = "2024, EASL"
author = "EASL Authors"

sys.path.insert(0, os.path.abspath("../../"))
print(os.path.abspath("../../"))

extensions = [
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "myst_parser",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "furo"
html_static_path = ["_static"]
source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}
myst_enable_extensions = [
    "colon_fence"
]
