"""Sphinx configuration for grasp-sampler."""
import importlib.metadata

project = "grasp-sampler"
author = "grasp-sampler contributors"
copyright = "2026, grasp-sampler contributors"
try:
    release = importlib.metadata.version("grasp-sampler")
except importlib.metadata.PackageNotFoundError:
    release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",      # NumPy/Google docstrings
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",              # Markdown sources
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "grasp-sampler"

autodoc_typehints = "description"
autodoc_member_order = "bysource"
napoleon_use_rtype = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "trimesh": ("https://trimesh.org", None),
}

# Mock heavy/optional deps so the API builds without them installed.
autodoc_mock_imports = ["pybullet", "imageio"]
