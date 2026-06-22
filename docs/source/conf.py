"""Sphinx configuration for the CytoOne documentation."""

import os
import re
import sys
from datetime import datetime

# -- Path setup --------------------------------------------------------------
# Make the package importable for autodoc (repo root is two levels up:
# docs/source/conf.py -> docs/source -> docs -> repo root).
_REPO_ROOT = os.path.abspath("../..")
sys.path.insert(0, _REPO_ROOT)


def _read_metadata():
    """Parse version/author from CytoOne/__version__.py without importing it.

    Importing the package would pull in torch / pyro / scanpy, which we
    deliberately avoid on the docs builder.
    """
    meta = {}
    version_file = os.path.join(_REPO_ROOT, "CytoOne", "__version__.py")
    with open(version_file, "rt", encoding="utf-8") as fh:
        text = fh.read()
    for key in ("version", "author", "title"):
        match = re.search(
            r"^__%s__\s*=\s*['\"]([^'\"]*)['\"]" % key, text, re.M
        )
        meta[key] = match.group(1) if match else ""
    return meta


_META = _read_metadata()

# -- Project information -----------------------------------------------------
project = "CytoOne"
author = _META.get("author", "CytoOne authors")
copyright = f"{datetime.now():%Y}, {author}"
release = _META.get("version", "0.0.0")
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",        # NumPy-style docstrings
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",                # Markdown support
    "sphinx_design",              # grids / cards on the landing page
    "sphinx_copybutton",          # copy button on code blocks
]

# Accept both reStructuredText and Markdown source files.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- MyST (Markdown) options -------------------------------------------------
myst_enable_extensions = [
    "colon_fence",   # ::: fenced directives
    "deflist",
    "linkify",       # turn bare URLs into links
    "substitution",
]
myst_heading_anchors = 3

# -- autodoc / autosummary ---------------------------------------------------
autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# Mock the heavy / compiled runtime dependencies so the API reference builds
# without installing them on the docs runner.
autodoc_mock_imports = [
    "torch",
    "pyro",
    "scanpy",
    "anndata",
    "scipy",
    "seaborn",
    "psutil",
    "tqdm",
    "sklearn",
    "umap",
]

# -- intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# -- HTML output -------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_title = f"CytoOne {release}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
    "style_external_links": True,
}

# Link source on GitHub from the docs.
html_context = {
    "display_github": True,
    "github_user": "Yuqiu-Yang",
    "github_repo": "CytoOne",
    "github_version": "main",
    "conf_py_path": "/docs/",
}
