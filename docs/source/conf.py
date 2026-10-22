# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Make the gwlens package importable for autodoc, even if it isn't installed.
sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
project = 'GWLENS'
copyright = '2026'
author = 'Sreekanth Harikumar'
copyright = f'2026, {author}'
release = '0.1'

# -- General configuration ---------------------------------------------------
extensions = [
    'myst_parser',                 # Markdown pages
    'sphinx.ext.napoleon',         # NumPy-style docstrings
    'sphinx.ext.autodoc',          # API docs from docstrings
    'sphinx.ext.githubpages',
    'sphinx.ext.viewcode',         # [source] links in the API reference
    'sphinx.ext.mathjax',          # LaTeX math
    'nbsphinx',                    # Jupyter notebooks as pages (needs pandoc)
    'sphinx_copybutton',           # copy button on code blocks
    'sphinx_design',               # buttons, cards, grids
]

myst_enable_extensions = ["colon_fence", "dollarmath", "amsmath"]  # lets sphinx_design use ::: fences in .md

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '**.ipynb_checkpoints','examples/animations.ipynb']
napoleon_numpy_docstring = True
napoleon_google_docstring = False

# Packages that autodoc should fake, so modules import without them installed
autodoc_mock_imports = ['mpi4py', 'bilby', 'bilby_pipe', 'gwcosmo',
                        'lens2d_general', 'general']

# Show the outputs saved in notebooks; don't re-run them during the build
nbsphinx_execute = 'never'
nbsphinx_codecell_lexer = "python3"

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
    "sticky_navigation": True,
    "titles_only": False,
}
html_css_files = ['custom.css']
html_js_files = ['custom.js', 'page-toc.js']

# "Edit on GitHub" link at the top of every page
html_context = {
    "display_github": True,
    "github_user": "hsreekanth15",
    "github_repo": "GWLENS",
    "github_version": "main",
    "conf_py_path": "/docs/source/",
}
