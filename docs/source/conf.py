# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


import os
import sys

# Make the gwlens package importable for autodoc, even if it isn't installed.

sys.path.insert(0, os.path.abspath("../.."))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'GWLENS'
copyright = '2026'
author = 'Sreekanth Harikumar'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


#extensions = ['myst_parser','sphinx.ext.napoleon','sphinx.ext.autodoc']
#myst_enable_extensions = ["colon_fence"]
#=======
extensions = ['myst_parser','sphinx.ext.napoleon','sphinx.ext.autodoc', 'sphinx.ext.githubpages']

#>>>>>>> 2c8ae488d7b9fe31a9cf6a31071f3392245ccd16
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db']
napoleon_numpy_docstring = True
napoleon_google_docstring = False


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
    "sticky_navigation": True,
    "titles_only": False,
    "display_version": True,
}

# "Edit on GitHub" link at the top of every page
html_context = {
    "display_github": True,
    "github_user": "hsreekanth15",
    "github_repo": "GWLENS",
    "github_version": "main",
    "conf_py_path": "/docs/source/",
}
