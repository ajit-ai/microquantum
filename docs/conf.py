"""Sphinx configuration for the microquantum documentation."""

project = "MicroQuantum"
copyright = "2026, Ajit Kumar"
author = "Ajit Kumar"
version = "0.4.0"
release = "0.4.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "autoapi.extension",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "autoapi",
]

html_theme = "furo"
html_title = "MicroQuantum"
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "announcement": (
        "MicroQuantum is a Developer Preview. APIs may change before a stable "
        "1.0 release."
    ),
    "light_css_variables": {
        "color-announcement-background": "#dee5ff",
        "color-announcement-text": "#314159",
    },
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

extlinks = {
    "pypi": ("https://pypi.org/project/%s/", "%s"),
}

autoapi_type = "python"
autoapi_dirs = ["../src/microquantum"]
autoapi_ignore = ["*/tests/*"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
]
autoapi_add_toctree_entry = False
autoapi_root = "api"
autoapi_keep_files = True

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_ivar = True

suppress_warnings = [
    "autoapi.python_import_resolution",
    # Docstrings legitimately use bra-ket notation (e.g. ``|psi>``); docutils
    # parses the leading pipe as an inline substitution reference and warns.
    # The rendered output is correct, so silence that docutils message.
    "docutils",
]

# The docs build already runs with warnings-as-errors.  nitpicky is off
# because autoapi docstrings reference package-surface names (e.g.
# ``microquantum.BackendResult``) that are valid Python but not resolvable
# cross-reference targets; see docs/releases/developer-preview.rst.
nitpicky = False