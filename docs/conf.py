"""Sphinx configuration for FlashQuad documentation."""

project = "FlashQuad"
copyright = "2025, Ze Ouyang & Zijian Yi"
author = "Ze Ouyang, Zijian Yi"

extensions = [
    "myst_parser",
    "autodoc2",
]

autodoc2_packages = [
    "../flashquad",
]
autodoc2_render_plugin = "myst"

myst_enable_extensions = [
    "colon_fence",
    "fieldlist",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "FlashQuad"
html_theme_options = {
    "source_repository": "https://github.com/FlashQuad-dev/flashquad",
    "source_branch": "main",
    "source_directory": "docs/",
}
