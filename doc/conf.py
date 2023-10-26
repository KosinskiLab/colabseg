import os
import sys

sys.path.insert(0, os.path.abspath("../colabseg"))

project = "colabseg"
copyright = "2023 European Molecular Biology Laboratory"
author = "Marc Siggel, Valentin Maurer"
release = "0.1"

extensions = [
    "sphinx.ext.autodoc",
    "numpydoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx_tabs.tabs",
]
copybutton_prompt_text = ">>> "
copybutton_prompt_is_regexp = False

autosummary_generate = True
autosummary_generate_overwrite = True
autosummary_imported_members = True
add_module_names = False

autodoc_typehints_format = "short"

autodoc_typehints = "none"


templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = [
    "css/general.css",
]
html_context = {
    "github_user": "MSiggel",
    "github_repo": "https://github.com/KosinskiLab/colabseg",
    "github_version": "master",
    "doc_path": "docs",
}
html_theme_options = {
    "use_edit_page_button": False,
    "navigation_depth": 3,
    "show_nav_level": 0,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/KosinskiLab/colabseg",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
    ],
}
