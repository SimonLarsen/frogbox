import pdoc
from pathlib import Path


here = Path(__file__).parent

if __name__ == "__main__":
    pdoc.render.configure(
        docformat="numpy",
        include_undocumented=True,
        show_source=True,
        template_directory="./docs/template",
        logo="/frogbox/logo.png",
    )

    pdoc.pdoc(
        "frogbox",
        output_directory=here,
    )
