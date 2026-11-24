# Installation

GWLENS is available on PyPI and can be installed with pip:

```bash
pip install gwlens
```

If the user wants to install the latest development version, they can clone the repository and install it with pip:

```bash
git clone https://github.com/hsreekanth15/GWLENS.git
cd GWLENS
pip install .
```

If you want to install the development version with the documentation dependencies, you can use:

```bash
git clone https://github.com/hsreekanth15/GWLENS.git
cd GWLENS
pip install ".[docs]"
```

Any changes pushed to the `main` branch will automatically trigger a GitHub Actions workflow that builds the documentation and deploys it to GitHub Pages.

The documentation is built using [Sphinx](https://www.sphinx-doc.org/en/master/). Once the dependencies are installed, you can build the documentation with:

```bash
cd docs
make html
```

This will generate the HTML files in the `build/html/` directory. You can then open the `index.html` file in your web browser to view the documentation.

If you would like to have a live preview of the documentation while you are editing it, you can use the `make live-html` command. This will start a local web server and automatically refresh the page whenever you make changes to the documentation files.

It uses `sphinx-autobuild` to have the live preview. If the port is occupied, you can ask for different ports:

```bash
cd docs
make live-html O="--port 8765"
```
