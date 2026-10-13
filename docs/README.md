To prepare the documentation run
```bash
make html
```
# GWLENS documentation

If you would like to contribute to the documentation, you can build it locally with edits. The documentation is built using [Sphinx](https://www.sphinx-doc.org/en/master/).

To install the dependencies, you can use pip:

```bash
pip install ".[docs]"
```

Once the dependencies are installed, you can build the documentation with:

```bash
make html
```

This will generate the HTML files in the `build/html/` directory. You can then open the `index.html` file in your web browser to view the documentation.

If you would like to have a live preview of the documentation while you are editing it, you can use the `make live-html` command. This will start a local web server and automatically refresh the page whenever you make changes to the documentation files.

It uses `sphinx-autobuild` to have the live preview. If the port is occupied, you can ask for different ports:

```bash
make live-html O="--port 8765"
```

The site is published to GitHub Pages automatically by `.github/workflows/docs.yml` on every push to `main`.
