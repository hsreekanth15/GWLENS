To prepare the documentation run
```bash
make html
```
# GWLENS documentation

Install the documentation dependencies and build the HTML:

```bash
pip install -r requirements.txt
make html
```

The output is in `build/html/`. 

To build and preview it live locally:

```bash
make live-html
```

It uses `sphinx-autobuild` to have the live preview. If the port is occupied, you can ask for different ports:

```bash
make live-html O="--port 8765"
```

The site is published to GitHub Pages automatically by `.github/workflows/docs.yml` on every push to `main`.
