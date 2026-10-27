# Nana launch presentation

The launch page and its assets are saved in `webui/static/launch.html` and
`webui/static/launch/`. They include the API-cost problem section, Reef Rally,
recorded local-run evidence, architecture, tools, routing results and team.
The matching speaking notes are in `presentation-script.md`.

## Preview the static presentation

From this branch's checkout:

```sh
python3 -m http.server 8643 --bind 127.0.0.1 --directory webui/static
```

Open <http://127.0.0.1:8643/launch.html>. Scroll through the presentation.
The game, screenshots and bundled evidence work in this preview. Chat,
workspace, routing explainer and the Nana home link require the full app.
Reef Rally requests its display fonts from Google Fonts.

## Add the page to the Nana app

The app source is on `main`; `privet` is the presentation-material branch.
`launch-page-integration.patch` contains the matching server route and HTTP
tests, based on main commit `6610d66`. In a main-based checkout, copy the
launch HTML and launch directory into `webui/static/`, then apply the patch:

```sh
git apply /path/to/launch-page-integration.patch
.venv/bin/python -m unittest tests.test_webui
.venv/bin/python scripts/start_workspace.py --real
```

The integrated page is <http://127.0.0.1:8642/launch>. Starting real mode
enables live model requests; cloud requests incur the configured API charges.
No credentials, live chat database or workspace storage are included here.
