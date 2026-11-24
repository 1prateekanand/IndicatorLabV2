# Indicator plugins

Drop `*.py` files here. Each decorated function becomes a discoverable indicator. Files that start
with `_` are ignored, so `_template.py` is a reference only.

No agent or frontend code needs to change. Reload from the Indicators page or with
`POST /api/indicators/reload`. Set `IL_INDICATOR_PLUGIN_DIR` to use another folder.
