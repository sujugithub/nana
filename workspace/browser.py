"""Isolated, bounded browser tools for agent tasks, separate from user Chrome."""
import json
import threading
from . import store, tools

_LOCK = threading.RLock()


def action(args):
    """Each call is reviewed by the user before launching the browser."""
    url = str(args.get("url", ""))
    tools.public_url(url)
    command = args.get("action", "read")
    if command not in {"read", "click", "fill"}:
        raise ValueError("Browser action must be read, click or fill")
    selector = args.get("selector", "")
    if command != "read" and (not isinstance(selector, str) or not selector):
        raise ValueError("Browser interaction needs a selector")
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise RuntimeError("Install browser tools with: pip install playwright; python -m playwright install chromium") from None
    with _LOCK, sync_playwright() as playwright:
        profile = store.home() / "browser-profile"
        context = playwright.chromium.launch_persistent_context(str(profile), headless=True,
                    viewport={"width": 1200, "height": 800}, accept_downloads=False)
        try:
            def guard(route):
                try:
                    tools.public_url(route.request.url)
                    route.continue_()
                except Exception:
                    route.abort()
            context.route("**/*", guard)
            page = context.pages[0] if context.pages else context.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=20000)
            if command == "click":
                page.locator(selector).click(timeout=5000)
            elif command == "fill":
                page.locator(selector).fill(str(args.get("value", "")), timeout=5000)
            snapshot = page.locator("body").inner_text(timeout=5000)[:24000]
            links = page.locator("a[href]").evaluate_all("nodes => nodes.slice(0, 60).map(n => ({text:n.innerText,url:n.href}))")
            return {"url": page.url, "title": page.title(), "content": snapshot, "links": links}
        finally:
            context.close()
