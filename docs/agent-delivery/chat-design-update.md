# Chat menu and composer model picker — 9 October 2026

The user's design feedback replaced the large attachment/tool checkbox dialog with an anchored icon menu. Files and photos, web search, workspace items and skills appear first. More tools contains saved chats, Gmail, Calendar, file reading and reviewed command/connected/browser tools. Open workspace retains every sidebar destination. The menu dismisses outside/with Escape, supports arrow keys, and returns focus when a tool is selected. Tool names are friendly labels; schema arguments are no longer hover tooltips.

The composer now has a model button beside Send, with a visible chevron. Its menu contains installed local models, the configured cloud choices, and Auto routing. Choosing a model sets both mode and model, uses the existing per-chat settings endpoint, and preserves the draft and attachment selection. New chats use the chosen model. Cloud rows explicitly indicate usage charges. The existing Settings model control remains available; no unsupported reasoning-effort setting was added.

The UI inherits the existing attachment capability checks, tool review/budgets and remote restrictions. The water-canvas source is unchanged. There were no backend, credential, model-weight or user-chat edits in this design increment. Static files are read per request, so the existing port 8642 app loads the design on refresh without a restart.

Acceptance on disposable port 8643:

- Switched an existing fixture chat from local 4B to local 27B, then cloud GPT OSS, and back. Each change saved, with the draft retained. No live inference or paid cloud request was made.
- Uploaded the public PDF through Files & photos, changed the model in the composer, and sent a preview turn. Its persisted assistant model was `qwen3.8:27b`; attachment references were retained.
- Keyboard ArrowDown/Enter enabled Web search; Escape dismissed a submenu. The More tools and workspace destinations were inspected.
- JavaScript syntax check passes. The 13 attachment regressions still pass. The unchanged backend's last complete suite remains 247 passing tests.
- Desktop layout inspected at the browser's default size; responsive CSS retains viewport bounds and larger touch targets. No phone viewport override or phone screenshot acceptance is claimed.

Evidence: `chat-compact-menu-ui.png`, `chat-composer-model-ui.png`, `chat-design-update.json`. These use generated fixture documents and temporary chats; no private user content is included.
