# Read-only internet connection — 9 October 2026

## Delivered

Chat now has a visible, off-by-default **Web** control. It is enabled for live Local chat with the installed `qwen3.5:4b` model. With a public URL, Nana uses its existing public webpage reader. Without a URL, a short local native-tool request prepares a query; the selected search provider receives that query. The final answer gets bounded excerpts and no executable tools. Successfully retrieved source URLs are appended to the saved reply and displayed as clickable links.

Connections now offers **Off**, **Exa**, and **SearXNG**. Exa was enabled for the owner's workspace under the user's instruction to choose the best practical provider. Existing SearXNG configurations remain compatible. New accounts with no configured provider remain off. Agent/Research's existing `search` tool uses the same provider adapter.

## Provider decision

Use Exa's hosted, keyless MCP search as the practical initial choice on this Mac. No additional daemon, container runtime, account, API key, model download or model API payment is needed for the verified path. This is hosted search: queries go to Exa. Chat history and saved memories are excluded from query preparation and the provider request. A Web-enabled current request can itself contain personal data, so the UI makes the provider visible and browsing opt-in.

Primary references checked on 9 October 2026:

- [Exa official MCP page](https://exa.ai/mcp): documents its hosted URL and access without an API key.
- [Exa official MCP repository](https://github.com/exa-labs/exa-mcp-server/blob/main/README.md): default search/fetch capabilities and separate authenticated agent capability.
- [OpenCode tools](https://docs.opencode.ai/docs/tools/): describes keyless hosted Exa/Parallel search and the distinction between discovery and URL fetching. [Current provider-selection documentation](https://opencode.ai/v2/docs/websearch) also offers authenticated providers. The hosted path was checked directly rather than assuming both product versions behave identically.
- [Deep Agents research example](https://github.com/langchain-ai/deepagents/blob/main/examples/deep_research/README.md): uses Tavily, requiring an API key.
- [SearXNG container installation](https://docs.searxng.org/admin/installation-docker.html): requires an existing Docker/Podman setup and a local service. Retain this option for a future self-hosting choice.

No upstream implementation was copied or vendored. Nana's small JSON/SSE adapter is original code using the existing urllib/certifi dependencies. The live Exa `tools/list` schema was checked: `web_search_exa` requires `query` and `objective`, with optional `numResults`. Only this read-only operation is called. Generic MCP, privileged tools and external writes remain separate capabilities.

## Acceptance

- Full suite: **234 tests passed in 8.733 seconds**; 12 new focused tests cover grounding, provider parsing, explicit rate limits, source failures, opt-in/preview isolation, query privacy, cancellation, unsupported-model rejection and SSE sources. The stdlib mocked harness and JavaScript syntax checks also pass.
- [Live search fixture](live-web-chat.json): actual 4B query preparation → Exa search → local cited answer. Three official Ollama source links, two model requests, two saved messages in disposable storage, 6.370 seconds, no model API cost. Swap was unchanged at the measurement points.
- [Live public URL fixture](live-web-url-chat.json): actual read of `https://example.com` → local cited answer, 1.378 seconds, two disposable messages. Dispatch auditing confirmed `num_ctx=8192` and the preserved `num_predict=4096` final allowance.
- [UI evidence](web-chat-ui.png): **mocked local completion and source data** in a disposable server at port 8643. The Web toggle, provider hint, source link and completed reply were inspected. Switching to 27B disabled Web and cleared the toggle. Provider Off/Exa changes survived reload. No fixture was written into the user's chats.
- Main app restarted using the existing manual launcher; `/chat`, `/workspace`, `/welcome` return 200. The live Chat config identifies Exa and the installed model choices. The user's chat database hash matches before/after restart. Original test/handoff files and water simulation code match the protected snapshot.

## Limits and next work

- Web Chat is verified only for `qwen3.5:4b` in local Ollama live mode. It does not silently substitute models. Regular Chat and Agent/Research model choices are retained.
- Search is bounded to three source excerpts; URL reading to three URLs, 1 MB per response, and 1,200 excerpt characters per source. Chat answers may need a more specific request or Research mode for longer investigations. Search excerpts are labeled separately from page excerpts; citations establish provenance, not independent fact verification.
- The Web turn checks a 180-second deadline between stages. Blocking network reads can wait until their socket timeout (20 seconds for the page reader, 30 for search). This is not immediate network cancellation.
- **Context discrepancy remains:** 8,192 tokens are requested for the final Web answer, but `/api/ps` reported 4,096 after both live fixtures. The request was audited and the short examples passed; effective full-window retention is unproven. Keep H07 open. The character-based budget is an estimate, and the original 4,096-token output allowance is preserved.
- Exa can rate-limit or change its hosted schema/format. These produce explicit failures, with no paid fallback or automatic account setup. Extend the parser only against a fresh verified provider response.
- Google's HTTP 403 and absent configured general MCP servers remain open integration blockers.

## Rollback

Turn Web off in Chat; choose Off or your SearXNG endpoint in Connections. Code rollback is included in the refreshed delivery-only patch; the protected-file slices are relative to the user's original edited copies. No new schema or model files were installed. Inspect later changes before applying a reverse patch.
