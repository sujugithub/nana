# Nana workspace

Nana's native workspace runs at `/workspace` on port 8642, alongside its
water chat GUI and existing routing runtime. See [the current handoff](../handoff.md)
for application status and remaining work.

## Start on this Mac

Connect the T7, then run from the project directory:

```sh
.venv/bin/python scripts/start_workspace.py --real
```

Open http://127.0.0.1:8642/workspace or http://127.0.0.1:8642/chat.
The bundled Ollama executable is a gitignored local installation under
`data/runtime/ollama`. The launcher uses port 11435 and model storage at
`/Volumes/T7/NanaModels`, keeping other Ollama installations separate.
Override `--model-dir` or `--ollama-port` for another machine. Downloads are
explicit actions in Local models. The selected model/backend is user-scoped.

Without `--real`, model requests are mocked and image/model downloads are
disabled. Notes/tasks/documents and connection configuration still work.
Mock runs verify wiring, not model quality.

## Google setup

1. Open [Google Cloud Console](https://console.cloud.google.com/) and create
   a project named **Nana**, or select your existing project.
2. Open **APIs & Services → Library**. Search for and enable **Gmail API**
   and **Google Calendar API** in that project.
3. Open **Google Auth Platform → Branding**. If shown, select **Get started**.
   Use **Nana** for the app name and your email for support/contact details.
4. Under **Audience**, choose **External** for a personal Gmail account.
   Keep the app in **Testing** and add your own Gmail address under **Test users**.
5. Under **Data Access**, add these requested scopes:
   `https://www.googleapis.com/auth/gmail.modify` and
   `https://www.googleapis.com/auth/calendar.events`.
6. Under **Clients → Create client**, choose **Desktop app** and name it
   **Nana on my Mac**. Create it, then copy the client ID and client secret.
   Desktop clients use a loopback callback; no manually entered redirect URI
   is needed. A Web application client is a different client type.
7. Open [Nana Connections](http://127.0.0.1:8642/workspace#connections).
   Paste the credentials into **OAuth client ID** and **OAuth client secret**,
   then select **Save client & connect**. Enter them here rather than in chat.
   Once saved, the button is **Connect Google**; leave the credential fields
   blank to reuse the saved client when reconnecting.
8. Sign in to the same Google account added as a test user and review the
   Gmail/Calendar permissions. The callback returns to Nana and should show
   **Connected · Gmail & Calendar**. Keep Nana running during this step.

If Google reports access blocked, check the selected Cloud project, enabled
APIs and test-user address. If it reports `redirect_uri_mismatch`, check that
the client is **Desktop app**, not **Web application**. If Google labels your
development app unverified, check its name and client against your own Cloud
project before continuing. In external Testing mode, these scopes normally
require reconnection after seven days; publishing publicly is a separate
verification step.

Provider HTTPS uses certifi's CA bundle with certificate and hostname checks
enabled. This also works with python.org macOS installations missing a system
CA setup. After a failed OAuth callback, start a fresh connection from Nana;
the previous callback state and authorization code must not be replayed.
Multiple sign-in tabs have independent, bounded, ten-minute requests. A callback
consumes only its own request; stale links display a recovery page with a link
back to Connections.

Nana uses a loopback redirect, random one-use state and PKCE. Tokens are stored
in user-scoped `connections.json` with mode 0600; APIs only report readiness,
never the token or client secret. Account passwords are not Google passwords.
Scopes: Gmail modify and Calendar events. In testing mode Google may require
periodic reconnection. API enablement, OAuth consent and possible app
verification are managed in the owner's Google project.

References:
https://developers.google.com/identity/protocols/oauth2/native-app
https://developers.google.com/workspace/guides/configure-oauth-consent
https://developers.google.com/workspace/gmail/api/reference/rest
https://developers.google.com/workspace/calendar/api/v3/reference

Email sending requires a saved draft and an explicit review/send action.
If a send fails after submission, its sending marker remains: inspect Gmail
Sent before retrying, since the provider may already have accepted it.

## Search, images and MCP

- Search: configure a SearXNG base URL with JSON output enabled. The source
  reader can read public text/HTML URLs without a search provider.
- Images: configure a full generation URL, model ID and API key for a service
  supporting the images/generations JSON format and `b64_json` output. No
  image-provider credentials are assumed from the chat provider.
- MCP: configure a JSON mapping such as
  `{"example":{"command":["/absolute/path/to/server","argument"]}}`.
  Discovery starts that process; review it first. Each agent MCP call is
  also reviewed. Alternatively configure `{"url":"https://server.example/mcp"}`
  for Streamable HTTP JSON/SSE. Host MCP execution is owner-only.
- Shell: commands require a review step and run with workspace files as cwd.
  This cwd is not an OS sandbox; approved commands can access other files.
- Browser: reviewed read/click/fill actions use an isolated Chromium profile
  for each Nana user, separate from their normal Chrome profile. Install its
  browser runtime with `.venv/bin/python -m playwright install chromium`.

## Data and accounts

Workspace records, revisions, schedules and run logs live in
`data/workspace/<user>/workspace.db`. Assets live in that user's files
directory. Existing owner chats retain `data/nana-chat.db`; additional users
have separate chat databases. Runtime data is gitignored.

Accounts are optional on the localhost demo. Creating the first account assigns
existing owner data to it and enables sign-in for all routes. Additional
accounts can be created by the owner. TOTP is enabled only after verifying a
code; keep access to the authenticator. Accounts are not a deployment guide:
this server remains bound to localhost.
Host command/MCP execution is restricted to the owner account. Other users
cannot gain access to owner files by approving arbitrary host commands.

Saved memories and saved instructions are inserted into later chats and
agents. Delete/edit them in the workspace. Original research experiments can
still run directly through the CLI without this chat context.

Agent requests execute through `webui.service.execute`, which uses the
existing routing runtime. Each model step records its route/model/provider,
tokens and estimated cost; tool results and sources are inspectable. Tools
use a bounded JSON planning protocol, not native provider function calling.

Chat streams actual Fireworks/Ollama output. Hybrid can replace a provisional
cheap answer when escalation starts. Stop cancels at the next provider event;
the unfinished turn is not saved. A cancelled remote call may still be billed;
usage that the provider did not return is unknown, not zero.

Scheduled reminders/agents run only while Nana is running. Interrupted jobs
are visible rather than silently rerun after a crash. Scheduled agents receive
read-only tools and do not send email or modify external calendars.

See WORKSPACE_PLAN.md for remaining feature-parity work and live checks.
