# Optional Arch-Router integration

Built with DigitalOcean.

Nana's experimental adapter uses the prompt format described by
[Katanemo's Arch-Router-1.5B model card](https://huggingface.co/katanemo/Arch-Router-1.5B).
Model weights are installed separately and are not redistributed with Nana.
See the upstream model repository's license and notices for its terms.

## Settings and behavior (25 October 2026)

Workspace → Settings → Router offers three choices, saved per user:

| Choice | Behavior | Limitation |
| --- | --- | --- |
| Heuristic | Existing keyword rules; no extra router model | Quality depends on the workload |
| Learned | Existing trained routing artifact | Requires a compatible artifact for the answer model pair; incompatible requests are rejected |
| Arch-Router | Optional local intent classification through loopback Ollama | Experimental intent-to-tier mapping, without a current-pair quality or savings claim |

The selection applies to Auto/Hybrid chat and hybrid workspace agents on the
next request. Explicit local or cloud model choices bypass the router. Existing
chats, instructions, model selections and accounts are retained. Hybrid can
still call the configured cloud answer model and incur its normal charges.

Arch-Router uses `ARCH_ROUTER_MODEL` (default
`hf.co/katanemo/Arch-Router-1.5B.gguf:Q4_K_M`) via the configured loopback Ollama
server. It does not download weights. It sends at most three recent user or
assistant turns, bounded to 2,000 characters total, without system messages,
tools or image payloads. Its `local` and `strong` route descriptions are Nana
policy, not measured predictions of answer quality. It releases router memory
after classification, which can add model reload latency on the next request.

An unavailable model, invalid response or unsupported route uses heuristic
fallback, reported in reply warnings and routing metadata. Preview mode reports
`arch-preview` and simulates the route with heuristics; it does not invoke the
model. Cancellation is propagated rather than converted to fallback.

## Verification and remaining work

The settings, persistence, request propagation, model bypasses, parser,
cancellation and fallback were verified offline on 25 October 2026. The full
274-test suite and mock routing harness passed. No weights were loaded or
downloaded and no paid inference was used for this verification.

A fresh live Arch-Router decision remains unverified because the T7 model drive
was disconnected. Current-pair quality, latency and cost comparisons remain
required before recommending Arch-Router over the other choices. The earlier
spike and benchmark evidence in `ARCH_ROUTER_PLAN.md` is historical and does
not validate this settings integration. Feedback training and the rest of that
plan are not implemented by this change.
