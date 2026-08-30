# License options

> **Decision (2026-08-30):** **Position B** — **Apache-2.0 for code, CC-BY-NC-4.0 for model
> weights**. The code is fully permissive (anyone can use, including commercially); the
> weights are research / non-commercial only. A commercial entity that wants to use the
> weights must negotiate with the maintainer (Stephane Fourdrinier) directly.
>
> This document is kept as a record of the decision and the alternatives considered. The
> active files are: `LICENSE` (Apache-2.0, code), `LICENSE-WEIGHTS` (CC-BY-NC-4.0, weights),
> `NOTICE` (combined attribution), `CITATION.cff` (Apache-2.0 for the package), and
> `model_cards/glucofm_encoder.md` (HF Hub license tag).

This is the **decision record**. OpenCGM-StateEvent is being prepared
for release under the GitHub user `sfourdrinier`, with model weights published to Hugging
Face at <https://huggingface.co/sfourdrinier>. The final license is **Position B**.

The sections below lay out the realistic options for **code** and **model weights**
separately, with their trade-offs.

## Code license options

| License | Patent grant | NOTICE / attribution | Ecosystem fit | Practical effect |
|---|---|---|---|---|
| **Apache-2.0** | yes (explicit) | required | matches PyTorch, TensorFlow, ONNX Runtime, scikit-learn | "do what you want, keep NOTICE, no patent trolling" |
| **MIT** | no | minimal | matches React, Next.js, ONNX Runtime Web | "do what you want, no warranty" |
| **BSD-3-Clause** | no | minimal | matches NumPy historically | like MIT but you can't use the author's name to endorse derivatives |
| **MPL-2.0** | yes | required (file-level) | matches Firefox | file-level copyleft; modifies-only-the-MPL-files remain MPL |

For medical-adjacent code where you might integrate third-party ML libraries, Apache-2.0
is the conventional choice because of the patent grant. MIT is the simpler alternative if
you don't expect patent friction.

## Model weights license options

This is the more consequential choice. The released encoder is a binary artifact; the
license you pick controls who can use it commercially, with modifications, etc.

| Weights license | Key clause | Compatible with HF Hub | Effect |
|---|---|---|---|
| **Apache-2.0** | explicit patent grant + NOTICE preservation | yes | most permissive + patent-safe; the default for ML model releases |
| **MIT** | permissive, no patent grant | yes | simplest; users get no patent protection from you |
| **CC-BY-4.0** | attribution, no patent grant | yes | normal for documentation/data; unusual for binary weights |
| **CC-BY-NC-4.0** | non-commercial | yes (HF has a license-tag for this) | forbids commercial use; common for research artifacts |
| **CC-BY-NC-SA-4.0** | non-commercial + share-alike | yes | forbids commercial use; derivatives must carry the same license |
| **OpenRAIL-S** | "responsible AI" license, broad use, narrow restrictions | yes (HF has a license-tag) | allows commercial; restricts a small set of harmful uses (e.g. biometrics for surveillance) |
| **DCLS** | "data + compiler" license; permissive with safety carve-outs | yes | like OpenRAIL but framed for ML/data artifacts |
| **BigScience RAIL** | similar to OpenRAIL-S, more restrictive | yes | less commonly used |
| **Gemma license terms** | bespoke Google-style terms | yes (via HF gated access) | forbids some commercial uses; high friction |
| **Llama 3 community license** | bespoke Meta-style terms | yes (via HF gated access) | >700M MAU restriction; high friction |

### Why this matters for this project specifically

1. **The encoder is a derivative of public CGM data.** Apache-2.0 for weights assumes you
   have the right to ship them. Lane A–D sources are permissive (CC-BY-4.0, ODC-By-1.0,
   CC0, or `verify` for Stanford). Lane E (cgmacros, uchtt1dm, glucofm_bench) has NC/ND/SA
   clauses and **never enters the released weights** — already handled.

2. **HF Hub has first-class license tags.** Apache-2.0, MIT, CC-BY-4.0, CC-BY-NC-4.0,
   OpenRAIL-S, DCLS, Llama-3, and Gemma all show up in the HF model-card UI. The tag
   determines the "Use this model" gating behaviour.

3. **If you want commercial users, you need a permissive weights license.** CC-BY-NC-*,
   OpenRAIL-S-with-restrictions, and the bespoke Llama/Gemma licenses all narrow who can
   use the model in production.

4. **If you want to keep the encoder as a research artifact** (not used in production),
   CC-BY-NC-4.0 is a defensible choice and is common in the research-CGM space.

5. **If you want maximum reach**, Apache-2.0 (or MIT) is the conventional pick.

## Where each licence is recorded

| Artefact | Licence | File |
|---|---|---|
| Code | Apache-2.0 | `LICENSE` |
| Encoder weights | CC-BY-NC-4.0 | `LICENSE-WEIGHTS` |
| Probe heads | CC-BY-NC-SA-4.0 | `LICENSE-HEADS` |

The heads are one step stricter because eight of the eighteen are fitted on CGMacros, whose
terms are share-alike. Keeping them a separate artefact stops that term reaching the encoder,
which never saw CGMacros. Recorded as D025 and enforced by `tests/unit/test_source_rights.py`.
