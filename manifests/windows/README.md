# Window manifests

`strict_public_seed17_legal_start_fraction.json.gz` is the frozen manifest naming every
window in the strict pretraining corpus by session and local start time. It fixes the corpus
at 353,127 windows, and every run record pins its hash, so a result can always be traced to
the exact set of windows it was computed over.

It is stored gzipped because the uncompressed JSON is 64 MB and compresses to 1 MB — the
content is highly repetitive. Every consumer reads either form; nothing needs unpacking by
hand.

    uncompressed sha256  af559560321148d625114677ad3eb157635157df1fa52809ebff316753417398
    uncompressed bytes   66,890,713

The hash above is of the *uncompressed* bytes, which is what run records store, so the
provenance chain is unchanged by the compression.
