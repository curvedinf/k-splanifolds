# Hadamard KS2 follow-up study

Read [the Markdown report](REPORT.md), which renders directly in GitHub.

The package includes every completed run record, three-seed comparisons, fresh reserved scores, timing rounds, numerical and protocol checks, figures, and a content-addressed archive of exact training sources. FFN learned parameter counts are separate from fixed matrix storage.

Training text, reserved text, model checkpoints, and optimizer states are intentionally excluded. Their hashes and local provenance paths remain in the evidence records. `source_bundle.tar.gz` contains an index mapping each executed source file to its SHA256-named blob; current analysis tooling is indexed separately. `prior_context/` preserves records for the earlier reference checkpoints scored on the new reserved interval.
