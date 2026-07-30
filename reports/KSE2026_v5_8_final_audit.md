# KSE 2026 v5.8 Final Artifact Audit

Status: PASS

## Scope and anonymity

- The repository contains the v5.8 allowlisted artifact and these audit reports.
- Text scanning found no author identity, personal or institutional email,
  institution name, personal filesystem path, personal account name,
  persistent researcher identifier, or team website reference.
- Both PDF figures have no Author metadata. PNG metadata contains only the
  fixed artifact software label and resolution.
- Git author and committer identity are anonymous.
- The remote is an anonymous-account repository; no personal-account remote is
  configured.

## Scientific and release checks

- Module D is absent from executable code, generated tables, and both figures.
  README mentions it only to document its removal from the permission path.
- The permission path contains Modules A, B, and E, with E conditional on an
  event-level claim.
- No manuscript PDF, authored source archive, repository history from another
  project, or team-identifying website asset is included.
- The generated workflow and Figure 2 PDFs render correctly with legible text
  and no clipping or overlap.
- The regenerated manifest and `SHA256SUMS.txt` cover every release file except
  the two checksum index files themselves, and the final integrity check passes.
