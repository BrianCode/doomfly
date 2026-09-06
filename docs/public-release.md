# DOOMFLY public source snapshot

This repository contains the MaleCNS/ViZDoom simulator, experimental learning
candidates, spectator app, tests, original arenas and scientific evidence. The
public branch starts with one initial commit, using a GitHub noreply identity.

Large graph downloads, mutable checkpoints, raw operational logs, dependencies,
credentials, machine diagnostic dumps, unrelated simulations, external research
workbooks/PDFs and separately commissioned social banners are excluded. Source
registries and hashes support retrieving the required scientific inputs.

Original project code uses MIT. Copied components and game-derived content retain
their required notices. Dataset and research-data attributions identify source,
license and project modifications. See `THIRD_PARTY.md`, `THIRD_PARTY_NOTICES.md`
and `licenses/`. This documents licensing scope; it is not legal or trademark
clearance and does not certify scientific validity.

The MaleCNS importer and transmitter-sign helper are now in `doom/`. Their graph
retention and sign rules remain unchanged. Historical source snapshots use the
new import paths; their old recorded hashes still identify the original runs.
No measured result or failed scientific gate has been rewritten as a success.

The three bundled source archives are publication copies: unrelated helpers and
external workbooks are removed, required MaleCNS helpers and notices are added,
and import paths are updated. They are not byte-for-byte original run archives.
Each archive includes `PUBLICATION-NOTES.md` and a fresh member-hash manifest.
`doom-ui/public/source-archives.json` records the published ZIP hashes; the
learning-iterations manifest also identifies its actual published members.

Privacy processing removes user-specific paths, temporary hostnames and optional
image metadata where found. Generic localhost/container bind settings remain.
Public third-party authorship and copyright notices are preserved verbatim; they
are attribution, not private information about the project maintainer.

`public-source-manifest.json` hashes the actual repository files excluding itself.
Historical provenance hashes elsewhere identify original experimental inputs or
run artifacts, which may be deliberately omitted or repackaged here. A recorded
hash is not a promise that its artifact is included in the public checkout.

Release verification: all 85 DOOMFLY tests passed. The extracted importer produced
the same full 166,700-neuron table as the original MaleCNS import, including all
annotations. All three source ZIPs passed member-hash, license-presence and
DOOMFLY-only scope checks. These checks validate packaging and implementation;
they do not change the failed scientific learning gates.
