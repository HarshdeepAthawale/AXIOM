"""Source files in, :class:`~axiom.schema.Chunk` records out.

The only place in Axiom where a ``chunk_id`` or a ``content_hash`` is minted
(Rules.md Rule 1), and the first stage of the offline index build
(Appflow.md Flow 1, step 2).

Public surface:

* :func:`chunk_file` -- chunk one file. The canonical entrypoint named in
  Appflow.md, Rules.md section 3, and TechSpecifications.md section 3.7.
* :func:`chunk_repo` -- chunk a whole tree, applying the NG-26 ignore list.
* :func:`discover_source_files` -- the file list ``chunk_repo`` walks, exposed
  so the CLI can report or filter it before chunking starts.
* :func:`estimate_tokens` -- the deterministic token estimator the size bounds
  are measured with.
* :func:`split_text`, :func:`regex_chunks`, :func:`module_chunk` -- the lower
  rungs of the degradation ladder, individually addressable so tests can drive
  one rung without uninstalling ``tree-sitter``.

Importing this package pulls in no optional dependency: ``tree_sitter`` is
imported inside the function that needs it and nowhere else (NFR-07).
"""

from __future__ import annotations

from .ast_chunker import chunk_file, chunk_repo, discover_source_files
from .fallback import SourceIndex, module_chunk, regex_chunks, split_text
from .tokens import estimate_tokens

__all__ = [
    "SourceIndex",
    "chunk_file",
    "chunk_repo",
    "discover_source_files",
    "estimate_tokens",
    "module_chunk",
    "regex_chunks",
    "split_text",
]
