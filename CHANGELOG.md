<!-- markdownlint-disable MD024 -->
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.1] - 2025-xx-xx

### Added

- Delete examples - Added `examples/delete_examples.py` demonstrating supported deletion operations and workarounds.
- Test coverage - Added tests for ID-based deletion and proper error handling for unsupported operations.

### Changed

- Filter format alignment — Updated _filters_to_zeusdb() to produce a flat dictionary with implicit AND (e.g., { "key": value, "other": { "op": value } }) instead of a nested {"and": [...]} structure, matching the Rust implementation.
- Test infrastructure — Updated _match_filter() in the test fake from the nested format to the flat format to reflect production behavior.

### Fixed

- Filter translation for metadata queries - _filters_to_zeusdb() now emits the flat format expected by the ZeusDB backend. Single filters and AND combinations are handled correctly. The previous nested format could cause filtered queries to return zero results.
- Deletion behavior - Correctly implemented ID-based deletion using `remove_point()`. Delete operations now properly remove vectors from the index and update vector counts.

### Removed
<!-- Add removals/deprecations here -->

---

## [0.1.0] - 2025-09-19

### Added

- Initial release of ZeusDB vector database integration for LlamaIndex.
- Support for connecting LlamaIndex’s RAG framework with ZeusDB for high-performance retrieval.
- Trusted Publishing workflow for PyPI releases via GitHub Actions.
- Build validation workflow to check distributions without publishing.
- Documentation updates including project description and setup instructions.

---

## [Unreleased]

### Added
<!-- Add new features here -->

### Changed
<!-- Add changed behavior here -->

### Fixed
<!-- Add bug fixes here -->

### Removed
<!-- Add removals/deprecations here -->

---
