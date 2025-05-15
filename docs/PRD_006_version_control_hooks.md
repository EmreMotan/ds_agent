# PRD‑006: Version‑Control Hooks & Artifact Snapshotting  
*Version 0.1 • May 3 2025*

---

## 1 | Overview  
Adds Git automation for reproducible, auditable outputs.

1. **Branch policy hook** – validates branch names, blocks direct pushes to `main`.  
2. **Post‑merge snapshot** – copies executed notebooks + artifacts into `/episodes/<id>/outputs/DATE_HASH/`.  
3. **Snapshot manifest** – registers each file (type, SHA‑256, path) in `episode.json`.  
4. **CLI helper** – manual snapshot command.

---

## 2 | Problem / Goal  
Enable easy “what exactly shipped?” answers and prevent repo hygiene issues.

---

## 3 | Scope (MVP)

* Git hooks (`scripts/pre-push`, `scripts/post-merge`).  
* Snapshot logic (`src/ds_agent/snapshot.py`).  
* CLI (`bin/snapshot_outputs.py`).  
* Manifest schema update.  
* Tests + sequence diagram.

---

## 4 | Success Criteria  
Pre‑push hook blocks invalid pushes; snapshot within 5 s/10 MB; manifest correct; ≥ 90 % coverage.

---

## 5 | Deliverables  
See table in full spec.

---

*End of PRD*