# Agentic Improvement Workflow

One task at a time. The full cycle goes from adopting the best known ONNX nets, syncing them to editable Python source, improving one task, rebuilding, and submitting.

---

## Score Formula

```
points = max(1, 25 - ln(params + memory))
```

- **params** = total element count of all ONNX initializers
- **memory** = sum of `(num_elements × itemsize)` for every intermediate tensor (shape-inferred, excludes input/output)
- Goal: reduce `params + memory` while keeping `pass = 266/266`

---

## Full Pipeline (start here after receiving new bundles)

| Step | Script | Purpose |
|---|---|---|
| 1. Adopt best nets | `merge_network_dirs.py` | ONNX → `networks/` |
| 2. Source sync | `live_to_exact_source.py --write-src` | `networks/` → `src/custom/` (baseline) |
| 3. Verify | `source_live_reconcile.py` | Confirm `mismatches: 0` |
| 4. **Improve score** | Edit `src/custom/taskNNN.py` | Semantic/cheaper rewrites |
| 5. Rebuild | `rebuild_networks_from_source.py` | `src/custom/` → `networks/` |
| 6. Submit | `from src.pipeline import pack; pack()` | Zip for Kaggle |

Steps 1–3 are one-time setup when new bundles arrive. Steps 4–6 repeat per task.

---

### Step 1 — Adopt best nets

Merge new submission bundles into `networks/`, keeping the best ONNX per task:

```bash
PYTHONPATH=. .venv/bin/python reports/scripts/merge_network_dirs.py \
  --sources submission/merged_nets submission_XXXX.XX submission_YYYY.YY \
  --static-only \
  --to-networks \
  --write-manifest \
  --pack
```

### Step 2 — Sync networks → source

Convert the live `networks/*.onnx` into editable Python builders in `src/custom/`:

```bash
PYTHONPATH=. .venv/bin/python reports/scripts/live_to_exact_source.py --write-src
```

### Step 3 — Verify sync

```bash
PYTHONPATH=. .venv/bin/python reports/scripts/source_live_reconcile.py
```

Must show `mismatches: 0` before proceeding.

---

## Per-Task Improvement Loop (Steps 4–6, repeat)

### Step 4a — Pick a task

Read the tasklog for open angles and current score:

```bash
cat reports/tasklog/taskNNN.md
```

### Step 4b — Read the task data

Understand the input→output transformation before touching code:

```python
PYTHONPATH=. .venv/bin/python -c "
import json
d = json.load(open('data/taskNNN.json'))
for i, ex in enumerate(d['train']):
    print(f'--- Example {i} ---')
    for row in ex['input']: print(row)
    print('Output:')
    for row in ex['output']: print(row)
"
```

### Step 4c — Measure current score

```bash
PYTHONPATH=. .venv/bin/python reports/scripts/measure_task.py NNN
```

Output: `{'ok': True, 'pass': 266, 'fail': 0, 'memory': XXXX, 'params': YY, 'points': ZZ.ZZ}`

### Step 4d — Inspect intermediate tensor memory breakdown

Find which tensors are wasting bytes:

```python
PYTHONPATH=. .venv/bin/python -c "
import onnx, numpy as np, math, importlib
from src.harness import load_task
import src.custom.taskNNN as mod
importlib.reload(mod)
task = load_task(NNN)
m = mod.build(task)
m2 = onnx.shape_inference.infer_shapes(m, strict_mode=True)
total = 0
for vi in m2.graph.value_info:
    t = vi.type.tensor_type
    dtype = onnx.helper.tensor_dtype_to_np_dtype(t.elem_type)
    dims = [d.dim_value for d in t.shape.dim]
    mem = math.prod(dims) * np.dtype(dtype).itemsize
    total += mem
    print(f'  {vi.name}: {dtype} {dims} = {mem}B')
print(f'Total: {total}B')
"
```

Look for `float32` tensors that carry only 0/1 values — they can become `bool` (1B vs 4B per element).

### Step 4e — Apply an optimization

Edit `src/custom/taskNNN.py`. Common patterns:

| Pattern | Saving |
|---|---|
| `Cast(bool→float) + Sub(1, _)` → `Not(bool)` | Keeps mask as bool (1B vs 4B/elem) |
| `Mul(float_mask, Cast(bool))` → `Where(bool, float_mask, 0.0)` | Eliminates Cast intermediate |
| `Gather(float_0_1_mask, idx)` → `Gather(bool_mask, idx)` | Switch Gather data to bool |
| `Sub + Abs + Greater(_, thresh)` → `Equal` (fp16) | When operands are integer-valued |
| `Greater(Cast(bool), 0.5)` → keep bool directly | Collapse redundant threshold |

**CAUTION:** `Min`/`Max` with fp16 inputs crashes ORT under `DISABLE_ALL`. Keep index-clamping in float32.  
**CAUTION:** Einsum requires all inputs to share the same dtype. The `input` tensor is always `float32`, so any Einsum that uses it requires float32 masks — cannot switch those to bool/fp16 without an extra Cast.

### Step 4f — Local eval

```bash
PYTHONPATH=. .venv/bin/python reports/scripts/measure_task.py NNN
```

- `pass` must stay at its full count (e.g. 266)
- `points` must increase
- If it regresses or `ok: False`: revert and try a different angle

### Step 4g — Log the result

Append a dated entry to `reports/tasklog/taskNNN.md`:

```markdown
## SXX (YYYY-MM-DD) — <one-line description> (+X.XXX) ADOPTED, 266/266
<What changed, bytes before→after, score before→after.>
**Transferable:** <general rule usable on other tasks>
```

---

### Step 5 — Rebuild network

```bash
PYTHONPATH=. .venv/bin/python reports/scripts/rebuild_networks_from_source.py --tasks NNN
```

Confirm the output line shows the improved pts/mem/params.

---

### Step 6 — Pack and submit

```python
PYTHONPATH=. .venv/bin/python -c "
from src.pipeline import pack; pack()
"
```

Writes `submission/submission.zip`. Copy to root for easy upload:

```bash
cp submission/submission.zip submission.zip
```

Submit manually via Kaggle UI or CLI:

```bash
set -a && source .env && set +a && \
~/.local/bin/kaggle competitions submit -c neurogolf-2026 -f submission.zip -m "taskNNN: <description>"
```

---

## Optimization Checklist (per task, try in order)

- [ ] **Not + Gather(bool) + Where** — replace Cast+Sub no-marker chains (biggest win, ~90B/elem saved)
- [ ] **bool Gather** — any Gather whose data is a 0/1 float mask
- [ ] **fp16 Equal** — replace `Sub+Abs+Greater` on integer-valued tensors
- [ ] **Collapse Cast+Greater** — `Greater(Cast(bool), 0.5)` → keep as bool
- [ ] **Remove redundant refinement pass** — only if fresh robustness tests confirm safe
- [ ] **Fuse Unsqueeze+Concat pairs** — check if tail structure can be simplified

---

## Key Constraints

- All intermediate tensor shapes must be **statically known** — required by `shape_inference strict_mode=True`
- `fp16 Min/Max` → ORT crash under `DISABLE_ALL` — keep clamping ops in float32
- Mixed-dtype Einsum (e.g. float32 + fp16) → error — all inputs must match
- `input` is always `float32 (1, 10, 30, 30)` — float32 masks required in any Einsum that uses it
- **params** counts elements (not bytes) of initializers — dtype of initializers does not affect the score directly

---

## Quick Reference

| Command | Purpose |
|---|---|
| `measure_task.py NNN` | Score a single task from `src/custom/` |
| `rebuild_networks_from_source.py --tasks NNN` | `src/custom/` → `networks/` for one task |
| `rebuild_networks_from_source.py` | Rebuild all 400 tasks |
| `live_to_exact_source.py --write-src` | `networks/` → `src/custom/` (baseline sync) |
| `source_live_reconcile.py` | Verify mismatches: 0 |
| `merge_network_dirs.py --sources ... --pack` | Merge bundles → `networks/` |
| `from src.pipeline import pack; pack()` | Pack all networks → `submission/submission.zip` |
