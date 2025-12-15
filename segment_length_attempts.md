# segment_length Implementation Attempts

## Goal
Add `segment_length` parameter to `nn.scan` in Linen to enable memory-efficient scanning with checkpointing at segment boundaries, similar to `nn.remat_scan`.

## Error Pattern
All attempts fail with:
```
File ".../jax/_src/lax/control_flow/loops.py", line 221, in scan
    lengths = [x.shape[0] for x in xs_flat]
               ~~~~~~~^^^
IndexError: tuple index out of range
```

This error occurs when `jax.lax.scan` flattens `xs` and tries to get the shape of each leaf. It fails when a leaf doesn't have a `.shape` attribute or has an empty shape (like `()` or strings).

---

## Attempt 1: Dynamic Slice with Segment Index

**Approach:**
- Reshape vars/rngs from `[length, ...]` to `[num_segments, segment_length, ...]`
- Pass segment indices `[0, 1, 2, ...]` through outer scan
- Use `jax.lax.dynamic_slice_in_dim(x, segment_idx, segment_length, axis=0)` to extract each segment

**Why it failed:**
- User reported dynamic slicing confuses sharding annotations
- XLA may not optimize dynamic slicing well for sharded arrays

---

## Attempt 2: Pass All Args Through Scan with () Placeholders

**Approach:**
- Replace broadcast args with `()` (empty tuple) placeholders
- Pass the modified args through scan
- Reconstruct by combining broadcast (closed over) with scanned args

**Why it failed:**
- `()` is still a leaf in the pytree
- When `jax.lax.scan` flattens `xs`, it sees `()` which has no `.shape`
- Error: `tuple index out of range`

---

## Attempt 3: Filter Broadcast Args, Pass Only JAX Arrays

**Approach:**
- Flatten args and separate into:
  - `scanned_args_list`: list of JAX arrays (reshaped)
  - `broadcast_args_dict`: dict mapping position -> broadcast value
- Pass `(segmented_vars, segmented_rngs, *scanned_args_list)` through scan
- Reconstruct args inside body using dict and positions

**Why it failed:**
- Same error - something in `outer_xs` still lacks `.shape`
- Possible causes:
  - `segmented_scan_vars` is a tuple of dicts, some may be empty `{}`
  - `segmented_rng_groups` may contain empty structures
  - `scanned_args_list` might be empty if all args are broadcast

---

## Attempt 4: Close Over Args, Pass Only Index

**Approach:**
- Close over all reshaped args
- Pass segment indices through scan
- Use `x[seg_idx]` indexing inside body

**Why it failed:**
- `x[seg_idx]` where `seg_idx` is traced = dynamic indexing
- Same sharding issues as dynamic_slice

---

## Attempt 5: Flatten All Structures to Plain List of Arrays

**Approach:**
- Flatten vars, rngs, and args into a single flat list of JAX arrays
- Store treedefs for reconstruction inside `process_segment`
- Pass only the flat list through `jax.lax.scan`
- Reconstruct structures inside the body using stored treedefs

**Why it failed initially:**
- `in_axes` is a **prefix tree** that doesn't match the full structure of `args`
- When flattening `in_axes` and `args` separately:
  - `flat_in_axes` might have N elements (one per top-level arg)
  - `flat_args` might have M elements (one per leaf)
- The `zip(flat_in_axes, flat_args)` would only iterate `min(N, M)` times
- This caused `scanned_args_list` to have fewer elements than expected
- Error: `IndexError: list index out of range` when reconstructing

**Fix attempt 1 (failed):**
- Use `jax.tree.map(lambda ax, _: ax, in_axes, args)` to expand `in_axes` to match `args` structure

**Why fix attempt 1 failed:**
- `jax.tree.map(lambda ax, _: ax, in_axes, args)` returns a tree with `in_axes` structure, NOT `args` structure
- The lambda `lambda ax, _: ax` discards the second argument entirely, so output preserves first arg's structure
- Example: `in_axes=(broadcast, broadcast, broadcast, broadcast)` (4 elements), `args=(a, b, c)` (3 leaves)
- Result: `flat_expanded_in_axes=4`, `flat_reshaped_args=3`, `flat_orig_args=3`
- Error: `AssertionError: Length mismatch`

**Fix attempt 2 (current):**
- Replace broadcast args with `()` (empty tuple) which has no leaves when flattened
- Use `jax.tree.map(make_scanned_args, in_axes, reshaped_args)` where broadcast → `()`
- Flatten to get only scanned arg leaves: `scanned_args_list = jax.tree.leaves(scanned_args_tree)`
- Reconstruct using `tree_map` to combine original args (for broadcast) with scanned values

---

## Root Cause Analysis

The fundamental issue is that `jax.lax.scan` requires ALL leaves in `xs` to be JAX arrays with a consistent leading dimension. The structures we're passing contain:

1. **Variable groups**: `tuple[dict[str, dict[str, Array]], ...]`
   - May contain empty dicts `{}`
   - Empty dicts are leaves but have no shape

2. **RNG groups**: Similar nested structure
   - Non-split RNGs are passed as-is (not reshaped)
   - May have mismatched structures

3. **Args**: User-provided, may contain:
   - Strings (broadcast)
   - None values
   - Empty containers

---

## What Works: remat_scan

`remat_scan` avoids these issues by:
1. Using **nested lift.scan calls** (not `jax.lax.scan` directly)
2. Each `lift.scan` handles its own variable axis management
3. The scan machinery already knows how to handle broadcast variables
4. No manual reshaping of variables needed

```python
# remat_scan structure (simplified)
@remat
def inner_loop(scope, carry):
    return remat_scan(body_fn, lengths[1:], ...)(scope, carry)

scan_fn(inner_loop, length=lengths[0])(scope, carry)
```

---

## Potential Solutions

### Solution A: Use nested lift.scan (like remat_scan)
- Restructure to call `lift.scan` recursively
- Outer scan: `length=num_segments`, vars shaped `[num_segments, segment_length, ...]`
- Inner scan: `length=segment_length`, wrapped in `jax.checkpoint`
- Challenge: Need to reshape vars before entering scan machinery

### Solution B: Filter at pack/unpack level
- Modify how we pack `outer_xs` to exclude empty structures
- Ensure all leaves are valid JAX arrays
- Handle reconstruction carefully

### Solution C: Use axes_scan.scan for outer loop too
- `axes_scan.scan` handles broadcast values via `axes_scan.broadcast`
- But nested `axes_scan.scan` calls had issues with `init_mode` tracing

### Solution D: Modify scan to accept segment_length natively
- Push segmentation logic deeper into the scan implementation
- Handle at the `axes_scan.scan` level where broadcast is already supported

---

## Sharding Preservation Through Reshape

### The Problem

When reshaping arrays for segmentation:
- Input: `[length, ...]` with PartitionSpec like `('fsdp', None, ...)`
- Reshape to: `[num_segments, segment_length, ...]`
- **Sharding metadata is lost** during reshape
- JAX's SPMD compiler sees ambiguous sharding for new dimensions

### Why remat_scan Doesn't Have This Problem

`remat_scan` uses **nested scans** instead of reshaping:
- Variables are expected to already be shaped `[outer, inner, ...]`
- Each nested `scan()` call consumes one axis
- No reshape = no sharding loss

### Solution: Explicit Sharding Re-application

**Step 1: Get original sharding before reshape**

```python
def get_sharding_spec(x):
  if hasattr(x, 'aval') and hasattr(x.aval, 'sharding') and x.aval.sharding is not None:
    return x.aval.sharding.spec
  return None
```

**Step 2: Apply new sharding after reshape (insert None for segment dim)**

```python
def apply_segmented_sharding(x, orig_spec):
  if orig_spec is None:
    return x
  # Insert None at position 0 for segment dimension
  # Original spec[0] moves to spec[1] (segment_length dim)
  new_spec = jax.sharding.PartitionSpec(None, *orig_spec)
  mesh = meta.get_global_mesh()
  if mesh is not None:
    sharding = jax.sharding.NamedSharding(mesh, new_spec)
    return jax.lax.with_sharding_constraint(x, sharding)
  return x
```

**Step 3: Restore original sharding after unreshape**

```python
def restore_sharding(x, orig_spec):
  if orig_spec is None:
    return x
  # orig_spec is (None, actual_spec...) - remove the leading None
  if len(orig_spec) > 0 and orig_spec[0] is None:
    restored_spec = jax.sharding.PartitionSpec(*orig_spec[1:])
  else:
    restored_spec = orig_spec
  mesh = meta.get_global_mesh()
  if mesh is not None:
    sharding = jax.sharding.NamedSharding(mesh, restored_spec)
    return jax.lax.with_sharding_constraint(x, sharding)
  return x
```

### Example Transformation

```
Input:  shape=[32, 768], spec=('fsdp', None)
  ↓ reshape to [4, 8, 768]
  ↓ apply_segmented_sharding
Segmented: shape=[4, 8, 768], spec=(None, 'fsdp', None)
  ↓ process segments...
  ↓ reshape back to [32, 768]
  ↓ restore_sharding
Output: shape=[32, 768], spec=('fsdp', None)
```

### Key Insight

The segment dimension (axis 0 after reshape) should be **unsharded** (`None` in PartitionSpec) because:
- Each segment is processed sequentially by the outer scan
- The original sharding applies to the `segment_length` dimension (axis 1)
- This preserves the data distribution across devices
