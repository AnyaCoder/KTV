# Segment Tree 0.62 Method Note

This note paperizes the current best 100-sample line in `run_eval_swarm_openai.py` and separates the method core from engineering robustness and ablation-only options.

## Target Configuration

The strongest current 100-sample setting is:

- `segment_tree_leaf_source=keyframe_windows`
- `segment_tree_plan_style=soft_temporal_chain`
- `segment_tree_payload_mode=representative_temporal_guard`
- `temporal_guard_strategy=edge_and_gap`

This line reached `62/100 = 0.62` on the STAR random-100 subset with about `6.82` final frames per question.

## Method Core

The method can be described as a three-part training-free routing pipeline.

First, the video clip is sampled every `0.5s`, but the tree leaves are not treated as uniform single frames. Instead, the method reuses KTV-style keyframe anchors and converts them into chronological temporal windows. Each window becomes a leaf interval, and the representative frame of that interval is chosen near the anchor time. This gives the tree a semantically informed non-uniform partition instead of a flat uniform timeline.

Second, before hierarchical routing, a small model generates a short visibly grounded evidence plan from the question and options. The plan does not answer the question. It only describes what should be preserved, using a soft temporal chain with an anchor event, short temporal relations, and a few stage labels. This plan is then injected into every internal node decision so the selector expands an interval whenever collapsing it to one frame would risk losing stage transitions, identity evidence, or temporal ordering.

Third, after the tree frontier is selected, the final payload sent to the large model remains intentionally light. The base payload uses only one representative frame per frontier node. A lightweight temporal coverage guard is then added only when the evidence plan indicates temporal-chain reasoning. In the current best line, the guard restores sparse continuity cues from the temporal edges and the largest uncovered gap, rather than attaching dense node summaries everywhere.

## Operational View

At runtime, the pipeline is:

1. Sample the clip at `0.5s` intervals.
2. Partition sampled frames into KTV-anchored temporal windows.
3. Ask the small model for a soft grounded evidence plan.
4. Build a segment tree over the leaf windows.
5. For each internal node, choose one representative child and decide whether the node must stay expanded.
6. Collect the frontier representatives.
7. If the evidence plan implies temporal reasoning, add a very small number of guard frames for temporal coverage.
8. Send the final ordered frame set to the large VLM for answer prediction.

## What Belongs In The Main Method

The main method should emphasize only the following components:

- KTV-style keyframe-window tree construction
- Soft temporal-chain-guided hierarchical expansion
- Lightweight temporal coverage guard on top of representative-only frontier routing

This is the cleanest story because the gain does not come from making every node denser. It comes from better temporal partitioning, better grounded routing, and a very sparse continuity correction at the end.

## What Should Be Treated As Ablations

The following pieces should remain as ablations or implementation alternatives rather than the headline method:

- `segment_tree_summary_mode=boundary_representative`
- `segment_tree_payload_mode=node_summary`
- `segment_tree_payload_mode=sampled_in_frontier`
- `segment_tree_leaf_source=sampled_frames`
- `segment_tree_plan_style=grounded`
- `segment_tree_prompt_mode=question_aware_guard`
- `temporal_guard_strategy=edge_only`
- `temporal_guard_strategy=largest_gap`

Current evidence suggests `node_summary` is not the right mainline direction. It increases image count but did not outperform the lighter representative-based path. The current best result comes from sparse temporal guarding, not from pushing more images through every node.

## Hack Audit

The current pipeline is not obviously hacked in the sense of leaking answers or hardcoding dataset labels. The prompts explicitly forbid answering during planning and focus on visually grounded evidence only. The tree decisions are generic interval decisions rather than task-specific if/else branches.

The main heuristic concentration is in the temporal guard stage. The rule that activates the guard from temporal relations or multiple stages is hand-designed. The exact strategy `edge_and_gap` is also heuristic: it restores boundary frames and one large uncovered gap frame. This is lightweight and defensible, but it is still the most handcrafted part of the pipeline.

There are also engineering robustness heuristics that should not be presented as algorithmic novelty. These include JSON recovery, loose parser fallbacks for small-model outputs, and fallback from `keyframe_windows` to `sampled_frames` when keyframe metadata is unavailable. These choices are useful for running experiments, but they should be separated from the method contribution in any writeup.

## Recommended Paper Framing

A clean paper-level framing is:

We construct a KTV-anchored temporal segment tree, guide hierarchical expansion with a grounded soft temporal evidence plan, and apply only a sparse temporal coverage guard at the final frontier. This preserves long-range event structure without relying on dense per-node summaries or model fine-tuning.

The key empirical message is:

Better performance comes from preserving the right temporal structure with minimal extra visual payload, not from uniformly increasing the number of frames shown to the large model.
