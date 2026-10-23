## Findings and pursuit target

**The main target is the Pareto frontier of learned FFN size × training-step duration versus loss.** A candidate is useful when it lowers loss at comparable combined cost, lowers cost at comparable loss, or adds a useful tradeoff between existing frontier points. The following table uses fresh reserved text and five-round same-GPU timing medians, at 15,000 updates. All entries have three distinct training seeds; repeated runs within a seed are averaged first.

| Frontier configuration | Learned / FFN | ms/update | Combined cost × MLP | Fresh loss | Seed SD |
| --- | ---: | ---: | ---: | ---: | ---: |
| had_spine_r8_frozen + final-side prune | 1,632 | 9.351 | 0.0284 | 1.684414 | 0.003719 |
| pareto_frozen_r8_f1024_s2048 + final-side prune | 1,632 | 10.134 | 0.0308 | 1.682497 | 0.004796 |
| pareto_r8_warps_only_f512_s2048 + final-side prune | 1,760 | 9.463 | 0.0310 | 1.682154 | 0.003929 |
| pareto_frozen_r8_f2048_s2048 + final-side prune | 1,632 | 11.711 | 0.0355 | 1.680042 | 0.001877 |
| had_spine_r8_relu2 | 5,344 | 9.717 | 0.0966 | 1.679715 | 0.001474 |
| ks2_c8 | 18,848 | 7.814 | 0.2739 | 1.665350 | 0.003547 |
| ks2_c16 | 37,696 | 8.047 | 0.5642 | 1.648262 | 0.002093 |
| Original MLP (current repeat) | 73,728 | 7.293 | 1.0000 | 1.623065 | 0.002042 |
| spine_rank4_frozen | 70,656 | 7.712 | 1.0134 | 1.621002 | 0.000878 |
| combo_rank1_reference | 73,600 | 7.946 | 1.0876 | 1.620136 | 0.002536 |

The frontier is an observed mean frontier, not a claim that every neighboring point is statistically distinguishable. The full chart includes dominated candidates and the unchanged MLP. Pruned variants reuse the parent checkpoint loss after trained equivalence checks; they are implementation savings, not independent quality replications.

**The clearest targeted Hadamard gain is widening frozen R8 features while retaining the 2k side/history stream.** Increasing 512 to 2,048 features improves fresh loss from 1.684414 to 1.680042: −0.004373, with improvement in all three paired seeds. Learned size stays at 2,144 parameters per FFN before final-side pruning; fixed matrix storage and computation grow. The intermediate 1,024-feature result is 1.682497 and improves two of three seeds. These are complete width interventions, since widening also changes fixed codes and initial random geometry.

**Removing the unused final side output stacks an implementation saving onto the low-cost candidates.** The final layer's side residual never reaches the head. Removing those output rows and gains reduces the frozen candidate's average learned count from 2,144 to 1,632 per FFN. All four low-cost families passed checks on all three trained checkpoints. Their plotted times are measured separately. The architecture still has external normalization/time-mixing parameters; this is an FFN-module count, not a whole-model reduction.

Learning only the chart scales and shifts adds 128 parameters per FFN (2,272 total before pruning). Fresh loss is 1.682154, a −0.002260 change from the frozen parent with two of three seed wins, while its exploratory change was only −0.000160. It is a potentially useful cheap adaptation, but the small and variable gain needs replication. The fresh interval was not used to choose or train this intervention.

**Gains do not combine automatically.** ReLU² helps R8 spine geometry, but combining it with rank-1 does not add the improvements from the two separate interventions. Full R8 plus ReLU² improves fresh loss by only 0.001231 (two of three seeds); doubling its feature width to 1,024 improves another 0.010260 (three of three), but costs 107,104 learned parameters per FFN. That raw-loss improvement does not make it the best joint-cost option.

**Longer training improves all models without eliminating the Hadamard loss gap.** At 25k updates, fresh losses are 1.583371 for unchanged MLP, 1.626790 for original full Hadamard R8, 1.618287 for full F1024, and 1.643974 for frozen F512. These are matched 25k schedules, not evidence of convergence. Enlarging R1 to the original full-R8 learned budget reaches 1.626402, essentially the same quality as full R8, with substantially greater step duration.

## Research direction from this frontier

Prioritize combinations around the low-cost frozen Hadamard points: retain final-side pruning, test additional feature/side widths, and look for inexpensive adaptations that improve loss enough to justify their measured cost. The next replication should test whether the small warp-only benefit persists and whether it adds to the F2048 gain. These are proposed follow-ups, not experiments run after the two-hour cutoff.

Keep the compact learned-projection KS2 points in the same comparison: they provide the bridge toward MLP-level quality. Preserve the original MLP architecture and optimizer. Promote a candidate based on replicated frontier movement with matched training data, horizon, batch/context and timing methodology; do not promote a larger configuration solely because its loss is lower. Reserve new text for a subsequent confirmation so this final interval is not repeatedly used for selection.

The fixed-map implementation currently uses dense matrix multiplies. A more efficient implementation of the exact maps could move the Hadamard points left without changing their loss, but would need numerical and trained-checkpoint equivalence checks. These results do not isolate attention as the bottleneck: learned-projection models also differ in side/history width and trainable routing.
