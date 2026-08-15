# Milestone 10 Results

These are descriptive two-seed observations from the GTX 1650 SUPER experiment suite. They do not establish statistical significance, general performance rankings, or causal mechanisms.

| Study | Condition | Validation loss (mean [range]) | Approx. bits per byte (mean [range]) | Parameters | Mean training time |
| --- | --- | --- | --- | ---: | ---: |
| Attention heads | 2 heads | 2.1566 [2.1317, 2.1815] | 2.3387 [2.3117, 2.3657] | 1,927,296 | 79.8s |
| Attention heads | 4 heads | 2.1707 [2.1488, 2.1925] | 2.3539 [2.3302, 2.3776] | 1,927,296 | 87.2s |
| Attention heads | 8 heads | 2.2034 [2.1843, 2.2225] | 2.3895 [2.3687, 2.4102] | 1,927,296 | 98.8s |
| Position | Learned | 2.1707 [2.1488, 2.1925] | 2.3539 [2.3302, 2.3776] | 1,927,296 | 94.8s |
| Position | Sinusoidal | 2.4206 [2.4191, 2.4221] | 2.6250 [2.6233, 2.6266] | 1,902,720 | 92.9s |
| Tokenization | Byte-BPE (maximum vocab 320) | 2.1707 [2.1488, 2.1925] | 2.3539 [2.3302, 2.3776] | 1,927,296 | 94.8s |
| Tokenization | Character | 1.6955 [1.6931, 1.6978] | 2.4460 [2.4426, 2.4494] | 1,829,376 | 85.7s |

For the tokenizer study, compare approximate bits per byte rather than token-level loss: the units differ. This value is normalized from the deterministic sampled validation loss, not an exhaustive validation-corpus likelihood. On this fixed corpus, split, update budget, and two selected seeds, the byte-BPE condition had lower observed approximate bits per byte than the character condition. The 2/4/8-head conditions retain the same trainable parameter count; their observed differences are limited to this configuration and budget.

The reproducible JSON records, fixed-prompt samples, full configurations, checkpoint references, and per-evaluation loss histories live in `checkpoints/experiments/milestone-10/` after running the suite.
