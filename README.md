# Torchlet

Torchlet is a lightweight framework inspired by [micrograd](https://github.com/karpathy/micrograd), designed to be both educational and practical.

## Performance Benchmarks

We track performance improvements through the following benchmarks:

| Benchmark                | Initial Implementation (µs) | Current Implementation (µs) | Notes                            |
|--------------------------|----------------------------|-----------------------------|----------------------------------|
| Addition (1K ops)         | 5.127                      | 5.127                       | Baseline benchmark               |
| Multiplication (1K ops)   | 4.892                      | 4.892                       | Baseline benchmark               |
| Backward Pass             | 154.600                    | 154.600                     | Baseline benchmark               |
| MLP Forward Pass (100 runs)| 413.949                   | 413.949                     | Baseline benchmark               |
| MLP Backward Pass         | 884.800                    | 884.800                     | Baseline benchmark               |
| Zero Grad (100 runs)      | 5.948                      | 5.948                       | Baseline benchmark               |

Attention! The benchmarks may and most likely will be different on your machine. Those are mainly used to track the performance improvements over time in my implementation.

### Summary of Changes

- **[12/08/2024]**: Optimized `Element` operations by introducing lazy gradient initialization. This is the basic benchmark for all other operations as the performance was not measured earlier.
