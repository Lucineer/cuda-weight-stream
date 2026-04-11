# cuda-weight-stream

DDR4 to BRAM weight streaming — tiled weight loading for inference silicon with prefetch and LRU eviction

Part of the Cocapn chip layer — hardware design automation and silicon engineering.

## What It Does

### Key Types

- `WeightTile` — core data structure
- `BramSlot` — core data structure
- `BandwidthModel` — core data structure
- `TileConfig` — core data structure
- `WeightStreamer` — core data structure
- `StreamStats` — core data structure
- _and 1 more (see source)_

## Quick Start

```bash
# Clone
git clone https://github.com/Lucineer/cuda-weight-stream.git
cd cuda-weight-stream

# Build
cargo build

# Run tests
cargo test
```

## Usage

```rust
use cuda_weight_stream::*;

// See src/lib.rs for full API
// 8 unit tests included
```

### Available Implementations

- `Default for BandwidthModel` — see source for methods
- `TileConfig` — see source for methods
- `WeightStreamer` — see source for methods
- `PrefetchPlanner` — see source for methods

## Testing

```bash
cargo test
```

8 unit tests covering core functionality.

## Architecture

This crate is part of the **Cocapn Fleet** — a git-native multi-agent ecosystem.

- **Category**: chip
- **Language**: Rust
- **Dependencies**: See `Cargo.toml`
- **Status**: Active development

## Related Crates

- [cuda-thermal-sim](https://github.com/Lucineer/cuda-thermal-sim)
- [cuda-signal-integrity](https://github.com/Lucineer/cuda-signal-integrity)
- [cuda-floorplanner](https://github.com/Lucineer/cuda-floorplanner)
- [cuda-power-estimator](https://github.com/Lucineer/cuda-power-estimator)
- [cuda-clock-tree](https://github.com/Lucineer/cuda-clock-tree)
- [cuda-ir-drop](https://github.com/Lucineer/cuda-ir-drop)
- [cuda-electromigration](https://github.com/Lucineer/cuda-electromigration)
- [cuda-latchup](https://github.com/Lucineer/cuda-latchup)
- [cuda-esd](https://github.com/Lucineer/cuda-esd)
- [cuda-drc](https://github.com/Lucineer/cuda-drc)
- [cuda-pcie](https://github.com/Lucineer/cuda-pcie)
- [cuda-noc](https://github.com/Lucineer/cuda-noc)
- [cuda-packet-buffer](https://github.com/Lucineer/cuda-packet-buffer)
- [cuda-fpga-toolkit](https://github.com/Lucineer/cuda-fpga-toolkit)
- [cuda-synth](https://github.com/Lucineer/cuda-synth)
- [cuda-verilog](https://github.com/Lucineer/cuda-verilog)
- [cuda-weight-compiler](https://github.com/Lucineer/cuda-weight-compiler)
- [cuda-frozen-intelligence](https://github.com/Lucineer/cuda-frozen-intelligence)

## Fleet Position

```
Casey (Captain)
├── JetsonClaw1 (Lucineer realm — hardware, low-level systems, fleet infrastructure)
├── Oracle1 (SuperInstance — lighthouse, architecture, consensus)
└── Babel (SuperInstance — multilingual scout)
```

## Contributing

This is a fleet vessel component. Fork it, improve it, push a bottle to `message-in-a-bottle/for-jetsonclaw1/`.

## License

MIT

---

*Built by JetsonClaw1 — part of the Cocapn fleet*
*See [cocapn-fleet-readme](https://github.com/Lucineer/cocapn-fleet-readme) for the full fleet roadmap*
