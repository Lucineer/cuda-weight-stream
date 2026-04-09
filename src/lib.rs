//! Weight Streaming — DDR4 to BRAM tiled weight loading
//! Manages weight tile scheduling, prefetching, and buffer management
//! for inference silicon that can't hold full model in SRAM.

use std::collections::VecDeque;

/// A weight tile — a rectangular block of a weight matrix
#[derive(Debug, Clone)]
pub struct WeightTile {
    pub layer_id: usize,
    pub tile_row: usize,
    pub tile_col: usize,
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<i8>,  // INT8 quantized weights
    pub bram_address: u32,
    pub ddr4_offset: u64,
    pub loaded: bool,
    pub active: bool,
}

/// BRAM buffer slot
#[derive(Debug, Clone)]
pub struct BramSlot {
    pub index: usize,
    pub size_bytes: usize,
    pub tile: Option<WeightTile>,
    pub last_used_cycle: u64,
    pub locked: bool,
}

/// Memory bandwidth model
#[derive(Debug, Clone)]
pub struct BandwidthModel {
    pub ddr4_bandwidth_gbps: f64,
    pub bram_bandwidth_gbps: f64,
    pub ddr4_latency_ns: u64,
    pub bram_latency_ns: u64,
    pub bus_width_bits: u32,
}

impl Default for BandwidthModel {
    fn default() -> Self {
        Self {
            ddr4_bandwidth_gbps: 25.6,   // LPDDR4-3200 x2
            bram_bandwidth_gbps: 4096.0, // On-chip SRAM
            ddr4_latency_ns: 50,
            bram_latency_ns: 2,
            bus_width_bits: 128,
        }
    }
}

/// Tile dimensions configuration
#[derive(Debug, Clone)]
pub struct TileConfig {
    pub tile_rows: usize,
    pub tile_cols: usize,
    pub tile_size_bytes: usize,
    pub num_bram_slots: usize,
    pub bram_total_bytes: usize,
}

impl TileConfig {
    pub fn new(bram_total_kb: usize) -> Self {
        let tile_bytes = 4096; // 4KB tiles
        let tile_cols = 64;
        let tile_rows = tile_bytes / tile_cols; // 64x64 INT8 = 4KB
        Self {
            tile_rows, tile_cols, tile_size_bytes: tile_bytes,
            num_bram_slots: (bram_total_kb * 1024) / tile_bytes,
            bram_total_bytes: bram_total_kb * 1024,
        }
    }
}

/// The weight streamer — manages tile loading and prefetching
pub struct WeightStreamer {
    tiles: Vec<WeightTile>,
    bram_slots: Vec<BramSlot>,
    config: TileConfig,
    bandwidth: BandwidthModel,
    load_queue: VecDeque<usize>, // tile indices to load
    current_cycle: u64,
    tiles_loaded: u64,
    cache_hits: u64,
    cache_misses: u64,
    total_bytes_transferred: u64,
    stall_cycles: u64,
}

impl WeightStreamer {
    pub fn new(model_layers: &[(usize, usize)], bram_kb: usize) -> Self {
        let config = TileConfig::new(bram_kb);
        let mut tiles = vec![];
        let mut tile_idx = 0;
        let mut ddr4_offset = 0u64;
        
        for (layer_id, (rows, cols)) in model_layers.iter().enumerate() {
            let tiles_h = (rows + config.tile_rows - 1) / config.tile_rows;
            let tiles_w = (cols + config.tile_cols - 1) / config.tile_cols;
            
            for tr in 0..tiles_h {
                for tc in 0..tiles_w {
                    let actual_rows = config.tile_rows.min(rows - tr * config.tile_rows);
                    let actual_cols = config.tile_cols.min(cols - tc * config.tile_cols);
                    let data_size = actual_rows * actual_cols;
                    
                    tiles.push(WeightTile {
                        layer_id, tile_row: tr, tile_col: tc,
                        rows: actual_rows, cols: actual_cols,
                        data: vec![0i8; data_size],
                        bram_address: 0, ddr4_offset,
                        loaded: false, active: false,
                    });
                    ddr4_offset += data_size as u64;
                    tile_idx += 1;
                }
            }
        }
        
        // Initialize BRAM slots
        let bram_slots = (0..config.num_bram_slots).map(|i| BramSlot {
            index: i, size_bytes: config.tile_size_bytes,
            tile: None, last_used_cycle: 0, locked: false,
        }).collect();
        
        Self { tiles, bram_slots, config, bandwidth: BandwidthModel::default(),
            load_queue: VecDeque::new(), current_cycle: 0,
            tiles_loaded: 0, cache_hits: 0, cache_misses: 0,
            total_bytes_transferred: 0, stall_cycles: 0 }
    }

    /// Request a tile for computation — loads from DDR4 if not in BRAM
    pub fn request_tile(&mut self, layer_id: usize, tile_row: usize, tile_col: usize) -> Result<u32, String> {
        // Find tile
        let tile_idx = self.tiles.iter().position(|t|
            t.layer_id == layer_id && t.tile_row == tile_row && t.tile_col == tile_col
        ).ok_or("Tile not found")?;
        
        // Check if already in BRAM
        if let Some(slot_idx) = self.find_in_bram(tile_idx) {
            self.cache_hits += 1;
            self.bram_slots[slot_idx].last_used_cycle = self.current_cycle;
            self.tiles[tile_idx].active = true;
            return Ok(self.bram_slots[slot_idx].index as u32);
        }
        
        self.cache_misses += 1;
        
        // Find free BRAM slot
        let slot_idx = self.find_free_slot();
        
        // Calculate stall cycles for DDR4 load
        let tile = &self.tiles[tile_idx];
        let bytes = tile.rows * tile.cols;
        let bytes_per_cycle = (self.bandwidth.ddr4_bandwidth_gbps * 1e9 / 8.0) / 1e9; // GB/s at 1GHz
        let load_cycles = (bytes as f64 / bytes_per_cycle).ceil() as u64 + self.bandwidth.ddr4_latency_ns;
        self.stall_cycles += load_cycles;
        self.current_cycle += load_cycles;
        
        // Load tile
        self.tiles[tile_idx].bram_address = self.bram_slots[slot_idx].index as u32 * self.config.tile_size_bytes as u32;
        self.tiles[tile_idx].loaded = true;
        self.tiles[tile_idx].active = true;
        self.bram_slots[slot_idx].tile = Some(self.tiles[tile_idx].clone());
        self.bram_slots[slot_idx].last_used_cycle = self.current_cycle;
        self.tiles_loaded += 1;
        self.total_bytes_transferred += bytes as u64;
        
        Ok(self.bram_slots[slot_idx].index as u32)
    }

    /// Mark a tile as no longer active
    pub fn release_tile(&mut self, layer_id: usize, tile_row: usize, tile_col: usize) {
        if let Some(tile) = self.tiles.iter_mut().find(|t|
            t.layer_id == layer_id && t.tile_row == tile_row && t.tile_col == tile_col
        ) { tile.active = false; }
    }

    /// Prefetch tiles for upcoming layers
    pub fn prefetch(&mut self, layer_id: usize, max_tiles: usize) -> usize {
        let layer_tiles: Vec<usize> = self.tiles.iter().enumerate()
            .filter(|(_, t)| t.layer_id == layer_id && !t.loaded)
            .map(|(i, _)| i)
            .take(max_tiles)
            .collect();
        
        for &idx in &layer_tiles {
            if let Some(free) = self.find_free_slot_idx() {
                self.tiles[idx].loaded = true;
                self.tiles[idx].bram_address = free as u32 * self.config.tile_size_bytes as u32;
                self.bram_slots[free].tile = Some(self.tiles[idx].clone());
                self.bram_slots[free].last_used_cycle = self.current_cycle;
                self.tiles_loaded += 1;
                self.total_bytes_transferred += self.config.tile_size_bytes as u64;
            }
        }
        layer_tiles.len()
    }

    /// Evict least recently used tiles
    pub fn evict_lru(&mut self, count: usize) -> usize {
        let mut evicted = 0;
        let mut slots: Vec<usize> = self.bram_slots.iter().enumerate()
            .filter(|(_, s)| s.tile.is_some() && !s.locked)
            .map(|(i, _)| i).collect();
        
        // Sort by LRU
        slots.sort_by_key(|&i| self.bram_slots[i].last_used_cycle);
        
        for &idx in slots.iter().take(count) {
            if let Some(ref tile) = self.bram_slots[idx].tile {
                if let Some(t) = self.tiles.iter_mut().find(|t| t.bram_address == self.bram_slots[idx].index as u32 * self.config.tile_size_bytes as u32) {
                    t.loaded = false;
                }
            }
            self.bram_slots[idx].tile = None;
            evicted += 1;
        }
        evicted
    }

    /// Get streaming statistics
    pub fn stats(&self) -> StreamStats {
        StreamStats {
            total_tiles: self.tiles.len(),
            tiles_loaded: self.tiles_loaded,
            cache_hits: self.cache_hits,
            cache_misses: self.cache_misses,
            hit_rate: if self.cache_hits + self.cache_misses > 0 {
                self.cache_hits as f64 / (self.cache_hits + self.cache_misses) as f64
            } else { 0.0 },
            bram_utilization: self.bram_slots.iter().filter(|s| s.tile.is_some()).count(),
            bram_total: self.config.num_bram_slots,
            total_bytes: self.total_bytes_transferred,
            stall_cycles: self.stall_cycles,
            estimated_time_us: self.stall_cycles, // at 1GHz
        }
    }

    fn find_in_bram(&self, tile_idx: usize) -> Option<usize> {
        let tile = &self.tiles[tile_idx];
        self.bram_slots.iter().position(|s|
            s.tile.as_ref().map_or(false, |t|
                t.layer_id == tile.layer_id && t.tile_row == tile.tile_row && t.tile_col == tile.tile_col
            )
        )
    }

    fn find_free_slot(&mut self) -> usize {
        if let Some(i) = self.find_free_slot_idx() { return i; }
        // Evict LRU
        self.evict_lru(1);
        self.find_free_slot_idx().unwrap_or(0)
    }

    fn find_free_slot_idx(&self) -> Option<usize> {
        self.bram_slots.iter().position(|s| s.tile.is_none())
    }
}

#[derive(Debug, Clone)]
pub struct StreamStats {
    pub total_tiles: usize,
    pub tiles_loaded: usize,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub hit_rate: f64,
    pub bram_utilization: usize,
    pub bram_total: usize,
    pub total_bytes: u64,
    pub stall_cycles: u64,
    pub estimated_time_us: u64,
}

/// Prefetch planner — determines optimal tile loading order
pub struct PrefetchPlanner {
    lookahead_layers: usize,
}

impl PrefetchPlanner {
    pub fn new(lookahead: usize) -> Self { Self { lookahead_layers: lookahead } }

    /// Generate optimal prefetch schedule for a layer sequence
    pub fn plan(&self, layer_sequence: &[usize], streamer: &WeightStreamer) -> Vec<(usize, usize, usize)> {
        let mut schedule = vec![];
        let mut seen_layers = vec![];
        
        for &layer in layer_sequence {
            if !seen_layers.contains(&layer) {
                seen_layers.push(layer);
            }
        }
        
        // For each layer, schedule tile loads
        for &layer in &seen_layers {
            let layer_tiles: Vec<(usize, usize)> = streamer.tiles.iter()
                .filter(|t| t.layer_id == layer)
                .map(|t| (t.tile_row, t.tile_col))
                .collect();
            
            for (tr, tc) in layer_tiles {
                schedule.push((layer, tr, tc));
            }
        }
        
        schedule
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_model() -> Vec<(usize, usize)> {
        vec![(128, 64), (64, 32), (32, 16)] // 3 layers
    }

    #[test]
    fn test_streamer_creation() {
        let streamer = WeightStreamer::new(&small_model(), 64);
        assert!(streamer.tiles.len() > 0);
        let stats = streamer.stats();
        assert_eq!(stats.total_tiles, streamer.tiles.len());
    }

    #[test]
    fn test_request_tile() {
        let mut streamer = WeightStreamer::new(&small_model(), 64);
        let addr = streamer.request_tile(0, 0, 0).unwrap();
        assert!(addr > 0);
        assert_eq!(streamer.stats().cache_misses, 1);
    }

    #[test]
    fn test_cache_hit() {
        let mut streamer = WeightStreamer::new(&small_model(), 64);
        streamer.request_tile(0, 0, 0).unwrap();
        streamer.request_tile(0, 0, 0).unwrap(); // should hit
        assert_eq!(streamer.stats().cache_hits, 1);
    }

    #[test]
    fn test_bram_exhaustion() {
        let streamer = WeightStreamer::new(&[(256, 256)], 8); // 8KB BRAM, big model
        let mut streamer = streamer;
        // Load more tiles than BRAM can hold
        for i in 0..5 {
            streamer.request_tile(0, 0, i).ok();
        }
        let stats = streamer.stats();
        assert!(stats.bram_utilization > 0);
    }

    #[test]
    fn test_release_tile() {
        let mut streamer = WeightStreamer::new(&small_model(), 64);
        streamer.request_tile(0, 0, 0).unwrap();
        streamer.release_tile(0, 0, 0);
        let tile = streamer.tiles.iter().find(|t| t.layer_id == 0).unwrap();
        assert!(!tile.active);
    }

    #[test]
    fn test_prefetch() {
        let mut streamer = WeightStreamer::new(&small_model(), 64);
        let count = streamer.prefetch(1, 10);
        assert!(count > 0);
    }

    #[test]
    fn test_lru_eviction() {
        let mut streamer = WeightStreamer::new(&[(64, 64)], 8); // Small BRAM
        for i in 0..4 { streamer.request_tile(0, i, 0).ok(); }
        let before = streamer.stats().bram_utilization;
        let evicted = streamer.evict_lru(2);
        assert_eq!(evicted, 2);
    }

    #[test]
    fn test_stats() {
        let mut streamer = WeightStreamer::new(&small_model(), 64);
        streamer.request_tile(0, 0, 0).unwrap();
        streamer.request_tile(0, 0, 0).unwrap();
        let stats = streamer.stats();
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_misses, 1);
        assert!((stats.hit_rate - 0.5).abs() < 0.01);
    }
}
