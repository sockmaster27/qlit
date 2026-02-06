mod circuit;
mod simulate;
#[cfg(feature = "gpu")]
mod simulate_gpu;
mod tableau;
mod utils;

pub use circuit::{CircuitCreationError, CliffordTCircuit, CliffordTGate};
pub use simulate::simulate_circuit;

#[cfg(feature = "gpu")]
pub use simulate::simulate_circuit_gpu;

/// Initialization all global resources.
///
/// This will happen automatically the first time it's needed,
/// but this can be called to pre-empt that work at a more appropriate time.
pub fn initialize_global() {
    #[cfg(feature = "gpu")]
    simulate_gpu::initialize_gpu();
}
