use std::sync::OnceLock;

use wgpu::util::DeviceExt;

use crate::clifford_circuit::CliffordGate;

const BLOCK_SIZE: u32 = 32;
const WORKGROUP_SIZE: u32 = 64;
const U32_SIZE: u64 = size_of::<u32>() as u64;

pub enum BasisStateProbability {
    /// This basis state is the only one that is possible to sample.
    One,
    /// This basis state is one of several possible states that can be sampled.
    InBetween,
    /// This basis state is not possible to sample.
    Zero,
}

/// Initialize the global GPU context.
///
/// This will happen automatically the first time it's needed,
/// but this can be called to preempt that work at a more appropriate time.
pub fn initialize_gpu() {
    get_gpu();
}
fn get_gpu() -> &'static GpuContext {
    GPU_CONTEXT.get_or_init(|| pollster::block_on(GpuContext::new()))
}

/// The global GPU context.
/// Includes the initialized device, compiled shaders, etc.
static GPU_CONTEXT: OnceLock<GpuContext> = OnceLock::new();
pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    tableau_bind_group_layout: wgpu::BindGroupLayout,
    apply_gates_bind_group_layout: wgpu::BindGroupLayout,
    apply_gates_pipeline: wgpu::ComputePipeline,
    collapse_bind_group_layout: wgpu::BindGroupLayout,
    collapse_pipeline: wgpu::ComputePipeline,
    detect_impossible_bind_group_layout: wgpu::BindGroupLayout,
    detect_impossible_pipeline: wgpu::ComputePipeline,
}
impl GpuContext {
    pub async fn new() -> GpuContext {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .unwrap();

        let tableau_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Tableau"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // "Apply Gates" pipeline setup
        let apply_gates_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Apply Gates"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let apply_gates_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Apply Gates"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_generator/apply_gates.wgsl").into()),
        });
        let apply_gates_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Apply Gates"),
                bind_group_layouts: &[&tableau_bind_group_layout, &apply_gates_bind_group_layout],
                push_constant_ranges: &[],
            });
        let apply_gates_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Apply Gates"),
                layout: Some(&apply_gates_pipeline_layout),
                module: &apply_gates_module,
                entry_point: Some("main"),
                cache: None,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

        // "Collapse" pipeline setup
        let collapse_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Collapse"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let collapse_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Collapse"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_generator/collapse.wgsl").into()),
        });
        let collapse_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Collapse"),
                bind_group_layouts: &[&tableau_bind_group_layout, &collapse_bind_group_layout],
                push_constant_ranges: &[],
            });
        let collapse_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Collapse"),
            layout: Some(&collapse_pipeline_layout),
            module: &collapse_module,
            entry_point: Some("main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        // "Detect Impossible" pipeline setup
        let detect_impossible_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Detect Impossible"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let detect_impossible_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Detect Impossible"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("gpu_generator/detect_impossible.wgsl").into(),
            ),
        });
        let detect_impossible_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Detect Impossible"),
                bind_group_layouts: &[
                    &tableau_bind_group_layout,
                    &detect_impossible_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
        let detect_impossible_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Detect Impossible"),
                layout: Some(&detect_impossible_pipeline_layout),
                module: &detect_impossible_module,
                entry_point: Some("main"),
                cache: None,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

        GpuContext {
            device,
            queue,
            tableau_bind_group_layout,
            apply_gates_bind_group_layout,
            apply_gates_pipeline,
            collapse_bind_group_layout,
            collapse_pipeline,
            detect_impossible_bind_group_layout,
            detect_impossible_pipeline,
        }
    }
}

pub struct GpuGenerator {
    n: u32,
    tableau_bind_group: wgpu::BindGroup,
}
impl GpuGenerator {
    pub fn zero(n: u32) -> GpuGenerator {
        let gpu = get_gpu();

        let n_buf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("n"),
                contents: &n.to_ne_bytes(),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let mut tableau_contents: Vec<u8> = vec![0; tableau_byte_length(n).try_into().unwrap()];
        // Initialize stabilizers
        for i in 0..n {
            let byte_index: usize = z_column_byte_index(n, i / 8, i).try_into().unwrap();
            tableau_contents[byte_index] = byte_bitmask(i % 8);
        }
        // Initialize destabilizers
        for i in 0..n {
            let byte_index: usize = x_column_byte_index(n, half_column_byte_length(n) + (i / 8), i)
                .try_into()
                .unwrap();
            tableau_contents[byte_index] = byte_bitmask(i % 8);
        }
        let tableau_buf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tableau"),
                contents: &tableau_contents,
                usage: wgpu::BufferUsages::STORAGE,
            });
        let tableau_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tableau"),
            layout: &gpu.tableau_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: n_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: tableau_buf.as_entire_binding(),
                },
            ],
        });

        GpuGenerator {
            n,
            tableau_bind_group,
        }
    }

    pub fn apply_gates(&mut self, gates: &[CliffordGate]) {
        if gates.is_empty() {
            // Creating a an empty buffer is not allowed
            return;
        }

        let gpu = get_gpu();

        // If the number of gates exceeds the maximum buffer size,
        // we need to split the gates into multiple buffers and do multiple passes.
        let mut gate_contents = Vec::new();
        let mut gate_contents_pass: Vec<u8> = Vec::new();
        for &gate in gates {
            let added_size = match gate {
                CliffordGate::S(_) => 4 * 2,
                CliffordGate::H(_) => 4 * 2,
                CliffordGate::Cnot(_, _) => 4 * 3,
            };
            let max_buffer_size: usize = gpu
                .device
                .limits()
                .max_storage_buffer_binding_size
                .try_into()
                .unwrap();
            if gate_contents_pass.len() + added_size > max_buffer_size {
                gate_contents.push(gate_contents_pass);
                gate_contents_pass = Vec::new();
            }
            match gate {
                CliffordGate::S(a) => {
                    gate_contents_pass
                        .extend([0u32.to_ne_bytes(), (a as u32).to_ne_bytes()].as_flattened());
                }
                CliffordGate::H(a) => {
                    gate_contents_pass
                        .extend([1u32.to_ne_bytes(), (a as u32).to_ne_bytes()].as_flattened());
                }
                CliffordGate::Cnot(a, b) => {
                    gate_contents_pass.extend(
                        [
                            2u32.to_ne_bytes(),
                            (a as u32).to_ne_bytes(),
                            (b as u32).to_ne_bytes(),
                        ]
                        .as_flattened(),
                    );
                }
            }
        }
        gate_contents.push(gate_contents_pass);

        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        for gate_contents_pass in gate_contents {
            let gates_buf = gpu
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Gates"),
                    contents: &gate_contents_pass,
                    usage: wgpu::BufferUsages::STORAGE,
                });
            let apply_gates_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Gates"),
                layout: &gpu.apply_gates_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: gates_buf.as_entire_binding(),
                }],
            });

            let mut compute_pass = encoder.begin_compute_pass(&Default::default());
            compute_pass.set_pipeline(&gpu.apply_gates_pipeline);
            compute_pass.set_bind_group(0, &self.tableau_bind_group, &[]);
            compute_pass.set_bind_group(1, &apply_gates_bind_group, &[]);
            compute_pass.dispatch_workgroups(
                column_block_length(self.n).div_ceil(WORKGROUP_SIZE),
                1,
                1,
            );
        }
        gpu.queue.submit(Some(encoder.finish()));
    }

    pub fn probability(self, w: &[bool]) -> BasisStateProbability {
        let n = self.n;
        let gpu = get_gpu();

        let nondeterministic_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nondeterministic"),
            size: U32_SIZE,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let nondeterministic_download_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Download nondeterministic"),
            size: U32_SIZE,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let impossible_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("impossible"),
            size: U32_SIZE,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let impossible_download_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Download impossible"),
            size: U32_SIZE,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = gpu.device.create_command_encoder(&Default::default());

        // Collapse all qubits
        for a in 0..n {
            let a_buf = gpu
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("a"),
                    contents: &a.to_ne_bytes(),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let w_buf = gpu
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("w"),
                    contents: &(if w[a as usize] { 1u32 } else { 0u32 }).to_ne_bytes(),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let collapse_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("Collapse {a}")),
                layout: &gpu.collapse_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: a_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: w_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: nondeterministic_buf.as_entire_binding(),
                    },
                ],
            });
            let mut compute_pass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            compute_pass.set_pipeline(&gpu.collapse_pipeline);
            compute_pass.set_bind_group(0, &self.tableau_bind_group, &[]);
            compute_pass.set_bind_group(1, &collapse_bind_group, &[]);
            compute_pass.dispatch_workgroups(
                column_block_length((n + 1) - a).div_ceil(WORKGROUP_SIZE),
                1,
                1,
            );
        }

        // Detect if any measurements were impossible
        {
            let w_contents: Vec<u8> = w
                .iter()
                .map(|&x| if x { 1u32 } else { 0u32 })
                .flat_map(|x| x.to_ne_bytes())
                .collect();
            let w_buf = gpu
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("w"),
                    contents: &w_contents,
                    usage: wgpu::BufferUsages::STORAGE,
                });
            let detect_impossible_bind_group =
                gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Detect Impossible"),
                    layout: &gpu.detect_impossible_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: w_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: impossible_buf.as_entire_binding(),
                        },
                    ],
                });
            let mut compute_pass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            compute_pass.set_pipeline(&gpu.detect_impossible_pipeline);
            compute_pass.set_bind_group(0, &self.tableau_bind_group, &[]);
            compute_pass.set_bind_group(1, &detect_impossible_bind_group, &[]);
            compute_pass.dispatch_workgroups(n.div_ceil(WORKGROUP_SIZE), 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &nondeterministic_buf,
            0,
            &nondeterministic_download_buf,
            0,
            U32_SIZE,
        );
        encoder.copy_buffer_to_buffer(&impossible_buf, 0, &impossible_download_buf, 0, U32_SIZE);
        gpu.queue.submit([encoder.finish()]);
        let nondeterministic_buf_slice = nondeterministic_download_buf.slice(..);
        nondeterministic_buf_slice.map_async(wgpu::MapMode::Read, |_| {});
        let impossible_buf_slice = impossible_download_buf.slice(..);
        impossible_buf_slice.map_async(wgpu::MapMode::Read, |_| {
            // TODO: Sometimes we need to wait for this and can't trust device.poll()?
            // https://github.com/gfx-rs/wgpu/blob/c7c79a0dc9356081a884b5518d1c08ce7a09c7c5/examples/standalone/01_hello_compute/src/main.rs#L231-L244
        });
        gpu.device.poll(wgpu::Maintain::Wait);
        let nondeterministic_data = nondeterministic_buf_slice.get_mapped_range();
        let nondeterministic = u32::from_ne_bytes([
            nondeterministic_data[0],
            nondeterministic_data[1],
            nondeterministic_data[2],
            nondeterministic_data[3],
        ]);
        let impossible_data = impossible_buf_slice.get_mapped_range();
        let impossible = u32::from_ne_bytes([
            impossible_data[0],
            impossible_data[1],
            impossible_data[2],
            impossible_data[3],
        ]);

        if impossible == 1 {
            BasisStateProbability::Zero
        } else if nondeterministic == 1 {
            BasisStateProbability::InBetween
        } else {
            BasisStateProbability::One
        }
    }
}

/// Get the bitmask for the i'th bit in a byte, e.g.
///
/// ```text
/// bitmask(0) -> 10000000
/// bitmask(1) -> 01000000
/// bitmask(6) -> 00000010
/// ```
///
/// # Panics
/// If `i` is greater than or equal to 8 in debug mode.
fn byte_bitmask(i: u32) -> u8 {
    debug_assert!(i < 8);
    1 << (7 - i)
}

/// Get the index of the i'th byte of the column representing the x part of the `q`th tensor element.
/// The first half of the bytes will contain the stabilizer parts and the second half the destabilizer parts.
fn x_column_byte_index(n: u32, i: u32, q: u32) -> u32 {
    debug_assert!(i < column_byte_length(n));
    debug_assert!(q < n);
    2 * q * column_byte_length(n) + i
}
/// Get the index of the i'th byte of the column representing the z part of the `q`th tensor element.
/// The first half of the bytes will contain the stabilizer parts and the second half the destabilizer parts.
fn z_column_byte_index(n: u32, i: u32, q: u32) -> u32 {
    debug_assert!(i < column_byte_length(n));
    debug_assert!(q < n);
    (2 * q + 1) * column_byte_length(n) + i
}

/// Get the length of each half of the tableau column in blocks.
fn half_column_block_length(n: u32) -> u32 {
    // The stabilizer and destabilizer parts of each
    // column takes up a whole number of bit blocks.
    n.div_ceil(BLOCK_SIZE)
}
/// Get the length of each tableau column in blocks.
fn column_block_length(n: u32) -> u32 {
    2 * half_column_block_length(n)
}
/// Get the length of the tableau in blocks.
fn tableau_block_length(n: u32) -> u32 {
    column_block_length(n) * (n + n + 1)
}
/// Get the length of each half of the tableau column in bytes.
fn half_column_byte_length(n: u32) -> u32 {
    blocks_to_bytes(half_column_block_length(n))
}
/// Get the length of each tableau column in bytes.
fn column_byte_length(n: u32) -> u32 {
    blocks_to_bytes(column_block_length(n))
}
/// Get the length of the tableau in bytes.
fn tableau_byte_length(n: u32) -> u32 {
    blocks_to_bytes(tableau_block_length(n))
}

/// Convert some number of blocks to the corresponding number of bytes.
fn blocks_to_bytes(n: u32) -> u32 {
    n * (BLOCK_SIZE / 8)
}
