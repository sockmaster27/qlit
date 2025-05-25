use std::sync::OnceLock;

use num_complex::Complex;
use wgpu::util::DeviceExt;

const BLOCK_SIZE: u32 = 32;
const WORKGROUP_SIZE: u32 = 64;
const U32_SIZE: u64 = size_of::<u32>() as u64;

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

    apply_cnot_gate_bind_group_layout: wgpu::BindGroupLayout,
    apply_cnot_gate_pipeline: wgpu::ComputePipeline,

    apply_h_gate_bind_group_layout: wgpu::BindGroupLayout,
    apply_h_gate_pipeline: wgpu::ComputePipeline,

    apply_s_gate_bind_group_layout: wgpu::BindGroupLayout,
    apply_s_gate_pipeline: wgpu::ComputePipeline,

    apply_z_gate_bind_group_layout: wgpu::BindGroupLayout,
    apply_z_gate_pipeline: wgpu::ComputePipeline,

    elimination_pass_bind_group_layout: wgpu::BindGroupLayout,
    elimination_pass_pipeline: wgpu::ComputePipeline,

    swap_rows_bind_group_layout: wgpu::BindGroupLayout,
    swap_rows_pipeline: wgpu::ComputePipeline,
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

        // Gate application pipeline setups
        // - Cnot
        let apply_cnot_gate_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Apply Cnot Gate"),
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
                ],
            });
        let apply_cnot_gate_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Apply Cnot Gate"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("gpu_generator/apply_cnot_gate.wgsl").into(),
            ),
        });
        let apply_cnot_gate_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Apply Cnot Gate"),
                bind_group_layouts: &[
                    &tableau_bind_group_layout,
                    &apply_cnot_gate_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
        let apply_cnot_gate_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Apply Cnot Gate"),
                layout: Some(&apply_cnot_gate_pipeline_layout),
                module: &apply_cnot_gate_module,
                entry_point: Some("main"),
                cache: None,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });
        // - H
        let apply_h_gate_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Apply H Gate"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let apply_h_gate_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Apply H Gate"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("gpu_generator/apply_h_gate.wgsl").into(),
            ),
        });
        let apply_h_gate_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Apply H Gate"),
                bind_group_layouts: &[&tableau_bind_group_layout, &apply_h_gate_bind_group_layout],
                push_constant_ranges: &[],
            });
        let apply_h_gate_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Apply H Gate"),
                layout: Some(&apply_h_gate_pipeline_layout),
                module: &apply_h_gate_module,
                entry_point: Some("main"),
                cache: None,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });
        // - S
        let apply_s_gate_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Apply S Gate"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let apply_s_gate_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Apply S Gate"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("gpu_generator/apply_s_gate.wgsl").into(),
            ),
        });
        let apply_s_gate_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Apply S Gate"),
                bind_group_layouts: &[&tableau_bind_group_layout, &apply_s_gate_bind_group_layout],
                push_constant_ranges: &[],
            });
        let apply_s_gate_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Apply S Gate"),
                layout: Some(&apply_s_gate_pipeline_layout),
                module: &apply_s_gate_module,
                entry_point: Some("main"),
                cache: None,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });
        // - Z
        let apply_z_gate_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Apply Z Gate"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let apply_z_gate_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Apply Z Gate"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("gpu_generator/apply_z_gate.wgsl").into(),
            ),
        });
        let apply_z_gate_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Apply Z Gate"),
                bind_group_layouts: &[&tableau_bind_group_layout, &apply_z_gate_bind_group_layout],
                push_constant_ranges: &[],
            });
        let apply_z_gate_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Apply Z Gate"),
                layout: Some(&apply_z_gate_pipeline_layout),
                module: &apply_z_gate_module,
                entry_point: Some("main"),
                cache: None,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

        // Elimination pass pipeline setups
        let elimination_pass_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Elimination Pass"),
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
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
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
        let elimination_pass_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Elimination Pass"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("gpu_generator/elimination_pass.wgsl").into(),
            ),
        });
        let elimination_pass_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Elimination Pass"),
                bind_group_layouts: &[
                    &tableau_bind_group_layout,
                    &elimination_pass_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
        let elimination_pass_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Elimination Pass"),
                layout: Some(&elimination_pass_pipeline_layout),
                module: &elimination_pass_module,
                entry_point: Some("main"),
                cache: None,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

        // Swap rows pipeline setups
        let swap_rows_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Swap Rows"),
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
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let swap_rows_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Swap Rows"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_generator/swap_rows.wgsl").into()),
        });
        let swap_rows_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Swap Rows"),
                bind_group_layouts: &[&tableau_bind_group_layout, &swap_rows_bind_group_layout],
                push_constant_ranges: &[],
            });
        let swap_rows_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Swap Rows"),
            layout: Some(&swap_rows_pipeline_layout),
            module: &swap_rows_module,
            entry_point: Some("main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        GpuContext {
            device,
            queue,
            tableau_bind_group_layout,

            apply_cnot_gate_bind_group_layout,
            apply_cnot_gate_pipeline,

            apply_h_gate_bind_group_layout,
            apply_h_gate_pipeline,

            apply_s_gate_bind_group_layout,
            apply_s_gate_pipeline,

            apply_z_gate_bind_group_layout,
            apply_z_gate_pipeline,

            elimination_pass_bind_group_layout,
            elimination_pass_pipeline,

            swap_rows_bind_group_layout,
            swap_rows_pipeline,
        }
    }
}

pub struct GpuGenerator {
    n: u32,
    tableau_bind_group: wgpu::BindGroup,
}
impl GpuGenerator {
    pub fn zero(n: usize) -> GpuGenerator {
        let gpu = get_gpu();

        let n: u32 = n.try_into().expect("n does not fit into u32");

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

    pub fn apply_cnot_gate(&mut self, a: usize, b: usize) {
        let gpu = get_gpu();

        let a: u32 = a.try_into().expect("a does not fit into u32");
        let b: u32 = b.try_into().expect("b does not fit into u32");

        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        let a_buf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("a"),
                contents: &a.to_ne_bytes(),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let b_buf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("b"),
                contents: &b.to_ne_bytes(),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let apply_cnot_gate_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Cnot Bind Group"),
            layout: &gpu.apply_cnot_gate_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_buf.as_entire_binding(),
                },
            ],
        });

        let mut compute_pass = encoder.begin_compute_pass(&Default::default());
        compute_pass.set_pipeline(&gpu.apply_cnot_gate_pipeline);
        compute_pass.set_bind_group(0, &self.tableau_bind_group, &[]);
        compute_pass.set_bind_group(1, &apply_cnot_gate_bind_group, &[]);
        compute_pass.dispatch_workgroups(
            column_block_length(self.n).div_ceil(WORKGROUP_SIZE),
            1,
            1,
        );
        drop(compute_pass);

        gpu.queue.submit([encoder.finish()]);
    }
    pub fn apply_h_gate(&mut self, a: usize) {
        let gpu = get_gpu();

        let a: u32 = a.try_into().expect("a does not fit into u32");

        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        let a_buf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("a"),
                contents: &a.to_ne_bytes(),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let apply_h_gate_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("H Bind Group"),
            layout: &gpu.apply_h_gate_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: a_buf.as_entire_binding(),
            }],
        });

        let mut compute_pass = encoder.begin_compute_pass(&Default::default());
        compute_pass.set_pipeline(&gpu.apply_h_gate_pipeline);
        compute_pass.set_bind_group(0, &self.tableau_bind_group, &[]);
        compute_pass.set_bind_group(1, &apply_h_gate_bind_group, &[]);
        compute_pass.dispatch_workgroups(
            column_block_length(self.n).div_ceil(WORKGROUP_SIZE),
            1,
            1,
        );
        drop(compute_pass);

        gpu.queue.submit([encoder.finish()]);
    }
    pub fn apply_s_gate(&mut self, a: usize) {
        let gpu = get_gpu();

        let a: u32 = a.try_into().expect("a does not fit into u32");

        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        let a_buf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("a"),
                contents: &a.to_ne_bytes(),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let apply_s_gate_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("S Bind Group"),
            layout: &gpu.apply_s_gate_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: a_buf.as_entire_binding(),
            }],
        });

        let mut compute_pass = encoder.begin_compute_pass(&Default::default());
        compute_pass.set_pipeline(&gpu.apply_s_gate_pipeline);
        compute_pass.set_bind_group(0, &self.tableau_bind_group, &[]);
        compute_pass.set_bind_group(1, &apply_s_gate_bind_group, &[]);
        compute_pass.dispatch_workgroups(
            column_block_length(self.n).div_ceil(WORKGROUP_SIZE),
            1,
            1,
        );
        drop(compute_pass);

        gpu.queue.submit([encoder.finish()]);
    }
    pub fn apply_z_gate(&mut self, a: usize) {
        let gpu = get_gpu();

        let a: u32 = a.try_into().expect("a does not fit into u32");

        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        let a_buf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("a"),
                contents: &a.to_ne_bytes(),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let apply_z_gate_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Z Bind Group"),
            layout: &gpu.apply_z_gate_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: a_buf.as_entire_binding(),
            }],
        });

        let mut compute_pass = encoder.begin_compute_pass(&Default::default());
        compute_pass.set_pipeline(&gpu.apply_z_gate_pipeline);
        compute_pass.set_bind_group(0, &self.tableau_bind_group, &[]);
        compute_pass.set_bind_group(1, &apply_z_gate_bind_group, &[]);
        compute_pass.dispatch_workgroups(
            column_block_length(self.n).div_ceil(WORKGROUP_SIZE),
            1,
            1,
        );
        drop(compute_pass);

        gpu.queue.submit([encoder.finish()]);
    }

    pub fn coeff_ratio(&mut self, w1: &[bool], w2: &[bool]) -> Complex<f64> {
        let n = self.n;
        let w1_len: u32 = w1.len().try_into().expect("w1.len() does not fit into u32");
        let w2_len: u32 = w2.len().try_into().expect("w2.len() does not fit into u32");
        debug_assert_eq!(w1_len, n, "Basis state 1 must have length {n}");
        debug_assert_eq!(w2_len, n, "Basis state 2 must have length {n}");

        // Bring tableau's x part into reduced row echelon form.
        self.bring_into_rref();

        // Derive a stabilizer of the desired form.

        // Compute the (w2, w1) entry in the stabilizer of the correct form.
        Complex::ZERO
    }

    pub fn coeff_ratio_flipped_bit(&mut self, w1: &[bool], a: usize) -> Complex<f64> {
        let n = self.n;
        let w1_len: u32 = w1.len().try_into().expect("w1.len() does not fit into u32");
        debug_assert_eq!(w1_len, n, "Basis state 1 must have length {n}");

        // Bring tableau's x part into reduced row echelon form.
        self.bring_into_rref();

        // Pick the row with a set bit in the given position.

        // Compute the (w2, w1) entry in the stabilizer of the correct form.
        Complex::ZERO
    }

    fn bring_into_rref(&mut self) {
        let gpu = get_gpu();

        let n = self.n;

        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        let mut a_buf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("a"),
                contents: &0u32.to_ne_bytes(),
                usage: wgpu::BufferUsages::STORAGE,
            });
        for col in 0..n {
            let col_buf = gpu
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("col"),
                    contents: &col.to_ne_bytes(),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let a_new_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("a"),
                usage: wgpu::BufferUsages::STORAGE,
                size: U32_SIZE,
                mapped_at_creation: false,
            });
            let elimination_pass_bind_group =
                gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Elimination Pass Bind Group"),
                    layout: &gpu.elimination_pass_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: col_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: a_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: a_new_buf.as_entire_binding(),
                        },
                    ],
                });
            let mut elimination_pass = encoder.begin_compute_pass(&Default::default());
            elimination_pass.set_pipeline(&gpu.elimination_pass_pipeline);
            elimination_pass.set_bind_group(0, &self.tableau_bind_group, &[]);
            elimination_pass.set_bind_group(1, &elimination_pass_bind_group, &[]);
            elimination_pass.dispatch_workgroups(
                column_block_length(n).div_ceil(WORKGROUP_SIZE),
                1,
                1,
            );
            drop(elimination_pass);

            let swap_rows_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Elimination Pass Bind Group"),
                layout: &gpu.swap_rows_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: a_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: col_buf.as_entire_binding(),
                    },
                ],
            });
            let mut swap_pass = encoder.begin_compute_pass(&Default::default());
            swap_pass.set_pipeline(&gpu.swap_rows_pipeline);
            swap_pass.set_bind_group(0, &self.tableau_bind_group, &[]);
            swap_pass.set_bind_group(1, &swap_rows_bind_group, &[]);
            swap_pass.dispatch_workgroups((n + n + 1).div_ceil(WORKGROUP_SIZE), 1, 1);
            drop(swap_pass);

            a_buf = a_new_buf;
        }
        gpu.queue.submit([encoder.finish()]);
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

/// Get the block-length of the columns in the tableau.
fn column_block_length(n: u32) -> u32 {
    // Make room for the auxiliary row.
    (n + 1).div_ceil(BLOCK_SIZE)
}
/// Get the length of the tableau in blocks.
fn tableau_block_length(n: u32) -> u32 {
    column_block_length(n) * (n + n + 1)
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
