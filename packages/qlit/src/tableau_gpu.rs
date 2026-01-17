use std::{fmt::Debug, sync::OnceLock};

use num_complex::Complex;
use wgpu::util::DeviceExt;

type BitBlock = u32;
const BLOCK_SIZE_BYTES: u64 = size_of::<BitBlock>() as u64;
const BLOCK_SIZE: u32 = (BLOCK_SIZE_BYTES * 8) as u32;

const WORKGROUP_SIZE: u32 = 64;
const U32_SIZE: u64 = size_of::<u32>() as u64;

/// Initialize the global GPU context.
///
/// This will happen automatically the first time it's needed,
/// but this can be called to pre-empt that work at a more appropriate time.
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
    unary_gate_bind_group_layout: wgpu::BindGroupLayout,
    binary_gate_bind_group_layout: wgpu::BindGroupLayout,
    bring_into_rref_bind_group_layout: wgpu::BindGroupLayout,
    coeff_ratio_bind_group_layout: wgpu::BindGroupLayout,

    zero_pipeline: wgpu::ComputePipeline,
    apply_cnot_gate_pipeline: wgpu::ComputePipeline,
    apply_h_gate_pipeline: wgpu::ComputePipeline,
    apply_s_gate_pipeline: wgpu::ComputePipeline,
    apply_x_gate_pipeline: wgpu::ComputePipeline,
    apply_y_gate_pipeline: wgpu::ComputePipeline,
    apply_z_gate_pipeline: wgpu::ComputePipeline,
    elimination_pass_pipeline: wgpu::ComputePipeline,
    swap_pass_pipeline: wgpu::ComputePipeline,
    coeff_ratio_pipeline: wgpu::ComputePipeline,
}
impl GpuContext {
    pub async fn new() -> GpuContext {
        let instance = wgpu::Instance::new(&Default::default());
        let adapter = instance.request_adapter(&Default::default()).await.unwrap();
        let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();

        let shader_module = device.create_shader_module(wgpu::include_wgsl!("tableau.wgsl"));

        let tableau_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Tableau"),
                entries: &[
                    // n
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
                    // tableau
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
        let unary_gate_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Gate Application"),
                entries: &[
                    // a
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
                ],
            });
        let binary_gate_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Gate Application"),
                entries: &[
                    // a
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
                    // b
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
        let bring_into_rref_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Bring Into RREF"),
                entries: &[
                    // col
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
                    // a_in
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
                    // a_out
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
                    // pivot_out
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
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
        let coeff_ratio_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Bring Into RREF"),
                entries: &[
                    // w1
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
                    // w2
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
                    // factor
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
                    // phase
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
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

        let zero_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Zero Tableau"),
            bind_group_layouts: &[&tableau_bind_group_layout],
            immediate_size: 0,
        });
        let zero_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Zero Tableau"),
            layout: Some(&zero_pipeline_layout),
            module: &shader_module,
            entry_point: Some("zero"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let unary_gate_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Gate Application"),
                bind_group_layouts: &[&tableau_bind_group_layout, &unary_gate_bind_group_layout],
                immediate_size: 0,
            });
        let binary_gate_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Gate Application"),
                bind_group_layouts: &[&tableau_bind_group_layout, &binary_gate_bind_group_layout],
                immediate_size: 0,
            });
        let apply_cnot_gate_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Apply Cnot Gate"),
                layout: Some(&binary_gate_pipeline_layout),
                module: &shader_module,
                entry_point: Some("apply_cnot_gate"),
                cache: None,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });
        let apply_h_gate_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Apply H Gate"),
                layout: Some(&unary_gate_pipeline_layout),
                module: &shader_module,
                entry_point: Some("apply_h_gate"),
                cache: None,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });
        let apply_s_gate_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Apply S Gate"),
                layout: Some(&unary_gate_pipeline_layout),
                module: &shader_module,
                entry_point: Some("apply_s_gate"),
                cache: None,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });
        let apply_x_gate_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Apply X Gate"),
                layout: Some(&unary_gate_pipeline_layout),
                module: &shader_module,
                entry_point: Some("apply_x_gate"),
                cache: None,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });
        let apply_y_gate_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Apply Y Gate"),
                layout: Some(&unary_gate_pipeline_layout),
                module: &shader_module,
                entry_point: Some("apply_y_gate"),
                cache: None,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });
        let apply_z_gate_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Apply Z Gate"),
                layout: Some(&unary_gate_pipeline_layout),
                module: &shader_module,
                entry_point: Some("apply_z_gate"),
                cache: None,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

        let bring_into_rref_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Bring Into RREF"),
                bind_group_layouts: &[
                    &tableau_bind_group_layout,
                    &bring_into_rref_bind_group_layout,
                ],
                immediate_size: 0,
            });
        let elimination_pass_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Elimination Pass"),
                layout: Some(&bring_into_rref_pipeline_layout),
                module: &shader_module,
                entry_point: Some("elimination_pass"),
                cache: None,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });
        let swap_pass_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Swap Pass"),
            layout: Some(&bring_into_rref_pipeline_layout),
            module: &shader_module,
            entry_point: Some("swap_pass"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let coeff_ratio_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Bring Into RREF"),
                bind_group_layouts: &[&tableau_bind_group_layout, &coeff_ratio_bind_group_layout],
                immediate_size: 0,
            });
        let coeff_ratio_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Coeff Ratio"),
                layout: Some(&coeff_ratio_pipeline_layout),
                module: &shader_module,
                entry_point: Some("coeff_ratio"),
                cache: None,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

        GpuContext {
            device,
            queue,

            tableau_bind_group_layout,
            unary_gate_bind_group_layout,
            binary_gate_bind_group_layout,
            bring_into_rref_bind_group_layout,
            coeff_ratio_bind_group_layout,

            zero_pipeline,
            apply_cnot_gate_pipeline,
            apply_h_gate_pipeline,
            apply_s_gate_pipeline,
            apply_x_gate_pipeline,
            apply_y_gate_pipeline,
            apply_z_gate_pipeline,
            elimination_pass_pipeline,
            swap_pass_pipeline,
            coeff_ratio_pipeline,
        }
    }
}

pub struct TableauGpu {
    n: u32,
    tableau_buf: wgpu::Buffer,
    tableau_bind_group: wgpu::BindGroup,
}
impl TableauGpu {
    pub fn zero(n: usize) -> Self {
        let gpu = get_gpu();

        let n: u32 = n.try_into().expect("n does not fit into u32");
        let n_buf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("n"),
                contents: &n.to_ne_bytes(),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let tableau_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tableau"),
            size: tableau_block_length(n) as u64 * BLOCK_SIZE_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
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

        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        let mut compute_pass = encoder.begin_compute_pass(&Default::default());
        compute_pass.set_pipeline(&gpu.zero_pipeline);
        compute_pass.set_bind_group(0, &tableau_bind_group, &[]);
        compute_pass.dispatch_workgroups(column_block_length(n).div_ceil(WORKGROUP_SIZE), 1, 1);
        drop(compute_pass);
        gpu.queue.submit([encoder.finish()]);

        TableauGpu {
            n,
            tableau_buf,
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
            layout: &gpu.binary_gate_bind_group_layout,
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
            layout: &gpu.unary_gate_bind_group_layout,
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
            layout: &gpu.unary_gate_bind_group_layout,
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
    pub fn apply_x_gate(&mut self, a: usize) {
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
        let apply_x_gate_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("X Bind Group"),
            layout: &gpu.unary_gate_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: a_buf.as_entire_binding(),
            }],
        });

        let mut compute_pass = encoder.begin_compute_pass(&Default::default());
        compute_pass.set_pipeline(&gpu.apply_x_gate_pipeline);
        compute_pass.set_bind_group(0, &self.tableau_bind_group, &[]);
        compute_pass.set_bind_group(1, &apply_x_gate_bind_group, &[]);
        compute_pass.dispatch_workgroups(
            column_block_length(self.n).div_ceil(WORKGROUP_SIZE),
            1,
            1,
        );
        drop(compute_pass);

        gpu.queue.submit([encoder.finish()]);
    }
    pub fn apply_y_gate(&mut self, a: usize) {
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
        let apply_y_gate_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Y Bind Group"),
            layout: &gpu.unary_gate_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: a_buf.as_entire_binding(),
            }],
        });

        let mut compute_pass = encoder.begin_compute_pass(&Default::default());
        compute_pass.set_pipeline(&gpu.apply_y_gate_pipeline);
        compute_pass.set_bind_group(0, &self.tableau_bind_group, &[]);
        compute_pass.set_bind_group(1, &apply_y_gate_bind_group, &[]);
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
            layout: &gpu.unary_gate_bind_group_layout,
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
        let gpu = get_gpu();
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        let w1_buf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("w1"),
                contents: &w1
                    .iter()
                    .flat_map(|&b| (if b { 1u32 } else { 0u32 }).to_ne_bytes())
                    .collect::<Vec<u8>>(),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let w2_buf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("w2"),
                contents: &w2
                    .iter()
                    .flat_map(|&b| (if b { 1u32 } else { 0u32 }).to_ne_bytes())
                    .collect::<Vec<u8>>(),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let factor_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("factor"),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            size: U32_SIZE,
            mapped_at_creation: false,
        });
        let phase_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("phase"),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            size: U32_SIZE,
            mapped_at_creation: false,
        });
        let coeff_ratio_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Coeff Ratio Bind Group"),
            layout: &gpu.coeff_ratio_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: w1_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: w2_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: factor_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: phase_buf.as_entire_binding(),
                },
            ],
        });

        let mut compute_pass = encoder.begin_compute_pass(&Default::default());
        compute_pass.set_pipeline(&gpu.coeff_ratio_pipeline);
        compute_pass.set_bind_group(0, &self.tableau_bind_group, &[]);
        compute_pass.set_bind_group(1, &coeff_ratio_bind_group, &[]);
        compute_pass.dispatch_workgroups(1, 1, 1);
        drop(compute_pass);

        let factor_read_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("factor (Read Buffer)"),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            size: U32_SIZE,
            mapped_at_creation: false,
        });
        let phase_read_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("phase (Read Buffer)"),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            size: U32_SIZE,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&factor_buf, 0, &factor_read_buf, 0, U32_SIZE);
        encoder.copy_buffer_to_buffer(&phase_buf, 0, &phase_read_buf, 0, U32_SIZE);

        gpu.queue.submit([encoder.finish()]);

        factor_read_buf.map_async(wgpu::MapMode::Read, .., |_| {});
        phase_read_buf.map_async(wgpu::MapMode::Read, .., |_| {});
        // TODO: To support WebGPU, we need to wait for the callbacks to be invoked.
        // (See https://github.com/gfx-rs/wgpu/blob/993448ab2ca6155f0c859cad49624a119d8bc4b7/examples/standalone/01_hello_compute/src/main.rs)
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();

        let factor_bytes: &[u8] = &factor_read_buf.get_mapped_range(..);
        let phase_bytes: &[u8] = &phase_read_buf.get_mapped_range(..);
        let factor: f64 = u32::from_ne_bytes(factor_bytes.try_into().unwrap()).into();
        let phase = u32::from_ne_bytes(phase_bytes.try_into().unwrap());

        factor * Complex::I.powu(phase)
    }

    pub fn coeff_ratio_flipped_bit(&mut self, w1: &[bool], flipped_bit: usize) -> Complex<f64> {
        // TODO: Optimize (?)
        let mut w2 = w1.to_vec();
        w2[flipped_bit] = !w2[flipped_bit];
        self.coeff_ratio(w1, &w2)
    }

    fn bring_into_rref(&mut self) {
        let gpu = get_gpu();

        let n = self.n;

        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        let mut a_in_buf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("a1"),
                contents: &0u32.to_ne_bytes(),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let mut a_out_buf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("a2"),
                contents: &0u32.to_ne_bytes(),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let pivot_buf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pivot"),
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
            let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Bring-into-RREF Pass Bind Group"),
                layout: &gpu.bring_into_rref_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: col_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: a_in_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: a_out_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: pivot_buf.as_entire_binding(),
                    },
                ],
            });

            let mut elimination_pass = encoder.begin_compute_pass(&Default::default());
            elimination_pass.set_pipeline(&gpu.elimination_pass_pipeline);
            elimination_pass.set_bind_group(0, &self.tableau_bind_group, &[]);
            elimination_pass.set_bind_group(1, &bind_group, &[]);
            elimination_pass.dispatch_workgroups(
                column_block_length(n).div_ceil(WORKGROUP_SIZE),
                1,
                1,
            );
            drop(elimination_pass);

            let mut swap_pass = encoder.begin_compute_pass(&Default::default());
            swap_pass.set_pipeline(&gpu.swap_pass_pipeline);
            swap_pass.set_bind_group(0, &self.tableau_bind_group, &[]);
            swap_pass.set_bind_group(1, &bind_group, &[]);
            swap_pass.dispatch_workgroups((n + n + 1).div_ceil(WORKGROUP_SIZE), 1, 1);
            drop(swap_pass);

            (a_in_buf, a_out_buf) = (a_out_buf, a_in_buf);
        }
        gpu.queue.submit([encoder.finish()]);
    }
}
impl Debug for TableauGpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let gpu = get_gpu();

        let n = self.n;

        let tableau_block_length: u64 = tableau_block_length(n).into();
        let tableau_byte_length: u64 = tableau_block_length * BLOCK_SIZE_BYTES;

        let tableau_read_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tableau (Read Buffer)"),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            size: tableau_byte_length,
            mapped_at_creation: false,
        });
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(
            &self.tableau_buf,
            0,
            &tableau_read_buf,
            0,
            tableau_byte_length,
        );
        gpu.queue.submit([encoder.finish()]);

        tableau_read_buf.map_async(wgpu::MapMode::Read, .., |_| {});
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        let tableau_bytes: &[u8] = &tableau_read_buf.get_mapped_range(..);

        let (chunks, chunk_remainder) = tableau_bytes.as_chunks::<4>();
        assert_eq!(chunks.len(), tableau_block_length as usize);
        assert_eq!(chunk_remainder.len(), 0);

        let tableau: Vec<u32> = chunks.iter().map(|&c| u32::from_ne_bytes(c)).collect();
        writeln!(f, "TableauGpu(n = {})", n)?;
        for row in 0..n as usize {
            for col in 0..n as usize {
                write!(
                    f,
                    " {:?} ",
                    if x_bit(n as usize, &tableau, row, col) {
                        1
                    } else {
                        0
                    }
                )?;
            }
            write!(f, " | ")?;
            for col in 0..n as usize {
                write!(
                    f,
                    " {:?} ",
                    if z_bit(n as usize, &tableau, row, col) {
                        1
                    } else {
                        0
                    }
                )?;
            }
            write!(f, " | ")?;
            write!(
                f,
                " {:?} \n",
                if r_bit(n as usize, &tableau, row) {
                    1
                } else {
                    0
                }
            )?;
        }

        Ok(())
    }
}

fn bit(n: usize, tableau: &Vec<u32>, row: usize, j: usize) -> bool {
    let row_block_index = row / 32;
    let row_bit_index = row % 32;
    let row_bitmask = bitmask(row_bit_index);
    tableau[column_block_index(n, row_block_index, j)] & row_bitmask != 0
}
fn x_bit(n: usize, tableau: &Vec<u32>, row: usize, q: usize) -> bool {
    bit(n, tableau, row, 2 * q)
}
fn z_bit(n: usize, tableau: &Vec<u32>, row: usize, q: usize) -> bool {
    bit(n, tableau, row, 2 * q + 1)
}
fn r_bit(n: usize, tableau: &Vec<u32>, row: usize) -> bool {
    bit(n, tableau, row, 2 * n)
}
fn bitmask(i: usize) -> BitBlock {
    debug_assert!(i < 32);
    1 << (32 - 1 - i)
}
fn column_block_index(n: usize, i: usize, j: usize) -> usize {
    j * column_block_length(n as u32) as usize + i
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

#[cfg(test)]
mod tests {
    use crate::circuit::{CliffordTCircuit, CliffordTGate::*};
    use crate::utils::bits_to_bools;

    use super::*;

    #[test]
    fn zero() {
        let circuit = CliffordTCircuit::new(8, []).unwrap();

        let w1 = bits_to_bools(0b0000_0000);
        for i in 0b0000_0000..=0b1111_1111 {
            let w2 = bits_to_bools(i);

            let mut g = TableauGpu::zero(8);
            apply_clifford_circuit(&mut g, &circuit);
            let result = g.coeff_ratio(&w1, &w2);

            let expected = if i == 0b0000_0000 {
                Complex::ONE
            } else {
                Complex::ZERO
            };
            assert_eq!(result, expected, "{i:008b}");
        }
    }

    #[test]
    fn hadamard() {
        let circuit = CliffordTCircuit::new(8, [H(1)]).unwrap();

        let w1 = bits_to_bools(0b0000_0000);
        let w2 = bits_to_bools(0b0100_0000);

        let mut g = TableauGpu::zero(8);
        apply_clifford_circuit(&mut g, &circuit);
        let result = g.coeff_ratio(&w1, &w2);

        assert_eq!(result, Complex::ONE);
    }

    #[test]
    fn imaginary() {
        let circuit = CliffordTCircuit::new(8, [H(0), S(0)]).unwrap();

        let w1 = bits_to_bools(0b0000_0000);
        for i in 0b0000_0000..=0b1111_1111 {
            let w2 = bits_to_bools(i);

            let mut g = TableauGpu::zero(8);
            apply_clifford_circuit(&mut g, &circuit);
            let result = g.coeff_ratio(&w1, &w2);

            let expected = if i == 0b0000_0000 {
                Complex::ONE
            } else if i == 0b1000_0000 {
                Complex::I
            } else {
                Complex::ZERO
            };
            assert_eq!(result, expected, "{i:008b}");
        }
    }

    #[test]
    fn negative_imaginary() {
        let circuit = CliffordTCircuit::new(8, [H(0), S(0)]).unwrap();

        let w1 = bits_to_bools(0b1000_0000);
        for i in 0b0000_0000..=0b1111_1111 {
            let w2 = bits_to_bools(i);

            let mut g = TableauGpu::zero(8);
            apply_clifford_circuit(&mut g, &circuit);
            let result = g.coeff_ratio(&w1, &w2);

            let expected = if i == 0b0000_0000 {
                -Complex::I
            } else if i == 0b1000_0000 {
                Complex::ONE
            } else {
                Complex::ZERO
            };
            assert_eq!(result, expected, "{i:008b}");
        }
    }

    #[test]
    fn flipped() {
        let circuit = CliffordTCircuit::new(8, [H(0), S(0), S(0), H(0)]).unwrap();

        let w1 = bits_to_bools(0b1000_0000);
        for i in 0b0000_0000..=0b1111_1111 {
            let w2 = bits_to_bools(i);

            let mut g = TableauGpu::zero(8);
            apply_clifford_circuit(&mut g, &circuit);
            let result = g.coeff_ratio(&w1, &w2);

            let expected = if i == 0b1000_0000 {
                Complex::ONE
            } else {
                Complex::ZERO
            };
            assert_eq!(result, expected, "{i:008b}");
        }
    }

    #[test]
    fn bell_state() {
        let circuit = CliffordTCircuit::new(8, [H(0), Cnot(0, 1)]).unwrap();

        let w1 = bits_to_bools(0b1100_0000);
        for i in 0b0000_0000..=0b1111_1111 {
            let w2 = bits_to_bools(i);

            let mut g = TableauGpu::zero(8);
            apply_clifford_circuit(&mut g, &circuit);
            let result = g.coeff_ratio(&w1, &w2);

            let expected = if [0b0000_0000, 0b1100_0000].contains(&i) {
                Complex::ONE
            } else {
                Complex::ZERO
            };
            assert_eq!(result, expected, "{i:008b}");
        }
    }

    #[test]
    fn larger_circuit() {
        let circuit = CliffordTCircuit::new(
            8,
            [
                H(0),
                H(1),
                S(2),
                H(3),
                S(1),
                S(0),
                Cnot(2, 3),
                S(1),
                H(0),
                S(3),
                Cnot(1, 0),
                S(3),
                H(1),
                S(3),
                S(1),
                S(3),
                H(1),
                Cnot(3, 2),
                H(1),
                Cnot(3, 1),
            ],
        )
        .unwrap();

        let w1 = bits_to_bools(0b1000_0000);
        for i in 0b0000_0000..=0b1111_1111 {
            let w2 = bits_to_bools(i);

            let mut g = TableauGpu::zero(8);
            apply_clifford_circuit(&mut g, &circuit);
            let result = g.coeff_ratio(&w1, &w2);

            let expected = if [
                0b0000_0000,
                0b0100_0000,
                0b1100_0000,
                0b0011_0000,
                0b0111_0000,
                0b1011_0000,
            ]
            .contains(&i)
            {
                -Complex::ONE
            } else if [0b1000_0000, 0b1111_0000].contains(&i) {
                Complex::ONE
            } else {
                Complex::ZERO
            };
            assert_eq!(result, expected, "{i:008b}");
        }
    }

    #[test]
    fn bitflip_ratio() {
        let circuit = CliffordTCircuit::new(
            8,
            [
                H(0),
                H(1),
                S(2),
                H(3),
                S(1),
                S(0),
                Cnot(2, 3),
                S(1),
                H(0),
                S(3),
                Cnot(1, 0),
                S(3),
                H(1),
                S(3),
                S(1),
                S(3),
                H(1),
                Cnot(3, 2),
                H(1),
                Cnot(3, 1),
            ],
        )
        .unwrap();

        let w = bits_to_bools(0b1000_0000);
        let mut g = TableauGpu::zero(8);
        apply_clifford_circuit(&mut g, &circuit);

        assert_eq!(g.coeff_ratio_flipped_bit(&w, 0), -Complex::ONE);
        assert_eq!(g.coeff_ratio_flipped_bit(&w, 1), -Complex::ONE);
        assert_eq!(g.coeff_ratio_flipped_bit(&w, 2), Complex::ZERO);
    }

    #[test]
    fn repeated_reading() {
        let circuit = CliffordTCircuit::new(
            8,
            [
                H(0),
                H(1),
                S(2),
                H(3),
                S(1),
                S(0),
                Cnot(2, 3),
                S(1),
                H(0),
                S(3),
                Cnot(1, 0),
                S(3),
                H(1),
                S(3),
                S(1),
                S(3),
                H(1),
                Cnot(3, 2),
                H(1),
                Cnot(3, 1),
            ],
        )
        .unwrap();

        let mut g = TableauGpu::zero(8);
        apply_clifford_circuit(&mut g, &circuit);

        let w1 = bits_to_bools(0b1000_0000);
        for i in 0b0000_0000..=0b1111_1111 {
            let w2 = bits_to_bools(i);

            let result = g.coeff_ratio(&w1, &w2);

            let expected = if [
                0b0000_0000,
                0b0100_0000,
                0b1100_0000,
                0b0011_0000,
                0b0111_0000,
                0b1011_0000,
            ]
            .contains(&i)
            {
                -Complex::ONE
            } else if [0b1000_0000, 0b1111_0000].contains(&i) {
                Complex::ONE
            } else {
                Complex::ZERO
            };
            assert_eq!(result, expected, "{i:008b}");
        }
    }

    fn apply_clifford_circuit(g: &mut TableauGpu, circuit: &CliffordTCircuit) {
        for &gate in circuit.gates() {
            match gate {
                S(a) => g.apply_s_gate(a),
                H(a) => g.apply_h_gate(a),
                Cnot(a, b) => g.apply_cnot_gate(a, b),
                _ => unreachable!(),
            }
        }
    }
}
