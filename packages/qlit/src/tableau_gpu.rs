use std::{fmt::Debug, sync::OnceLock, u8};

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
    apply_gates_bind_group_layout: wgpu::BindGroupLayout,
    split_batches_bind_group_layout: wgpu::BindGroupLayout,
    bring_into_rref_bind_group_layout: wgpu::BindGroupLayout,
    coeff_ratios_flipped_bit_bind_group_layout: wgpu::BindGroupLayout,
    coeff_ratios_bind_group_layout: wgpu::BindGroupLayout,

    zero_pipeline: wgpu::ComputePipeline,
    apply_gates_pipeline: wgpu::ComputePipeline,
    split_batches_pipeline: wgpu::ComputePipeline,
    elimination_pass_pipeline: wgpu::ComputePipeline,
    swap_pass_pipeline: wgpu::ComputePipeline,
    coeff_ratios_flipped_bit_pipeline: wgpu::ComputePipeline,
    coeff_ratios_pipeline: wgpu::ComputePipeline,
}
impl GpuContext {
    pub async fn new() -> GpuContext {
        let instance = wgpu::Instance::new(&Default::default());
        let adapter = instance.request_adapter(&Default::default()).await.unwrap();
        let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();
        println!("Using WGPU adapter: {:?}", adapter.get_info());

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
                    // max_batches
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
                    // active_batches
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
                    // tableau
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
        let apply_gates_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Apply Gates"),
                entries: &[
                    // gates
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
                    // qubit_params
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
                ],
            });
        let split_batches_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Split Batches"),
                entries: &[
                    // qubit
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
        let bring_into_rref_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Bring Into RREF"),
                entries: &[
                    // col_in
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
                    // col_out
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
                    // a_in
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
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
                        binding: 3,
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
                        binding: 4,
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
        let coeff_ratios_flipped_bit_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Coeff Ratios Flipped Bit"),
                entries: &[
                    // w1s
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
                    // flipped_bit
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
                    // factors
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
                    // phases
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
        let coeff_ratios_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Coeff Ratios"),
                entries: &[
                    // w1s
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
                    // factors
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
                    // phases
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

        let apply_gates_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Apply Gates"),
                bind_group_layouts: &[&tableau_bind_group_layout, &apply_gates_bind_group_layout],
                immediate_size: 0,
            });
        let apply_gates_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Apply Gates"),
                layout: Some(&apply_gates_pipeline_layout),
                module: &shader_module,
                entry_point: Some("apply_gates"),
                cache: None,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

        let split_batches_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Split Batches"),
                bind_group_layouts: &[&tableau_bind_group_layout, &split_batches_bind_group_layout],
                immediate_size: 0,
            });
        let split_batches_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Split Batches"),
                layout: Some(&split_batches_pipeline_layout),
                module: &shader_module,
                entry_point: Some("split_batches"),
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

        let coeff_ratios_flipped_bit_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Coeff Ratios Flipped Bit"),
                bind_group_layouts: &[
                    &tableau_bind_group_layout,
                    &coeff_ratios_flipped_bit_bind_group_layout,
                ],
                immediate_size: 0,
            });
        let coeff_ratios_flipped_bit_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Coeff Ratios Flipped Bit"),
                layout: Some(&coeff_ratios_flipped_bit_pipeline_layout),
                module: &shader_module,
                entry_point: Some("coeff_ratios_flipped_bit"),
                cache: None,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

        let coeff_ratios_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Coeff Ratios"),
                bind_group_layouts: &[&tableau_bind_group_layout, &coeff_ratios_bind_group_layout],
                immediate_size: 0,
            });
        let coeff_ratios_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Coeff Ratios"),
                layout: Some(&coeff_ratios_pipeline_layout),
                module: &shader_module,
                entry_point: Some("coeff_ratios"),
                cache: None,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

        GpuContext {
            device,
            queue,

            tableau_bind_group_layout,
            apply_gates_bind_group_layout,
            split_batches_bind_group_layout,
            bring_into_rref_bind_group_layout,
            coeff_ratios_flipped_bit_bind_group_layout,
            coeff_ratios_bind_group_layout,

            zero_pipeline,
            apply_gates_pipeline,
            split_batches_pipeline,
            elimination_pass_pipeline,
            swap_pass_pipeline,
            coeff_ratios_flipped_bit_pipeline: coeff_ratios_flipped_bit_pipeline,
            coeff_ratios_pipeline,
        }
    }
}

pub struct TableauGpu {
    n: u32,
    max_batches: u32,
    active_batches: u32,
    active_batches_buf: wgpu::Buffer,
    tableau_buf: wgpu::Buffer,
    tableau_bind_group: wgpu::BindGroup,

    col_bufs: [wgpu::Buffer; 2],
    a_bufs: [wgpu::Buffer; 2],
    pivot_buf: wgpu::Buffer,

    factors_buf: wgpu::Buffer,
    phases_buf: wgpu::Buffer,
    factors_read_buf: wgpu::Buffer,
    phases_read_buf: wgpu::Buffer,

    gates: Vec<u32>,
    qubit_params: Vec<u32>,

    /// Buffer used to store the output of [`Self::coeff_ratios`] and [`Self::coeff_ratios_flipped_bit`].
    /// Must have length of at least `active_batches` at all times.
    output: Vec<Complex<f64>>,
}
impl TableauGpu {
    pub fn new(n: usize, batch_size_log2: usize) -> Self {
        let gpu = get_gpu();

        let n: u32 = n.try_into().expect("n does not fit into u32");
        let max_batches: u32 = 1 << batch_size_log2;
        let active_batches = 1;

        let n_buf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("n"),
                contents: &n.to_ne_bytes(),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let max_batches_buf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("max_batches"),
                contents: &max_batches.to_ne_bytes(),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let active_batches_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("active_batches"),
            size: U32_SIZE,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let tableau_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tableau"),
            size: tableau_block_length(n, max_batches) as u64 * BLOCK_SIZE_BYTES,
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
                    resource: max_batches_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: active_batches_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: tableau_buf.as_entire_binding(),
                },
            ],
        });

        let col_bufs = [
            gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("col1"),
                size: U32_SIZE,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("col2"),
                size: U32_SIZE,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
        ];
        let a_bufs = [
            gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("a1"),
                size: U32_SIZE,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("a2"),
                size: U32_SIZE,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
        ];
        let pivot_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pivot"),
            size: U32_SIZE,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let factors_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("factors"),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            size: U32_SIZE * max_batches as u64,
            mapped_at_creation: false,
        });
        let phases_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("phases"),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            size: U32_SIZE * max_batches as u64,
            mapped_at_creation: false,
        });
        let factors_read_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("factors (Read Buffer)"),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            size: U32_SIZE * max_batches as u64,
            mapped_at_creation: false,
        });
        let phases_read_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("phases (Read Buffer)"),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            size: U32_SIZE * max_batches as u64,
            mapped_at_creation: false,
        });

        TableauGpu {
            n,
            max_batches,
            active_batches,
            active_batches_buf,
            tableau_buf,
            tableau_bind_group,

            col_bufs,
            a_bufs,
            pivot_buf,

            factors_buf,
            phases_buf,
            factors_read_buf,
            phases_read_buf,

            gates: Vec::new(),
            qubit_params: Vec::new(),

            output: vec![
                Complex::ZERO;
                max_batches
                    .try_into()
                    .expect("max_batches does not fit into usize")
            ],
        }
    }

    pub fn zero(&mut self) {
        let gpu = get_gpu();

        let n = self.n;

        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        let mut compute_pass = encoder.begin_compute_pass(&Default::default());
        compute_pass.set_pipeline(&gpu.zero_pipeline);
        compute_pass.set_bind_group(0, &self.tableau_bind_group, &[]);
        compute_pass.dispatch_workgroups(
            single_column_block_length(n).div_ceil(WORKGROUP_SIZE),
            1,
            1,
        );
        drop(compute_pass);
        gpu.queue.submit([encoder.finish()]);

        self.active_batches = 1;
        self.gates.clear();
        self.qubit_params.clear();
    }

    pub fn apply_cnot_gate(&mut self, a: usize, b: usize) {
        let a: u32 = a.try_into().expect("a does not fit into u32");
        let b: u32 = b.try_into().expect("b does not fit into u32");
        self.gates.push(0);
        self.qubit_params.push(a);
        self.qubit_params.push(b);
    }
    pub fn apply_cz_gate(&mut self, a: usize, b: usize) {
        let a: u32 = a.try_into().expect("a does not fit into u32");
        let b: u32 = b.try_into().expect("b does not fit into u32");
        self.gates.push(1);
        self.qubit_params.push(a);
        self.qubit_params.push(b);
    }
    pub fn apply_h_gate(&mut self, a: usize) {
        let a: u32 = a.try_into().expect("a does not fit into u32");
        self.gates.push(2);
        self.qubit_params.push(a);
    }
    pub fn apply_s_gate(&mut self, a: usize) {
        let a: u32 = a.try_into().expect("a does not fit into u32");
        self.gates.push(3);
        self.qubit_params.push(a);
    }
    pub fn apply_sdg_gate(&mut self, a: usize) {
        let a: u32 = a.try_into().expect("a does not fit into u32");
        self.gates.push(4);
        self.qubit_params.push(a);
    }
    pub fn apply_x_gate(&mut self, a: usize) {
        let a: u32 = a.try_into().expect("a does not fit into u32");
        self.gates.push(5);
        self.qubit_params.push(a);
    }
    pub fn apply_y_gate(&mut self, a: usize) {
        let a: u32 = a.try_into().expect("a does not fit into u32");
        self.gates.push(6);
        self.qubit_params.push(a);
    }
    pub fn apply_z_gate(&mut self, a: usize) {
        let a: u32 = a.try_into().expect("a does not fit into u32");
        self.gates.push(7);
        self.qubit_params.push(a);
    }
    fn submit_gates(&mut self) {
        if self.gates.is_empty() {
            return;
        }

        let gpu = get_gpu();

        let gates_buf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gates"),
                contents: &self
                    .gates
                    .iter()
                    .flat_map(|&g| g.to_ne_bytes())
                    .collect::<Vec<u8>>(),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let qubit_params_buf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("qubit_params"),
                contents: &self
                    .qubit_params
                    .iter()
                    .flat_map(|&p| p.to_ne_bytes())
                    .collect::<Vec<u8>>(),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Apply Gates Bind Group"),
            layout: &gpu.apply_gates_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: gates_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: qubit_params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        let mut compute_pass = encoder.begin_compute_pass(&Default::default());
        compute_pass.set_pipeline(&gpu.apply_gates_pipeline);
        compute_pass.set_bind_group(0, &self.tableau_bind_group, &[]);
        compute_pass.set_bind_group(1, &bind_group, &[]);
        compute_pass.dispatch_workgroups(
            self.active_column_block_length().div_ceil(WORKGROUP_SIZE),
            1,
            1,
        );
        drop(compute_pass);

        gpu.queue.submit([encoder.finish()]);

        self.gates.clear();
        self.qubit_params.clear();
    }

    /// Double the number of batches in the tableau to represent the current state of the tableau both with and without the Z(a) gate applied.
    ///
    /// The first half of the resulting batches will be unchanged,
    /// while the second half will be those where the Z(a) gate is applied.
    pub fn split_batches(&mut self, a: usize) {
        let a: u32 = a.try_into().expect("a does not fit into u32");
        debug_assert!(
            self.active_batches * 2 <= self.max_batches,
            "Cannot split batches beyond max_batches"
        );

        self.submit_gates();

        let gpu = get_gpu();

        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Split Batches Bind Group"),
            layout: &gpu.split_batches_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: gpu
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("qubit"),
                        contents: &a.to_ne_bytes(),
                        usage: wgpu::BufferUsages::UNIFORM,
                    })
                    .as_entire_binding(),
            }],
        });

        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        let mut compute_pass = encoder.begin_compute_pass(&Default::default());
        compute_pass.set_pipeline(&gpu.split_batches_pipeline);
        compute_pass.set_bind_group(0, &self.tableau_bind_group, &[]);
        compute_pass.set_bind_group(1, &bind_group, &[]);
        compute_pass.dispatch_workgroups(
            self.active_column_block_length().div_ceil(WORKGROUP_SIZE),
            1,
            1,
        );
        drop(compute_pass);

        gpu.queue.submit([encoder.finish()]);

        self.active_batches *= 2;
        gpu.queue.write_buffer(
            &self.active_batches_buf,
            0,
            &self.active_batches.to_ne_bytes(),
        );
    }

    /// The coeff. ratio describes the ratio of the coefficients of `w1` and `w2`, such that
    /// ```text
    /// coeff_ratio(w1, w2) * coeff(w1) = coeff(w2)
    /// ```
    ///
    /// This function takes an iterator over basis states `w1s` with length `r_cols`,
    /// and returns a slice of of the coeff. ratios between each `w1s[i]` and `w2`,
    /// each respecting the sign of the `i`th r-column.
    pub fn coeff_ratios<'a>(
        &mut self,
        w1s: impl 'a + IntoIterator<Item = &'a [bool]>,
        w2: &[bool],
    ) -> &[Complex<f64>] {
        let n = self.n;
        let w2_len: u32 = w2.len().try_into().expect("w2.len() does not fit into u32");
        debug_assert_eq!(w2_len, n, "w2 must have length {n}");

        let gpu = get_gpu();

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

        self.coeff_ratios_common(
            w1s,
            &w2_buf,
            &gpu.coeff_ratios_bind_group_layout,
            &gpu.coeff_ratios_pipeline,
        )
    }
    /// Same as [`Self::coeff_ratios`], but for the special case where `w2` is equal to `w1` except for a single flipped bit.
    pub fn coeff_ratios_flipped_bit<'a>(
        &mut self,
        w1s: impl 'a + IntoIterator<Item = &'a [bool]>,
        flipped_bit: usize,
    ) -> &[Complex<f64>] {
        let n = self.n;
        let flipped_bit: u32 = flipped_bit
            .try_into()
            .expect("flipped_bit does not fit into u32");
        debug_assert!(flipped_bit < n, "flipped_bit must be less than {n}");

        let gpu = get_gpu();

        let flipped_bit_buf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("flipped_bit"),
                contents: &flipped_bit.to_ne_bytes(),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        self.coeff_ratios_common(
            w1s,
            &flipped_bit_buf,
            &gpu.coeff_ratios_flipped_bit_bind_group_layout,
            &gpu.coeff_ratios_flipped_bit_pipeline,
        )
    }
    pub fn coeff_ratios_common<'a>(
        &mut self,
        w1s: impl 'a + IntoIterator<Item = &'a [bool]>,
        binding_1_buf: &wgpu::Buffer,
        bind_group_layout: &wgpu::BindGroupLayout,
        pipeline: &wgpu::ComputePipeline,
    ) -> &[Complex<f64>] {
        let n = self.n;
        let active_batches = self.active_batches;

        self.submit_gates();

        // Bring tableau's x part into reduced row echelon form.
        self.bring_into_rref();

        // Derive a stabilizer of the desired form.
        // Compute the (w2, w1) entry in the stabilizer of the correct form.
        let gpu = get_gpu();

        let w1s_contents: Vec<u8> = w1s
            .into_iter()
            .flatten()
            .flat_map(|&b| (if b { 1u32 } else { 0u32 }).to_ne_bytes())
            .collect();
        let w1s_contents_len: u32 = w1s_contents
            .len()
            .try_into()
            .expect("w1s_contents.len() does not fit into u32");
        debug_assert_eq!(
            w1s_contents_len,
            n * active_batches * 4,
            "w1s_contents must have length {n} * {active_batches} * 4"
        );
        let w1s_buf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("w1"),
                contents: &w1s_contents,
                usage: wgpu::BufferUsages::STORAGE,
            });

        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Coeff Ratios Bind Group"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: w1s_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: binding_1_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.factors_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.phases_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        let mut compute_pass = encoder.begin_compute_pass(&Default::default());
        compute_pass.set_pipeline(pipeline);
        compute_pass.set_bind_group(0, &self.tableau_bind_group, &[]);
        compute_pass.set_bind_group(1, &bind_group, &[]);
        compute_pass.dispatch_workgroups(active_batches.div_ceil(WORKGROUP_SIZE), 1, 1);
        drop(compute_pass);

        encoder.copy_buffer_to_buffer(&self.factors_buf, 0, &self.factors_read_buf, 0, None);
        encoder.copy_buffer_to_buffer(&self.phases_buf, 0, &self.phases_read_buf, 0, None);

        gpu.queue.submit([encoder.finish()]);

        self.factors_read_buf
            .map_async(wgpu::MapMode::Read, .., |_| {});
        self.phases_read_buf
            .map_async(wgpu::MapMode::Read, .., |_| {});
        // TODO: To support WebGPU, we need to wait for the callbacks to be invoked.
        // (See https://github.com/gfx-rs/wgpu/blob/993448ab2ca6155f0c859cad49624a119d8bc4b7/examples/standalone/01_hello_compute/src/main.rs)
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();

        let active_batches_usize: usize = active_batches.try_into().unwrap();
        {
            let factors_bytes: &[u8] = &self.factors_read_buf.get_mapped_range(..);
            let phases_bytes: &[u8] = &self.phases_read_buf.get_mapped_range(..);
            let mut factors = bytes_to_u32(factors_bytes);
            let mut phases = bytes_to_u32(phases_bytes);
            for i in 0..active_batches_usize {
                let factor: f64 = factors.next().unwrap().try_into().unwrap();
                let phase = phases.next().unwrap();
                self.output[i] = factor * Complex::I.powu(phase);
            }
        }
        self.factors_read_buf.unmap();
        self.phases_read_buf.unmap();

        return &self.output[..active_batches_usize];
    }

    fn bring_into_rref(&mut self) {
        let gpu = get_gpu();

        let n = self.n;
        let active_batches = self.active_batches;

        let mut col_in_buf = &self.col_bufs[0];
        let mut col_out_buf = &self.col_bufs[1];
        let mut a_in_buf = &self.a_bufs[0];
        let mut a_out_buf = &self.a_bufs[1];

        let mut encoder = gpu.device.create_command_encoder(&Default::default());

        encoder.clear_buffer(col_in_buf, 0, None);
        encoder.clear_buffer(a_in_buf, 0, None);

        for _ in 0..n {
            let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Bring-into-RREF Pass Bind Group"),
                layout: &gpu.bring_into_rref_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: col_in_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: col_out_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: a_in_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: a_out_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.pivot_buf.as_entire_binding(),
                    },
                ],
            });

            let mut elimination_pass = encoder.begin_compute_pass(&Default::default());
            elimination_pass.set_pipeline(&gpu.elimination_pass_pipeline);
            elimination_pass.set_bind_group(0, &self.tableau_bind_group, &[]);
            elimination_pass.set_bind_group(1, &bind_group, &[]);
            elimination_pass.dispatch_workgroups(
                self.active_column_block_length().div_ceil(WORKGROUP_SIZE),
                1,
                1,
            );
            drop(elimination_pass);

            let mut swap_pass = encoder.begin_compute_pass(&Default::default());
            swap_pass.set_pipeline(&gpu.swap_pass_pipeline);
            swap_pass.set_bind_group(0, &self.tableau_bind_group, &[]);
            swap_pass.set_bind_group(1, &bind_group, &[]);
            swap_pass.dispatch_workgroups(active_batches.div_ceil(WORKGROUP_SIZE), 1, 1);
            drop(swap_pass);

            (col_in_buf, col_out_buf) = (col_out_buf, col_in_buf);
            (a_in_buf, a_out_buf) = (a_out_buf, a_in_buf);
        }
        gpu.queue.submit([encoder.finish()]);
    }

    fn active_column_block_length(&self) -> u32 {
        // Make room for the auxiliary row.
        single_column_block_length(self.n) * self.active_batches
    }
}
impl Debug for TableauGpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let gpu = get_gpu();

        let n = self.n;
        let max_batches = self.max_batches;

        let tableau_block_length: u64 = tableau_block_length(n, max_batches).into();
        let tableau_byte_length: u64 = tableau_block_length * BLOCK_SIZE_BYTES;

        let tableau_read_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tableau (Read Buffer)"),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            size: tableau_byte_length,
            mapped_at_creation: false,
        });
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.tableau_buf, 0, &tableau_read_buf, 0, None);
        gpu.queue.submit([encoder.finish()]);

        tableau_read_buf.map_async(wgpu::MapMode::Read, .., |_| {});
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        let tableau_bytes: &[u8] = &tableau_read_buf.get_mapped_range(..);

        let tableau: Vec<u32> = bytes_to_u32(tableau_bytes).collect();
        writeln!(f, "TableauGpu(n = {})", n)?;
        for batch in 0..max_batches {
            writeln!(f, "--- Batch ---")?;
            for row in 0..n as usize {
                for col in 0..n as usize {
                    write!(
                        f,
                        " {:?} ",
                        if x_bit(
                            n as usize,
                            max_batches as usize,
                            &tableau,
                            batch as usize,
                            row,
                            col
                        ) {
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
                        if z_bit(
                            n as usize,
                            max_batches as usize,
                            &tableau,
                            batch as usize,
                            row,
                            col
                        ) {
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
                    if r_bit(
                        n as usize,
                        max_batches as usize,
                        &tableau,
                        batch as usize,
                        row
                    ) {
                        1
                    } else {
                        0
                    }
                )?;
            }
        }

        Ok(())
    }
}
fn bit(
    n: usize,
    max_batches: usize,
    tableau: &Vec<u32>,
    batch_index: usize,
    row: usize,
    j: usize,
) -> bool {
    let batch_start_block = batch_index * single_column_block_length(n as u32) as usize;
    let row_block_index = row / 32;
    let row_bit_index = row % 32;
    let row_bitmask = bitmask(row_bit_index);
    tableau[column_block_index(n, max_batches, batch_start_block + row_block_index, j)]
        & row_bitmask
        != 0
}
fn x_bit(
    n: usize,
    max_batches: usize,
    tableau: &Vec<u32>,
    batch_index: usize,
    row: usize,
    q: usize,
) -> bool {
    bit(n, max_batches, tableau, batch_index, row, 2 * q)
}
fn z_bit(
    n: usize,
    max_batches: usize,
    tableau: &Vec<u32>,
    batch_index: usize,
    row: usize,
    q: usize,
) -> bool {
    bit(n, max_batches, tableau, batch_index, row, 2 * q + 1)
}
fn r_bit(n: usize, max_batches: usize, tableau: &Vec<u32>, batch_index: usize, row: usize) -> bool {
    bit(n, max_batches, tableau, batch_index, row, 2 * n)
}
fn bitmask(i: usize) -> BitBlock {
    debug_assert!(i < 32);
    1 << (32 - 1 - i)
}
fn column_block_index(n: usize, max_batches: usize, i: usize, j: usize) -> usize {
    j * column_block_length(n as u32, max_batches as u32) as usize + i
}

// Get the block-length of the columns in a single tableau batch.
fn single_column_block_length(n: u32) -> u32 {
    // Make room for the auxiliary row.
    (n + 1).div_ceil(BLOCK_SIZE)
}
// Get the block-length of the columns of all the combined tableau batches.
fn column_block_length(n: u32, max_batches: u32) -> u32 {
    single_column_block_length(n) * max_batches
}
/// Get the length of the tableau in blocks.
fn tableau_block_length(n: u32, max_batches: u32) -> u32 {
    column_block_length(n, max_batches) * (n + n + 1)
}

/// Convert the contents of a slice of bytes into an iterator over u32s using native endianness.
fn bytes_to_u32(bytes: &[u8]) -> impl Iterator<Item = u32> {
    let (chunks, remainder) = bytes.as_chunks::<4>();
    debug_assert_eq!(remainder.len(), 0);
    chunks.iter().copied().map(u32::from_ne_bytes)
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

            let mut g = TableauGpu::new(8, 0);
            g.zero();
            apply_clifford_circuit(&mut g, &circuit);
            let result = g.coeff_ratios([w1.as_slice()], &w2);

            let expected = if i == 0b0000_0000 {
                Complex::ONE
            } else {
                Complex::ZERO
            };
            assert_eq!(result, [expected], "{i:008b}");
        }
    }

    #[test]
    fn hadamard() {
        let circuit = CliffordTCircuit::new(8, [H(1)]).unwrap();

        let w1 = bits_to_bools(0b0000_0000);
        let w2 = bits_to_bools(0b0100_0000);

        let mut g = TableauGpu::new(8, 0);
        g.zero();
        apply_clifford_circuit(&mut g, &circuit);
        let result = g.coeff_ratios([w1.as_slice()], &w2);

        assert_eq!(result, [Complex::ONE]);
    }

    #[test]
    fn imaginary() {
        let circuit = CliffordTCircuit::new(8, [H(0), S(0)]).unwrap();

        let w1 = bits_to_bools(0b0000_0000);
        for i in 0b0000_0000..=0b1111_1111 {
            let w2 = bits_to_bools(i);

            let mut g = TableauGpu::new(8, 0);
            g.zero();
            apply_clifford_circuit(&mut g, &circuit);
            let result = g.coeff_ratios([w1.as_slice()], &w2);

            let expected = if i == 0b0000_0000 {
                Complex::ONE
            } else if i == 0b1000_0000 {
                Complex::I
            } else {
                Complex::ZERO
            };
            assert_eq!(result, [expected], "{i:008b}");
        }
    }

    #[test]
    fn negative_imaginary() {
        let circuit = CliffordTCircuit::new(8, [H(0), S(0)]).unwrap();

        let w1 = bits_to_bools(0b1000_0000);
        for i in 0b0000_0000..=0b1111_1111 {
            let w2 = bits_to_bools(i);

            let mut g = TableauGpu::new(8, 0);
            g.zero();
            apply_clifford_circuit(&mut g, &circuit);
            let result = g.coeff_ratios([w1.as_slice()], &w2);

            let expected = if i == 0b0000_0000 {
                -Complex::I
            } else if i == 0b1000_0000 {
                Complex::ONE
            } else {
                Complex::ZERO
            };
            assert_eq!(result, [expected], "{i:008b}");
        }
    }

    #[test]
    fn flipped() {
        let circuit = CliffordTCircuit::new(8, [H(0), S(0), S(0), H(0)]).unwrap();

        let w1 = bits_to_bools(0b1000_0000);
        for i in 0b0000_0000..=0b1111_1111 {
            let w2 = bits_to_bools(i);

            let mut g = TableauGpu::new(8, 0);
            g.zero();
            apply_clifford_circuit(&mut g, &circuit);
            let result = g.coeff_ratios([w1.as_slice()], &w2);

            let expected = if i == 0b1000_0000 {
                Complex::ONE
            } else {
                Complex::ZERO
            };
            assert_eq!(result, [expected], "{i:008b}");
        }
    }

    #[test]
    fn bell_state() {
        let circuit = CliffordTCircuit::new(8, [H(0), Cnot(0, 1)]).unwrap();

        let w1 = bits_to_bools(0b1100_0000);
        for i in 0b0000_0000..=0b1111_1111 {
            let w2 = bits_to_bools(i);

            let mut g = TableauGpu::new(8, 0);
            g.zero();
            apply_clifford_circuit(&mut g, &circuit);
            let result = g.coeff_ratios([w1.as_slice()], &w2);

            let expected = if [0b0000_0000, 0b1100_0000].contains(&i) {
                Complex::ONE
            } else {
                Complex::ZERO
            };
            assert_eq!(result, [expected], "{i:008b}");
        }
    }

    #[test]
    fn split() {
        let mut g = TableauGpu::new(8, 5);
        g.zero();
        for _ in 0..5 {
            g.split_batches(0);
        }

        let w1: &[bool] = &bits_to_bools(0b0000_0000);
        let w1s = [w1; 32];

        for i in 0b0000_0000..=0b1111_1111 {
            let w2 = bits_to_bools(i);
            let result = g.coeff_ratios(w1s.clone(), &w2);

            let expected = match i {
                0b0000_0000 => Complex::ONE,
                _ => Complex::ZERO,
            };
            assert_eq!(result, [expected; 32], "{i:008b}");
        }
    }

    #[test]
    fn split_after_h() {
        let mut g = TableauGpu::new(8, 1);
        g.zero();
        g.apply_h_gate(0);
        g.split_batches(0);

        let w11: &[bool] = &bits_to_bools(0b0000_0000);
        let w12: &[bool] = &bits_to_bools(0b1000_0000);
        let w1s = [w11, w12];

        for i in 0b0000_0000..=0b1111_1111 {
            let w2 = bits_to_bools(i);
            let result = g.coeff_ratios(w1s.clone(), &w2);

            let expected = match i {
                0b0000_0000 => [Complex::ONE, -Complex::ONE],
                0b1000_0000 => [Complex::ONE, Complex::ONE],
                _ => [Complex::ZERO, Complex::ZERO],
            };
            assert_eq!(result, expected, "{i:008b}");
        }
    }

    #[test]
    fn split_before() {
        let mut g = TableauGpu::new(8, 1);
        g.zero();
        g.split_batches(0);
        g.apply_h_gate(1);
        g.apply_h_gate(2);
        g.apply_cnot_gate(1, 0);
        g.apply_cnot_gate(2, 1);

        let w11: &[bool] = &bits_to_bools(0b0000_0000);
        let w12: &[bool] = &bits_to_bools(0b1100_0000);
        let w1s = [w11, w12];

        for i in 0b0000_0000..=0b1111_1111 {
            let w2 = bits_to_bools(i);
            let result = g.coeff_ratios(w1s.clone(), &w2);

            let expected = match i {
                0b0000_0000 | 0b1100_0000 | 0b1010_0000 | 0b0110_0000 => [Complex::ONE; 2],
                _ => [Complex::ZERO; 2],
            };
            assert_eq!(result, expected, "{i:008b}");
        }
    }

    #[test]
    fn split_after() {
        let mut g = TableauGpu::new(8, 1);
        g.zero();
        g.apply_h_gate(0);
        g.apply_h_gate(1);
        g.apply_cnot_gate(1, 0);
        g.split_batches(0);

        let w1: &[bool] = &bits_to_bools(0b0000_0000);
        let w1s = [w1; 2];

        for i in 0b0000_0000..=0b1111_1111 {
            let w2 = bits_to_bools(i);
            let result = g.coeff_ratios(w1s.clone(), &w2);

            let expected = match i {
                0b0000_0000 | 0b0100_0000 => [Complex::ONE, Complex::ONE],
                0b1000_0000 | 0b1100_0000 => [Complex::ONE, -Complex::ONE],
                _ => [Complex::ZERO; 2],
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

            let mut g = TableauGpu::new(8, 0);
            g.zero();
            apply_clifford_circuit(&mut g, &circuit);
            let result = g.coeff_ratios([w1.as_slice()], &w2);

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
            assert_eq!(result, [expected], "{i:008b}");
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

        let w1 = bits_to_bools(0b1000_0000);
        let mut g = TableauGpu::new(8, 0);
        g.zero();
        apply_clifford_circuit(&mut g, &circuit);

        assert_eq!(
            g.coeff_ratios_flipped_bit([w1.as_slice()], 0),
            [-Complex::ONE]
        );
        assert_eq!(
            g.coeff_ratios_flipped_bit([w1.as_slice()], 1),
            [-Complex::ONE]
        );
        assert_eq!(
            g.coeff_ratios_flipped_bit([w1.as_slice()], 2),
            [Complex::ZERO]
        );
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

        let mut g = TableauGpu::new(8, 0);
        g.zero();
        apply_clifford_circuit(&mut g, &circuit);

        let w1 = bits_to_bools(0b1000_0000);
        for i in 0b0000_0000..=0b1111_1111 {
            let w2 = bits_to_bools(i);

            let result = g.coeff_ratios([w1.as_slice()], &w2);

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
            assert_eq!(result, [expected], "{i:008b}");
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
