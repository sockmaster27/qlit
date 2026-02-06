use std::{cmp::max, mem, sync::OnceLock};

use num_complex::Complex;
use wgpu::util::DeviceExt;

use crate::{CliffordTCircuit, CliffordTGate};

type BitBlock = u32;
const BLOCK_SIZE_BYTES: u64 = size_of::<BitBlock>() as u64;
const BLOCK_SIZE: u32 = (BLOCK_SIZE_BYTES * 8) as u32;

const WORKGROUP_SIZE: u32 = 64;
const U32_SIZE: u64 = size_of::<u32>() as u64;
const COMPLEX_SIZE: u64 = 2 * size_of::<f32>() as u64;

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
struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,

    global_bind_group_layout: wgpu::BindGroupLayout,
    single_qubit_bind_group_layout: wgpu::BindGroupLayout,
    apply_gates_bind_group_layout: wgpu::BindGroupLayout,
    bring_into_rref_bind_group_layout: wgpu::BindGroupLayout,
    compute_output_bind_group_layout: wgpu::BindGroupLayout,

    zero_pipeline: wgpu::ComputePipeline,
    apply_gates_pipeline: wgpu::ComputePipeline,
    apply_t_gate_parallel_pipeline: wgpu::ComputePipeline,
    apply_tdg_gate_parallel_pipeline: wgpu::ComputePipeline,
    elimination_pass_pipeline: wgpu::ComputePipeline,
    swap_pass_pipeline: wgpu::ComputePipeline,
    update_before_h_pipeline: wgpu::ComputePipeline,
    compute_output_pipeline: wgpu::ComputePipeline,
}
impl GpuContext {
    async fn new() -> GpuContext {
        let instance = wgpu::Instance::new(&Default::default());
        let adapter = instance.request_adapter(&Default::default()).await.unwrap();
        let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();
        println!("Using WGPU adapter: {:?}", adapter.get_info());

        let shader_module = device.create_shader_module(wgpu::include_wgsl!("simulate_gpu.wgsl"));

        let global_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Global"),
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
                    // path
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
                    // active_batches
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
                    // seen_t_gates
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // ws
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // w_coeffs
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
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
        let single_qubit_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Single Qubit"),
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
        let compute_output_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Compute Output"),
                entries: &[
                    // w_target
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
                    // output
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

        let zero_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Zero Tableau"),
            bind_group_layouts: &[&global_bind_group_layout],
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
                bind_group_layouts: &[&global_bind_group_layout, &apply_gates_bind_group_layout],
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

        let apply_gate_parallel_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Apply T(dg) Gate in Parallel"),
                bind_group_layouts: &[&global_bind_group_layout, &single_qubit_bind_group_layout],
                immediate_size: 0,
            });
        let apply_t_gate_parallel_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Apply T Gate in Parallel"),
                layout: Some(&apply_gate_parallel_pipeline_layout),
                module: &shader_module,
                entry_point: Some("apply_t_gate_parallel"),
                cache: None,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });
        let apply_tdg_gate_parallel_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Apply Tdg Gate in Parallel"),
                layout: Some(&apply_gate_parallel_pipeline_layout),
                module: &shader_module,
                entry_point: Some("apply_tdg_gate_parallel"),
                cache: None,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

        let bring_into_rref_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Bring Into RREF"),
                bind_group_layouts: &[
                    &global_bind_group_layout,
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

        let update_before_h_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Update Before H"),
                bind_group_layouts: &[&global_bind_group_layout, &single_qubit_bind_group_layout],
                immediate_size: 0,
            });
        let update_before_h_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Update Before H"),
                layout: Some(&update_before_h_pipeline_layout),
                module: &shader_module,
                entry_point: Some("update_before_h"),
                cache: None,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

        let compute_output_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Compute Output"),
                bind_group_layouts: &[&global_bind_group_layout, &compute_output_bind_group_layout],
                immediate_size: 0,
            });
        let compute_output_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Compute Output"),
                layout: Some(&compute_output_pipeline_layout),
                module: &shader_module,
                entry_point: Some("compute_output"),
                cache: None,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

        GpuContext {
            device,
            queue,

            global_bind_group_layout,
            apply_gates_bind_group_layout,
            single_qubit_bind_group_layout,
            bring_into_rref_bind_group_layout,
            compute_output_bind_group_layout,

            zero_pipeline,
            apply_gates_pipeline,
            apply_t_gate_parallel_pipeline,
            apply_tdg_gate_parallel_pipeline,
            elimination_pass_pipeline,
            swap_pass_pipeline,
            update_before_h_pipeline,
            compute_output_pipeline,
        }
    }
}

pub struct GpuSimulator<'a> {
    circuit: &'a CliffordTCircuit,

    gpu: &'static GpuContext,
    encoder: wgpu::CommandEncoder,

    gates: Vec<u32>,
    qubit_params: Vec<u32>,

    n: u32,
    max_batches: u32,
    active_batches: u32,
    seen_t_gates: u32,

    n_buf: wgpu::Buffer,
    max_batches_buf: wgpu::Buffer,
    path_buf: wgpu::Buffer,
    tableau_buf: wgpu::Buffer,
    active_batches_buf: wgpu::Buffer,
    seen_t_gates_buf: wgpu::Buffer,
    ws_buf: wgpu::Buffer,
    w_coeffs_buf: wgpu::Buffer,
    global_bind_group: wgpu::BindGroup,

    col_bufs: [wgpu::Buffer; 2],
    a_bufs: [wgpu::Buffer; 2],
    pivot_buf: wgpu::Buffer,

    output_buf: wgpu::Buffer,
    output_read_buf: wgpu::Buffer,
    compute_output_bind_group: wgpu::BindGroup,
}
impl<'a> GpuSimulator<'a> {
    pub fn new(circuit: &'a CliffordTCircuit, w: &[bool], batch_size_log2: usize) -> Self {
        let gpu = get_gpu();

        let n = circuit.qubits();
        let path_length = circuit.t_gates() - batch_size_log2;

        let n: u32 = n.try_into().expect("n does not fit into u32");
        let max_batches: u32 = 1 << batch_size_log2;
        let active_batches = 1;
        let seen_t_gates = 0;

        let encoder = gpu.device.create_command_encoder(&Default::default());

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
        let path_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("path"),
            // Buffers must not be zero-sized, so we allocate at least 1 block even if path_length is 0.
            size: U32_SIZE * max(1, path_length as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let tableau_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tableau"),
            size: tableau_block_length(n, max_batches) as u64 * BLOCK_SIZE_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let active_batches_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("active_batches"),
            size: U32_SIZE,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let seen_t_gates_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("seen_t_gates"),
            size: U32_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ws_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ws"),
            size: U32_SIZE * n as u64 * max_batches as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let w_coeffs_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("w_coeffs"),
            size: COMPLEX_SIZE * max_batches as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let global_bind_group = Self::new_global_bind_group(
            gpu,
            &n_buf,
            &max_batches_buf,
            &path_buf,
            &tableau_buf,
            &active_batches_buf,
            &seen_t_gates_buf,
            &ws_buf,
            &w_coeffs_buf,
        );

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

        let w_target_buf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("w_target"),
                contents: &encode_bitstring(w),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let output_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: COMPLEX_SIZE * max_batches as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let output_read_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output (Read)"),
            size: COMPLEX_SIZE * max_batches as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let compute_output_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Output"),
            layout: &gpu.compute_output_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: w_target_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buf.as_entire_binding(),
                },
            ],
        });

        Self {
            circuit,

            gpu,
            encoder,

            gates: Vec::new(),
            qubit_params: Vec::new(),

            n,
            max_batches,
            active_batches,
            seen_t_gates,

            n_buf,
            max_batches_buf,
            path_buf,
            tableau_buf,
            active_batches_buf,
            seen_t_gates_buf,
            ws_buf,
            w_coeffs_buf,
            global_bind_group,

            col_bufs,
            a_bufs,
            pivot_buf,

            output_buf,
            output_read_buf,
            compute_output_bind_group,
        }
    }

    pub fn run(&mut self, path: &[bool]) -> Complex<f64> {
        let t_gates = self.circuit.t_gates() as u32;
        let path_length = t_gates - self.max_batches.ilog2();
        self.gpu
            .queue
            .write_buffer(&self.path_buf, 0, &encode_bitstring(path));
        self.zero();
        for &gate in self.circuit.gates() {
            match gate {
                CliffordTGate::Cnot(a, b) => self.apply_cnot_gate(a, b),
                CliffordTGate::Cz(a, b) => self.apply_cz_gate(a, b),
                CliffordTGate::X(a) => self.apply_x_gate(a),
                CliffordTGate::Y(a) => self.apply_y_gate(a),
                CliffordTGate::Z(a) => self.apply_z_gate(a),
                CliffordTGate::S(a) => self.apply_s_gate(a),
                CliffordTGate::Sdg(a) => self.apply_sdg_gate(a),
                CliffordTGate::H(a) => {
                    self.submit_gates();
                    self.bring_into_rref();
                    self.update_before_h(a);
                    self.apply_h_gate(a);
                }
                CliffordTGate::T(a) => {
                    if self.seen_t_gates < path_length {
                        self.apply_t_gate_branch(a);
                    } else {
                        self.submit_gates();
                        self.apply_t_gate_parallel(a);
                    }
                }
                CliffordTGate::Tdg(a) => {
                    if self.seen_t_gates < path_length {
                        self.apply_tdg_gate_branch(a);
                    } else {
                        self.submit_gates();
                        self.apply_tdg_gate_parallel(a);
                    }
                }
            }
        }
        self.submit_gates();
        self.bring_into_rref();
        self.compute_output();

        let new_encoder = self.gpu.device.create_command_encoder(&Default::default());
        let encoder = mem::replace(&mut self.encoder, new_encoder);
        self.gpu.queue.submit([encoder.finish()]);

        self.output_read_buf
            .map_async(wgpu::MapMode::Read, .., |_| {});
        self.gpu
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();

        let res = {
            let output_data = &self.output_read_buf.get_mapped_range(..);
            bytes_to_complex(output_data).sum()
        };
        self.output_read_buf.unmap();

        res
    }

    fn new_global_bind_group(
        gpu: &GpuContext,
        n_buf: &wgpu::Buffer,
        max_batches_buf: &wgpu::Buffer,
        path_buf: &wgpu::Buffer,
        tableau_buf: &wgpu::Buffer,
        active_batches_buf: &wgpu::Buffer,
        seen_t_gates_buf: &wgpu::Buffer,
        ws_buf: &wgpu::Buffer,
        w_coeffs_buf: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tableau"),
            layout: &gpu.global_bind_group_layout,
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
                    resource: path_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: tableau_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: active_batches_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: seen_t_gates_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: ws_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: w_coeffs_buf.as_entire_binding(),
                },
            ],
        })
    }

    fn zero(&mut self) {
        let n = self.n;

        self.encoder.clear_buffer(&self.tableau_buf, 0, None);
        self.encoder.clear_buffer(&self.seen_t_gates_buf, 0, None);

        self.active_batches = 1;
        self.seen_t_gates = 0;

        let workgroups = single_column_block_length(n).div_ceil(WORKGROUP_SIZE);
        let mut pass = self.encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.gpu.zero_pipeline);
        pass.set_bind_group(0, &self.global_bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
        drop(pass);
    }

    fn apply_cnot_gate(&mut self, a: usize, b: usize) {
        let a: u32 = a.try_into().expect("a does not fit into u32");
        let b: u32 = b.try_into().expect("b does not fit into u32");
        self.gates.push(0);
        self.qubit_params.push(a);
        self.qubit_params.push(b);
    }
    fn apply_cz_gate(&mut self, a: usize, b: usize) {
        let a: u32 = a.try_into().expect("a does not fit into u32");
        let b: u32 = b.try_into().expect("b does not fit into u32");
        self.gates.push(1);
        self.qubit_params.push(a);
        self.qubit_params.push(b);
    }
    fn apply_h_gate(&mut self, a: usize) {
        let a: u32 = a.try_into().expect("a does not fit into u32");
        self.gates.push(2);
        self.qubit_params.push(a);
    }
    fn apply_s_gate(&mut self, a: usize) {
        let a: u32 = a.try_into().expect("a does not fit into u32");
        self.gates.push(3);
        self.qubit_params.push(a);
    }
    fn apply_sdg_gate(&mut self, a: usize) {
        let a: u32 = a.try_into().expect("a does not fit into u32");
        self.gates.push(4);
        self.qubit_params.push(a);
    }
    fn apply_x_gate(&mut self, a: usize) {
        let a: u32 = a.try_into().expect("a does not fit into u32");
        self.gates.push(5);
        self.qubit_params.push(a);
    }
    fn apply_y_gate(&mut self, a: usize) {
        let a: u32 = a.try_into().expect("a does not fit into u32");
        self.gates.push(6);
        self.qubit_params.push(a);
    }
    fn apply_z_gate(&mut self, a: usize) {
        let a: u32 = a.try_into().expect("a does not fit into u32");
        self.gates.push(7);
        self.qubit_params.push(a);
    }
    fn apply_t_gate_branch(&mut self, a: usize) {
        self.apply_gate_branch(8, a);
    }
    fn apply_tdg_gate_branch(&mut self, a: usize) {
        self.apply_gate_branch(9, a);
    }
    fn apply_gate_branch(&mut self, gate_encoding: u32, a: usize) {
        let a: u32 = a.try_into().expect("a does not fit into u32");
        self.gates.push(gate_encoding);
        self.qubit_params.push(a);

        self.submit_gates();
        self.seen_t_gates += 1;
        self.seen_t_gates_buf =
            self.gpu
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("seen_t_gates"),
                    contents: &self.seen_t_gates.to_ne_bytes(),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });
        self.global_bind_group = Self::new_global_bind_group(
            &self.gpu,
            &self.n_buf,
            &self.max_batches_buf,
            &self.path_buf,
            &self.tableau_buf,
            &self.active_batches_buf,
            &self.seen_t_gates_buf,
            &self.ws_buf,
            &self.w_coeffs_buf,
        );
    }
    fn submit_gates(&mut self) {
        if self.gates.is_empty() {
            return;
        }

        let gates_buf = self
            .gpu
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
        let qubit_params_buf =
            self.gpu
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
        let bind_group = self
            .gpu
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Apply Gates Bind Group"),
                layout: &self.gpu.apply_gates_bind_group_layout,
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

        let workgroups = self.active_column_block_length().div_ceil(WORKGROUP_SIZE);
        let mut pass = self.encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.gpu.apply_gates_pipeline);
        pass.set_bind_group(0, &self.global_bind_group, &[]);
        pass.set_bind_group(1, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);

        self.gates.clear();
        self.qubit_params.clear();
    }

    fn apply_t_gate_parallel(&mut self, a: usize) {
        self.apply_gate_parallel(a, &self.gpu.apply_t_gate_parallel_pipeline);
    }
    fn apply_tdg_gate_parallel(&mut self, a: usize) {
        self.apply_gate_parallel(a, &self.gpu.apply_tdg_gate_parallel_pipeline);
    }
    fn apply_gate_parallel(&mut self, a: usize, pipeline: &wgpu::ComputePipeline) {
        let a: u32 = a.try_into().expect("a does not fit into u32");
        debug_assert!(
            self.active_batches * 2 <= self.max_batches,
            "Cannot split batches beyond max_batches"
        );

        let bind_group = self
            .gpu
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Apply T Gate Parallel Bind Group"),
                layout: &self.gpu.single_qubit_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self
                        .gpu
                        .device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("qubit"),
                            contents: &a.to_ne_bytes(),
                            usage: wgpu::BufferUsages::UNIFORM,
                        })
                        .as_entire_binding(),
                }],
            });

        let workgroups = self.active_column_block_length().div_ceil(WORKGROUP_SIZE);
        let mut pass = self.encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.global_bind_group, &[]);
        pass.set_bind_group(1, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);

        self.active_batches *= 2;
        self.active_batches_buf =
            self.gpu
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("active_batches"),
                    contents: &self.active_batches.to_ne_bytes(),
                    usage: wgpu::BufferUsages::STORAGE,
                });
        self.global_bind_group = Self::new_global_bind_group(
            &self.gpu,
            &self.n_buf,
            &self.max_batches_buf,
            &self.path_buf,
            &self.tableau_buf,
            &self.active_batches_buf,
            &self.seen_t_gates_buf,
            &self.ws_buf,
            &self.w_coeffs_buf,
        )
    }

    fn bring_into_rref(&mut self) {
        let n = self.n;
        let active_batches = self.active_batches;

        let mut col_in_buf = &self.col_bufs[0];
        let mut col_out_buf = &self.col_bufs[1];
        let mut a_in_buf = &self.a_bufs[0];
        let mut a_out_buf = &self.a_bufs[1];

        self.encoder.clear_buffer(col_in_buf, 0, None);
        self.encoder.clear_buffer(a_in_buf, 0, None);

        for _ in 0..n {
            let bind_group = self
                .gpu
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Bring-into-RREF Pass Bind Group"),
                    layout: &self.gpu.bring_into_rref_bind_group_layout,
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

            let workgroups = self.active_column_block_length().div_ceil(WORKGROUP_SIZE);
            let mut elimination_pass = self.encoder.begin_compute_pass(&Default::default());
            elimination_pass.set_pipeline(&self.gpu.elimination_pass_pipeline);
            elimination_pass.set_bind_group(0, &self.global_bind_group, &[]);
            elimination_pass.set_bind_group(1, &bind_group, &[]);
            elimination_pass.dispatch_workgroups(workgroups, 1, 1);
            drop(elimination_pass);

            let workgroups = active_batches.div_ceil(WORKGROUP_SIZE);
            let mut swap_pass = self.encoder.begin_compute_pass(&Default::default());
            swap_pass.set_pipeline(&self.gpu.swap_pass_pipeline);
            swap_pass.set_bind_group(0, &self.global_bind_group, &[]);
            swap_pass.set_bind_group(1, &bind_group, &[]);
            swap_pass.dispatch_workgroups(workgroups, 1, 1);
            drop(swap_pass);

            (col_in_buf, col_out_buf) = (col_out_buf, col_in_buf);
            (a_in_buf, a_out_buf) = (a_out_buf, a_in_buf);
        }
    }

    fn update_before_h(&mut self, a: usize) {
        let a: u32 = a.try_into().expect("a does not fit into u32");

        let bind_group = self
            .gpu
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Update Before H Bind Group"),
                layout: &self.gpu.single_qubit_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self
                        .gpu
                        .device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("qubit"),
                            contents: &a.to_ne_bytes(),
                            usage: wgpu::BufferUsages::UNIFORM,
                        })
                        .as_entire_binding(),
                }],
            });

        let workgroups = self.active_batches.div_ceil(WORKGROUP_SIZE);
        let mut pass = self.encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.gpu.update_before_h_pipeline);
        pass.set_bind_group(0, &self.global_bind_group, &[]);
        pass.set_bind_group(1, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }

    fn compute_output(&mut self) {
        let workgroups = self.active_batches.div_ceil(WORKGROUP_SIZE);
        let mut pass = self.encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.gpu.compute_output_pipeline);
        pass.set_bind_group(0, &self.global_bind_group, &[]);
        pass.set_bind_group(1, &self.compute_output_bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
        drop(pass);

        self.encoder
            .copy_buffer_to_buffer(&self.output_buf, 0, &self.output_read_buf, 0, None);
    }

    fn active_column_block_length(&self) -> u32 {
        single_column_block_length(self.n) * self.active_batches
    }
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

fn encode_bitstring(bits: &[bool]) -> Vec<u8> {
    bits.iter()
        .map(|&b| if b { 1u32 } else { 0u32 })
        .flat_map(|b| b.to_ne_bytes())
        .collect()
}

/// Convert the contents of a slice of bytes into an iterator over Complex<f32> using native endianness.
/// Note that the returned values are additionally converted to Complex<f64> for convenience, but the underlying data is still f32 precision.
fn bytes_to_complex(bytes: &[u8]) -> impl Iterator<Item = Complex<f64>> {
    let (chunks, remainder) = bytes.as_chunks::<4>();
    debug_assert_eq!(remainder.len(), 0);
    let (chunks, remainder) = chunks.as_chunks::<2>();
    debug_assert_eq!(remainder.len(), 0);
    chunks.iter().copied().map(|[re_b, im_b]| {
        Complex::new(
            f32::from_ne_bytes(re_b).into(),
            f32::from_ne_bytes(im_b).into(),
        )
    })
}
