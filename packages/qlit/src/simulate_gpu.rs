use std::{
    cmp::max,
    mem,
    sync::{
        Arc, OnceLock,
        atomic::{self, AtomicBool},
    },
    thread,
};

use num_complex::Complex;
use wgpu::util::DeviceExt;

use crate::{
    BitString,
    decompose::{CliffordPCircuit, CliffordPGate},
};

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
    fork_apply_piz_gate_bind_group_layout: wgpu::BindGroupLayout,
    apply_gates_bind_group_layout: wgpu::BindGroupLayout,
    bring_into_rref_bind_group_layout: wgpu::BindGroupLayout,
    compute_output_bind_group_layout: wgpu::BindGroupLayout,

    zero_pipeline: wgpu::ComputePipeline,
    apply_gates_pipeline: wgpu::ComputePipeline,
    fork_apply_piz_gate_pipeline: wgpu::ComputePipeline,
    elimination_pass_pipeline: wgpu::ComputePipeline,
    swap_pass_pipeline: wgpu::ComputePipeline,
    update_before_h_pipeline: wgpu::ComputePipeline,
    compute_output_pipeline: wgpu::ComputePipeline,
}
impl GpuContext {
    async fn new() -> GpuContext {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance.request_adapter(&Default::default()).await.unwrap();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: Default::default(),
                required_limits: adapter.limits(),
                experimental_features: Default::default(),
                memory_hints: Default::default(),
                trace: Default::default(),
            })
            .await
            .unwrap();

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
                    // c_cols
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
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
                        binding: 5,
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
                        binding: 6,
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
        let fork_apply_piz_gate_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Fork Apply Piz Gate"),
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
                    // ci
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
                    // cz
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
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
                    // ci_params
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
                    // cz_params
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // initial_seen_p_gates
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
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
                    // input
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
                    // input_next
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
                    // pivot_out
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
            bind_group_layouts: &[Some(&global_bind_group_layout)],
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
                bind_group_layouts: &[
                    Some(&global_bind_group_layout),
                    Some(&apply_gates_bind_group_layout),
                ],
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

        let fork_apply_piz_gate_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Fork Apply Piz Gate"),
                bind_group_layouts: &[
                    Some(&global_bind_group_layout),
                    Some(&fork_apply_piz_gate_bind_group_layout),
                ],
                immediate_size: 0,
            });
        let fork_apply_piz_gate_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Fork Apply Piz Gate"),
                layout: Some(&fork_apply_piz_gate_pipeline_layout),
                module: &shader_module,
                entry_point: Some("fork_apply_piz_gate"),
                cache: None,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

        let bring_into_rref_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Bring Into RREF"),
                bind_group_layouts: &[
                    Some(&global_bind_group_layout),
                    Some(&bring_into_rref_bind_group_layout),
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
                bind_group_layouts: &[
                    Some(&global_bind_group_layout),
                    Some(&single_qubit_bind_group_layout),
                ],
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
                bind_group_layouts: &[
                    Some(&global_bind_group_layout),
                    Some(&compute_output_bind_group_layout),
                ],
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
            single_qubit_bind_group_layout,
            fork_apply_piz_gate_bind_group_layout,
            apply_gates_bind_group_layout,
            bring_into_rref_bind_group_layout,
            compute_output_bind_group_layout,

            zero_pipeline,
            apply_gates_pipeline,
            fork_apply_piz_gate_pipeline,
            elimination_pass_pipeline,
            swap_pass_pipeline,
            update_before_h_pipeline,
            compute_output_pipeline,
        }
    }
}

pub struct GpuSimulator<'a> {
    circuit: &'a CliffordPCircuit,

    gpu: &'static GpuContext,
    encoder: wgpu::CommandEncoder,

    n: u32,
    max_c_cols: u32,
    c_cols: u32,
    path_length: u32,

    n_buf: wgpu::Buffer,
    max_batches_buf: wgpu::Buffer,
    path_buf: wgpu::Buffer,
    tableau_buf: wgpu::Buffer,
    c_cols_buf: wgpu::Buffer,
    seen_p_gates_buf: wgpu::Buffer,
    ws_buf: wgpu::Buffer,
    w_coeffs_buf: wgpu::Buffer,
    global_bind_group: wgpu::BindGroup,

    apply_gates_bind_group_index: usize,
    apply_gates_bind_groups: Vec<Option<wgpu::BindGroup>>,

    rref_input_bufs: [wgpu::Buffer; 2],
    pivot_buf: wgpu::Buffer,

    output_buf: wgpu::Buffer,
    output_read_buf: wgpu::Buffer,
    compute_output_bind_group: wgpu::BindGroup,
}
impl<'a> GpuSimulator<'a> {
    pub fn new(circuit: &'a CliffordPCircuit, w: &BitString, batch_size_log2: usize) -> Self {
        let gpu = get_gpu();

        let n: u32 = circuit.qubits();
        let path_length: u32 = (circuit.p_gates() - batch_size_log2)
            .try_into()
            .expect("Number of T gates exceeds maximum supported by GPU simulator");
        debug_assert!(
            batch_size_log2 < 32,
            "batch_size_log2 must be strictly less than 32 to avoid overflow in max_batches"
        );
        let max_c_cols: u32 = batch_size_log2
            .try_into()
            .expect("batch_size_log2 exceeds maximum supported by GPU simulator");
        let max_batches: u32 = 1 << max_c_cols;
        let max_batches_u64: u64 = max_batches.into();
        let c_cols: u32 = 0;

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
            size: U32_SIZE * max(1, path_length.into()),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let tableau_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tableau"),
            size: u64::from(tableau_block_length(n, max_c_cols)) * BLOCK_SIZE_BYTES,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let c_cols_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("c_cols"),
            size: U32_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let seen_p_gates_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("seen_p_gates"),
            size: U32_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ws_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ws"),
            size: U32_SIZE * u64::from(n) * max_batches_u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let w_coeffs_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("w_coeffs"),
            size: COMPLEX_SIZE * max_batches_u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let global_bind_group = Self::new_global_bind_group(
            gpu,
            &n_buf,
            &max_batches_buf,
            &path_buf,
            &tableau_buf,
            &c_cols_buf,
            &ws_buf,
            &w_coeffs_buf,
        );

        let apply_gates_bind_groups = Self::apply_gates_bind_groups(gpu, circuit, path_length);

        let rref_input_bufs = [
            gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("RREF Pass Input 1"),
                size: 2 * U32_SIZE,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("RREF Pass Input 2"),
                size: 2 * U32_SIZE,
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
                contents: &encode_bitstring(w.iter()),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let output_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: COMPLEX_SIZE * max_batches_u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let output_read_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output (Read)"),
            size: COMPLEX_SIZE * max_batches_u64,
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

            n,
            max_c_cols,
            c_cols,
            path_length,

            n_buf,
            max_batches_buf,
            path_buf,
            tableau_buf,
            c_cols_buf,
            seen_p_gates_buf,
            ws_buf,
            w_coeffs_buf,
            global_bind_group,

            apply_gates_bind_group_index: 0,
            apply_gates_bind_groups,

            rref_input_bufs,
            pivot_buf,

            output_buf,
            output_read_buf,
            compute_output_bind_group,
        }
    }

    fn new_global_bind_group(
        gpu: &GpuContext,
        n_buf: &wgpu::Buffer,
        max_batches_buf: &wgpu::Buffer,
        path_buf: &wgpu::Buffer,
        tableau_buf: &wgpu::Buffer,
        c_cols_buf: &wgpu::Buffer,
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
                    resource: c_cols_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: ws_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: w_coeffs_buf.as_entire_binding(),
                },
            ],
        })
    }

    fn apply_gates_bind_groups(
        gpu: &GpuContext,
        circuit: &CliffordPCircuit,
        path_length: u32,
    ) -> Vec<Option<wgpu::BindGroup>> {
        let mut bind_groups = Vec::new();

        let mut gates = Vec::new();
        let mut qubit_params = Vec::new();

        // The coefficients of every P gate on the path, indexed by seen_p_gates in the shader.
        let (mut ci_params, mut cz_params): (Vec<_>, Vec<_>) = circuit
            .gates()
            .iter()
            .filter_map(|&gate| match gate {
                CliffordPGate::Piz(ci, cz, _) => Some((ci, cz)),
                _ => None,
            })
            .take(path_length as usize)
            .unzip();
        // Buffers must not be zero-sized, so we allocate at least 1 element even if path_length is 0.
        if ci_params.is_empty() {
            ci_params.push(Complex::ZERO);
            cz_params.push(Complex::ZERO);
        }
        let ci_params_buf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ci_params"),
                contents: &encode_complex(ci_params),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let cz_params_buf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cz_params"),
                contents: &encode_complex(cz_params),
                usage: wgpu::BufferUsages::STORAGE,
            });

        fn commit_buffer(
            gpu: &GpuContext,
            gates: &mut Vec<u32>,
            qubit_params: &mut Vec<u32>,
            ci_params_buf: &wgpu::Buffer,
            cz_params_buf: &wgpu::Buffer,
            initial_seen_p_gates: u32,
            bind_groups: &mut Vec<Option<wgpu::BindGroup>>,
        ) {
            if gates.is_empty() {
                bind_groups.push(None);
                return;
            }

            let gates_buf = gpu
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("gates"),
                    contents: &gates
                        .iter()
                        .flat_map(|&g| g.to_ne_bytes())
                        .collect::<Vec<u8>>(),
                    usage: wgpu::BufferUsages::STORAGE,
                });
            let qubit_params_buf =
                gpu.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("qubit_params"),
                        contents: &qubit_params
                            .iter()
                            .flat_map(|&q| q.to_ne_bytes())
                            .collect::<Vec<u8>>(),
                        usage: wgpu::BufferUsages::STORAGE,
                    });
            let initial_seen_p_gates_buf =
                gpu.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("initial_seen_p_gates"),
                        contents: &initial_seen_p_gates.to_ne_bytes(),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });
            let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Apply Gates"),
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
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: ci_params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: cz_params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: initial_seen_p_gates_buf.as_entire_binding(),
                    },
                ],
            });
            bind_groups.push(Some(bind_group));
            gates.clear();
            qubit_params.clear();
        }

        let mut initial_seen_p_gates = 0;
        let mut seen_p_gates: u32 = 0;
        for &gate in circuit.gates() {
            match gate {
                CliffordPGate::Cnot(a, b) => {
                    gates.push(0);
                    qubit_params.push(a);
                    qubit_params.push(b);
                }
                CliffordPGate::Cz(a, b) => {
                    gates.push(1);
                    qubit_params.push(a);
                    qubit_params.push(b);
                }
                CliffordPGate::X(a) => {
                    gates.push(2);
                    qubit_params.push(a);
                }
                CliffordPGate::Y(a) => {
                    gates.push(3);
                    qubit_params.push(a);
                }
                CliffordPGate::Z(a) => {
                    gates.push(4);
                    qubit_params.push(a);
                }
                CliffordPGate::S(a) => {
                    gates.push(5);
                    qubit_params.push(a);
                }
                CliffordPGate::Sdg(a) => {
                    gates.push(6);
                    qubit_params.push(a);
                }
                CliffordPGate::H(a) => {
                    commit_buffer(
                        gpu,
                        &mut gates,
                        &mut qubit_params,
                        &ci_params_buf,
                        &cz_params_buf,
                        initial_seen_p_gates,
                        &mut bind_groups,
                    );
                    initial_seen_p_gates = seen_p_gates;

                    gates.push(7);
                    qubit_params.push(a);
                }
                CliffordPGate::Piz(_, _, a) => {
                    if seen_p_gates < path_length {
                        gates.push(8);
                        qubit_params.push(a);
                        seen_p_gates += 1;
                    } else {
                        commit_buffer(
                            gpu,
                            &mut gates,
                            &mut qubit_params,
                            &ci_params_buf,
                            &cz_params_buf,
                            initial_seen_p_gates,
                            &mut bind_groups,
                        );
                        initial_seen_p_gates = seen_p_gates;
                    }
                }
            }
        }
        commit_buffer(
            gpu,
            &mut gates,
            &mut qubit_params,
            &ci_params_buf,
            &cz_params_buf,
            initial_seen_p_gates,
            &mut bind_groups,
        );
        bind_groups
    }

    pub fn run(&mut self, path: &[bool]) -> Complex<f64> {
        self.gpu
            .queue
            .write_buffer(&self.path_buf, 0, &encode_bitstring(path.iter().copied()));
        self.zero();
        let mut seen_p_gates = 0;
        for &gate in self.circuit.gates() {
            match gate {
                CliffordPGate::H(a) => {
                    self.dispatch_gates();
                    self.bring_into_rref();
                    self.update_before_h(a);
                    // This is the single heaviest operation for the circuit,
                    // therefore we perform a submission of the full pipeline up to this point before moving on,
                    // to get the GPU started as soon as possible and avoid a single huge submission timing out.
                    self.submit_encoder();
                }
                CliffordPGate::Piz(ci, cz, a) => {
                    if seen_p_gates < self.path_length {
                        seen_p_gates += 1;
                    } else {
                        self.dispatch_gates();
                        self.fork_apply_piz_gate(ci, cz, a);
                    }
                }
                _ => {}
            }
        }
        self.dispatch_gates();
        self.bring_into_rref();
        self.compute_output();
        self.submit_encoder();

        let done1 = Arc::new(AtomicBool::new(false));
        let done2 = Arc::clone(&done1);
        let handle = thread::current();

        self.output_read_buf
            .map_async(wgpu::MapMode::Read, .., move |r| {
                r.expect("Failed to get output from GPU");

                done1.store(true, atomic::Ordering::Relaxed);
                handle.unpark();
            });
        self.gpu
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();

        while !done2.load(atomic::Ordering::Acquire) {
            thread::park();
        }

        let res = {
            let output_data = &self
                .output_read_buf
                .get_mapped_range(..)
                .expect("failed to load output from GPU");
            bytes_to_complex(output_data).sum()
        };
        self.output_read_buf.unmap();

        res
    }

    fn zero(&mut self) {
        let n = self.n;

        self.encoder.clear_buffer(&self.tableau_buf, 0, None);
        self.encoder.clear_buffer(&self.seen_p_gates_buf, 0, None);
        self.encoder.clear_buffer(&self.ws_buf, 0, None);
        self.encoder.clear_buffer(&self.c_cols_buf, 0, None);

        self.c_cols = 0;
        self.apply_gates_bind_group_index = 0;

        let workgroups = column_block_length(n).div_ceil(WORKGROUP_SIZE);
        let mut pass = self.encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.gpu.zero_pipeline);
        pass.set_bind_group(0, &self.global_bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
        drop(pass);
    }

    fn dispatch_gates(&mut self) {
        match &self.apply_gates_bind_groups[self.apply_gates_bind_group_index] {
            Some(bind_group) => {
                let workgroups = self.block_or_state_threads().div_ceil(WORKGROUP_SIZE);
                let mut pass = self.encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.gpu.apply_gates_pipeline);
                pass.set_bind_group(0, &self.global_bind_group, &[]);
                pass.set_bind_group(1, bind_group, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
            None => {
                // No gates to apply - dispatching an empty buffer will fail.
            }
        }
        self.apply_gates_bind_group_index += 1;
    }

    /// Submits everything in the current command encoder to the GPU and resets the encoder.
    fn submit_encoder(&mut self) {
        let new_encoder = self.gpu.device.create_command_encoder(&Default::default());
        let encoder = mem::replace(&mut self.encoder, new_encoder);
        self.gpu.queue.submit([encoder.finish()]);
    }

    fn fork_apply_piz_gate(&mut self, ci: Complex<f64>, cz: Complex<f64>, a: u32) {
        debug_assert!(
            self.c_cols < self.max_c_cols,
            "Cannot fork beyond max_c_cols"
        );

        let bind_group = self
            .gpu
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Fork Apply Piz Gate Bind Group"),
                layout: &self.gpu.fork_apply_piz_gate_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
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
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self
                            .gpu
                            .device
                            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("ci"),
                                contents: &encode_complex([ci]),
                                usage: wgpu::BufferUsages::UNIFORM,
                            })
                            .as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self
                            .gpu
                            .device
                            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("cz"),
                                contents: &encode_complex([cz]),
                                usage: wgpu::BufferUsages::UNIFORM,
                            })
                            .as_entire_binding(),
                    },
                ],
            });

        let workgroups = self.block_or_state_threads().div_ceil(WORKGROUP_SIZE);
        let mut pass = self.encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.gpu.fork_apply_piz_gate_pipeline);
        pass.set_bind_group(0, &self.global_bind_group, &[]);
        pass.set_bind_group(1, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
        drop(pass);

        self.c_cols += 1;
        self.c_cols_buf = self
            .gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("c_cols"),
                contents: &self.c_cols.to_ne_bytes(),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        self.global_bind_group = Self::new_global_bind_group(
            &self.gpu,
            &self.n_buf,
            &self.max_batches_buf,
            &self.path_buf,
            &self.tableau_buf,
            &self.c_cols_buf,
            &self.ws_buf,
            &self.w_coeffs_buf,
        )
    }

    fn bring_into_rref(&mut self) {
        let n = self.n;
        let c_cols = self.c_cols;

        let mut input_buf = &self.rref_input_bufs[0];
        let mut input_next_buf = &self.rref_input_bufs[1];

        self.encoder.clear_buffer(input_buf, 0, None);

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
                            resource: input_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: input_next_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.pivot_buf.as_entire_binding(),
                        },
                    ],
                });

            let workgroups = column_block_length(n).div_ceil(WORKGROUP_SIZE);
            let mut elimination_pass = self.encoder.begin_compute_pass(&Default::default());
            elimination_pass.set_pipeline(&self.gpu.elimination_pass_pipeline);
            elimination_pass.set_bind_group(0, &self.global_bind_group, &[]);
            elimination_pass.set_bind_group(1, &bind_group, &[]);
            elimination_pass.dispatch_workgroups(workgroups, 1, 1);
            drop(elimination_pass);

            let workgroups = (n + n + 1 + c_cols).div_ceil(WORKGROUP_SIZE);
            let mut swap_pass = self.encoder.begin_compute_pass(&Default::default());
            swap_pass.set_pipeline(&self.gpu.swap_pass_pipeline);
            swap_pass.set_bind_group(0, &self.global_bind_group, &[]);
            swap_pass.set_bind_group(1, &bind_group, &[]);
            swap_pass.dispatch_workgroups(workgroups, 1, 1);
            drop(swap_pass);

            (input_buf, input_next_buf) = (input_next_buf, input_buf);
        }
    }

    fn update_before_h(&mut self, a: u32) {
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

        let workgroups = self.contained_states().div_ceil(WORKGROUP_SIZE);
        let mut pass = self.encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.gpu.update_before_h_pipeline);
        pass.set_bind_group(0, &self.global_bind_group, &[]);
        pass.set_bind_group(1, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }

    fn compute_output(&mut self) {
        let workgroups = self.contained_states().div_ceil(WORKGROUP_SIZE);
        let mut pass = self.encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.gpu.compute_output_pipeline);
        pass.set_bind_group(0, &self.global_bind_group, &[]);
        pass.set_bind_group(1, &self.compute_output_bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
        drop(pass);

        self.encoder
            .copy_buffer_to_buffer(&self.output_buf, 0, &self.output_read_buf, 0, None);
    }

    /// Get the number of states currently represented by the tableau.
    fn contained_states(&self) -> u32 {
        1 << self.c_cols
    }

    /// Get the number of threads needed by kernels that assign one thread to each row (block),
    /// as well as one thread to each contained state.
    fn block_or_state_threads(&self) -> u32 {
        max(column_block_length(self.n), self.contained_states())
    }
}

// Get the block-length of the columns in the tableau.
fn column_block_length(n: u32) -> u32 {
    // Make room for the auxiliary row.
    (n + 1).div_ceil(BLOCK_SIZE)
}
/// Get the length of the tableau in blocks, with room for `max_c_cols` c columns.
fn tableau_block_length(n: u32, max_c_cols: u32) -> u32 {
    column_block_length(n) * (n + n + 1 + max_c_cols)
}

fn encode_bitstring<T>(bits: T) -> Vec<u8>
where
    T: Iterator<Item = bool>,
{
    bits.map(|b| if b { 1u32 } else { 0u32 })
        .flat_map(|b| b.to_ne_bytes())
        .collect()
}

/// Encode complex values as a sequence of WGSL `Complex` structs (pairs of f32) using native endianness.
/// Note that the values are narrowed from f64 to f32 precision.
fn encode_complex<T>(values: T) -> Vec<u8>
where
    T: IntoIterator<Item = Complex<f64>>,
{
    values
        .into_iter()
        .flat_map(|c| [c.re as f32, c.im as f32])
        .flat_map(|x| x.to_ne_bytes())
        .collect()
}

// TEMPORARY: Diagnostics for the macos-15-intel CI failure.
#[cfg(test)]
impl GpuSimulator<'_> {
    /// Submit all pending work and read back the full contents of `buf`.
    fn read_back(&mut self, buf: &wgpu::Buffer) -> Vec<u8> {
        self.submit_encoder();
        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Read Back"),
            size: buf.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, None);
        self.gpu.queue.submit([encoder.finish()]);

        let done1 = Arc::new(AtomicBool::new(false));
        let done2 = Arc::clone(&done1);
        staging.map_async(wgpu::MapMode::Read, .., move |r| {
            r.expect("Failed to read back buffer");
            done1.store(true, atomic::Ordering::Release);
        });
        self.gpu
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        assert!(done2.load(atomic::Ordering::Acquire));

        let data = staging
            .get_mapped_range(..)
            .expect("failed to read back buffer")
            .to_vec();
        staging.unmap();
        data
    }

    /// Render the tableau, ws and w_coeffs of the currently contained states.
    fn dump(&mut self, label: &str) -> String {
        let n = self.n;
        let cbl = column_block_length(n);
        let cols = n + n + 1 + self.c_cols;

        let tableau_bytes = self.read_back(&self.tableau_buf.clone());
        let ws_bytes = self.read_back(&self.ws_buf.clone());
        let w_coeffs_bytes = self.read_back(&self.w_coeffs_buf.clone());
        let tableau: Vec<u32> = tableau_bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| u32::from_ne_bytes(b))
            .collect();
        let ws: Vec<u32> = ws_bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| u32::from_ne_bytes(b))
            .collect();
        let w_coeffs: Vec<Complex<f64>> = bytes_to_complex(&w_coeffs_bytes).collect();

        let bit = |row: u32, j: u32| {
            let block = tableau[(j * cbl + row / BLOCK_SIZE) as usize];
            (block >> (BLOCK_SIZE - 1 - row % BLOCK_SIZE)) & 1
        };
        let mut s = format!("[{label}] c_cols={}\n", self.c_cols);
        for row in 0..=n {
            let x: String = (0..n).map(|q| bit(row, 2 * q).to_string()).collect();
            let z: String = (0..n).map(|q| bit(row, 2 * q + 1).to_string()).collect();
            let r = bit(row, 2 * n);
            let c: String = (2 * n + 1..cols).map(|j| bit(row, j).to_string()).collect();
            s += &format!("  row {row:2}: x={x} z={z} r={r} c={c}\n");
        }
        for state in 0..self.contained_states() {
            let w: String = (0..n)
                .map(|q| ws[(state * n + q) as usize].to_string())
                .collect();
            let c = w_coeffs[state as usize];
            s += &format!(
                "  state {state:2}: ws={w} w_coeff={:+.9e}{:+.9e}i\n",
                c.re, c.im
            );
        }
        s
    }

    /// Same as `run`, but records the GPU state after every step.
    fn run_traced(&mut self, path: &[bool]) -> (Complex<f64>, Vec<String>) {
        let mut trace = Vec::new();
        self.gpu
            .queue
            .write_buffer(&self.path_buf, 0, &encode_bitstring(path.iter().copied()));
        self.zero();
        trace.push(self.dump("zero"));
        let mut seen_p_gates = 0;
        for &gate in self.circuit.gates() {
            match gate {
                CliffordPGate::H(a) => {
                    self.dispatch_gates();
                    trace.push(self.dump("apply_gates"));
                    self.bring_into_rref();
                    trace.push(self.dump("bring_into_rref"));
                    self.update_before_h(a);
                    trace.push(self.dump(&format!("update_before_h({a})")));
                }
                CliffordPGate::Piz(ci, cz, a) => {
                    if seen_p_gates < self.path_length {
                        seen_p_gates += 1;
                    } else {
                        self.dispatch_gates();
                        trace.push(self.dump("apply_gates"));
                        self.fork_apply_piz_gate(ci, cz, a);
                        trace.push(self.dump(&format!("fork_apply_piz_gate({a})")));
                    }
                }
                _ => {}
            }
        }
        self.dispatch_gates();
        trace.push(self.dump("apply_gates"));
        self.bring_into_rref();
        trace.push(self.dump("bring_into_rref"));
        self.compute_output();
        let output = self.read_back(&self.output_buf.clone());
        let outputs: Vec<_> = bytes_to_complex(&output)
            .take(self.contained_states() as usize)
            .collect();
        trace.push(format!("[compute_output] outputs={outputs:?}\n"));
        (outputs.iter().sum(), trace)
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use super::*;
    use crate::{Circuit, Gate::*, decompose::decompose};

    // TEMPORARY: Diagnostics for the macos-15-intel CI failure.
    // Run with `cargo test -p qlit trace_gpu_state -- --ignored --nocapture`.
    #[test]
    #[ignore = "diagnostic: prints a step-by-step trace of the GPU state"]
    fn trace_gpu_state() {
        let cases = [
            (
                "smaller_circuit_1",
                Circuit::new(8, [T(0), H(1), H(2), Cnot(1, 0), Cnot(2, 1)]).unwrap(),
                0b0110_0000,
            ),
            (
                "rotations_by_pi",
                Circuit::new(8, [Rx(PI, 0), Ry(PI, 1), H(2), Rz(PI, 2)]).unwrap(),
                0b0000_0000,
            ),
            (
                "larger_circuit_1",
                Circuit::new(
                    8,
                    [
                        T(0),
                        H(1),
                        S(1),
                        H(3),
                        H(0),
                        S(0),
                        S(1),
                        S(2),
                        T(1),
                        H(0),
                        Cnot(1, 0),
                        T(0),
                        S(3),
                    ],
                )
                .unwrap(),
                0b0100_0000,
            ),
        ];
        for (name, circuit, w) in cases {
            let circuit = decompose(&circuit);
            let w_bits = BitString::from_u8_ltr(w);
            // Tracing submits after every step, so also record an untraced run for comparison.
            let untraced = GpuSimulator::new(&circuit, &w_bits, circuit.p_gates()).run(&[]);
            let mut g = GpuSimulator::new(&circuit, &w_bits, circuit.p_gates());
            let (result, trace) = g.run_traced(&[]);
            println!("=== {name} w={w:08b} traced_result={result} untraced_result={untraced}");
            for step in trace {
                print!("{step}");
            }
        }
    }
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
