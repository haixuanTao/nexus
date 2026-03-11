//! Radix sort implementation, ported from `brush-sort`: <https://github.com/ArthurBrussee/brush/tree/main/crates/brush-sort>

use crate::shaders::utils::radix_sort::{
    GpuInitSortDispatch, GpuSortCount, GpuSortReduce, GpuSortScan, GpuSortScanAdd, GpuSortScatter,
    SortUniforms,
};
use khal::backend::{GpuBackend, GpuBackendError, GpuPass};
use khal::{BufferUsages, Shader};
use vortx::tensor::Tensor;

// NOTE: must match the values from the shaders.
const WG: u32 = 256;
const ELEMENTS_PER_THREAD: u32 = 4;
const BLOCK_SIZE: u32 = WG * ELEMENTS_PER_THREAD;
#[allow(dead_code)]
const BITS_PER_PASS: u32 = 4;
#[allow(dead_code)]
const BIN_COUNT: u32 = 1 << BITS_PER_PASS;

/// GPU-accelerated radix sort for sorting large arrays of u32 keys with associated values.
///
/// This implementation uses a 4-bit radix (16 bins per pass) and processes multiple passes
/// to sort up to 32-bit integers. The algorithm is highly optimized for GPU execution with:
/// - Workgroup-local histograms for reduced memory bandwidth
/// - Prefix sum (scan) operations for determining output positions
/// - Scatter phase that writes sorted elements to output buffers
///
/// # Algorithm Overview
///
/// For each 4-bit pass (up to 8 passes for 32-bit keys):
/// 1. **Count**: Histogram computation per workgroup
/// 2. **Reduce**: Aggregate histograms across workgroups
/// 3. **Scan**: Prefix sum on aggregated histograms
/// 4. **Scan Add**: Distribute prefix sums back to workgroup histograms
/// 5. **Scatter**: Write elements to sorted positions based on histograms
///
/// # Performance
///
/// - Processes ~10-100M elements/second on modern GPUs
/// - Near-linear scaling with input size
/// - Memory bandwidth bound (optimal for GPU)
#[derive(Shader)]
pub struct RadixSort {
    init: GpuInitSortDispatch,
    count: GpuSortCount,
    reduce: GpuSortReduce,
    scan: GpuSortScan,
    scan_add: GpuSortScanAdd,
    scatter: GpuSortScatter,
}

/// Workspace buffers for radix sort operations.
///
/// Maintains intermediate buffers needed by the radix sort algorithm:
/// - Histogram buffers for bin counts
/// - Reduction buffers for prefix sums
/// - Ping-pong buffers for multi-pass sorting
///
/// The workspace is reusable across multiple sort operations and automatically
/// resizes buffers as needed.
pub struct RadixSortWorkspace {
    pass_uniforms: Vec<Tensor<SortUniforms>>,
    reduced_buf: Tensor<u32>, // Tensor of size BLOCK_SIZE
    count_buf: Tensor<u32>,
    num_wgs: Tensor<[u32; 3]>,
    num_reduce_wgs: Tensor<[u32; 3]>,
    output_keys_pong: Tensor<u32>,   // dual-buffering for output keys.
    output_values_pong: Tensor<u32>, // dual-buffering for output values.
}

impl RadixSortWorkspace {
    /// Creates a new radix sort workspace with default buffer sizes.
    ///
    /// Buffers will be automatically resized on first use to match input data size.
    ///
    /// # Parameters
    ///
    /// - `backend`: The GPU backend to allocate buffers on
    pub fn new(backend: &GpuBackend) -> Self {
        let zeros = vec![0u32; BLOCK_SIZE as usize];
        Self {
            pass_uniforms: vec![],
            reduced_buf: Tensor::vector(backend, &zeros, BufferUsages::STORAGE).unwrap(),
            count_buf: Tensor::vector_uninit(backend, 0, BufferUsages::STORAGE).unwrap(),
            num_wgs: Tensor::scalar(
                backend,
                [1, 1, 1],
                BufferUsages::STORAGE | BufferUsages::INDIRECT,
            )
            .unwrap(),
            num_reduce_wgs: Tensor::scalar(
                backend,
                [1, 1, 1],
                BufferUsages::STORAGE | BufferUsages::INDIRECT,
            )
            .unwrap(),
            output_keys_pong: Tensor::vector_uninit(backend, 0, BufferUsages::STORAGE).unwrap(),
            output_values_pong: Tensor::vector_uninit(backend, 0, BufferUsages::STORAGE).unwrap(),
        }
    }
}

impl RadixSort {
    /// Dispatches the radix sort operation to sort keys with associated values.
    ///
    /// The sort is stable: elements with equal keys maintain their relative order.
    /// Both keys and values are sorted together, making this useful for indirect sorting
    /// (where values are indices into another array).
    ///
    /// # Parameters
    ///
    /// - `backend`: The GPU backend
    /// - `pass`: The compute pass to record commands into
    /// - `workspace`: Workspace buffers (automatically resized if needed)
    /// - `input_keys`: The u32 keys to sort
    /// - `input_values`: Associated values to sort alongside keys
    /// - `n_sort`: Number of elements to sort (must be <= input buffer size)
    /// - `sorting_bits`: Number of bits to sort (1-32). Use 32 for full sorting,
    ///   or fewer bits if your keys have a limited range (e.g., 24 for Morton codes)
    /// - `output_keys`: Buffer to write sorted keys to
    /// - `output_values`: Buffer to write sorted values to
    ///
    /// # Panics
    ///
    /// - Panics if `input_keys` and `input_values` have different lengths
    /// - Panics if `sorting_bits > 32`
    pub fn dispatch(
        &self,
        backend: &GpuBackend,
        pass: &mut GpuPass,
        workspace: &mut RadixSortWorkspace,
        input_keys: &Tensor<u32>,
        input_values: &Tensor<u32>,
        n_sort: &Tensor<u32>,
        sorting_bits: u32,
        num_batches: u32,
        output_keys: &mut Tensor<u32>,
        output_values: &mut Tensor<u32>,
    ) -> Result<(), GpuBackendError> {
        assert_eq!(
            input_keys.len(),
            input_values.len(),
            "Input keys and values must have the same number of elements"
        );
        assert!(sorting_bits <= 32, "Can only sort up to 32 bits");

        let max_n = input_keys.len() as u32;
        let per_batch_max = max_n / num_batches;

        // compute buffer and dispatch sizes (per-batch workgroups, scaled by num_batches)
        let max_needed_wgs = per_batch_max.div_ceil(BLOCK_SIZE);
        let needed_count = (num_batches as u64) * (max_needed_wgs as u64) * (BIN_COUNT as u64);
        if workspace.count_buf.len() < needed_count {
            workspace.count_buf =
                Tensor::vector_uninit(backend, needed_count as u32, BufferUsages::STORAGE)?;
        }

        let needed_reduced = (num_batches as u64) * (BLOCK_SIZE as u64);
        if workspace.reduced_buf.len() < needed_reduced {
            let zeros = vec![0u32; needed_reduced as usize];
            workspace.reduced_buf =
                Tensor::vector(backend, &zeros, BufferUsages::STORAGE)?;
        }

        self.init.call(
            pass,
            1u32,
            n_sort,
            &mut workspace.num_wgs,
            &mut workspace.num_reduce_wgs,
        )?;

        if workspace.output_keys_pong.len() < input_keys.len() {
            // TODO: is this OK even in the case where we call the radix sort multiple times
            //       successively but with increasing input buffer sizes? Wondering if that could
            //       free the previous buffer and then crash the previous invocation.
            workspace.output_keys_pong =
                Tensor::vector_uninit(backend, input_keys.len() as u32, BufferUsages::STORAGE)?;
            workspace.output_values_pong =
                Tensor::vector_uninit(backend, input_values.len() as u32, BufferUsages::STORAGE)?;
        }

        let num_passes = sorting_bits.div_ceil(4);
        let mut output_keys = output_keys;
        let mut output_values = output_values;
        let mut output_keys_pong = &mut workspace.output_keys_pong;
        let mut output_values_pong = &mut workspace.output_values_pong;

        if num_passes.is_multiple_of(2) {
            // Make sure the last pass has the user provided `output_keys`
            // set as the output buffer so that the final results doesn’t end
            // up stored in the workspace’s pong buffers instead.
            std::mem::swap(&mut output_keys, &mut output_keys_pong);
            std::mem::swap(&mut output_values, &mut output_values_pong);
        }

        macro_rules! run_pass {
            ($pass_id: expr, $src: ident, $values: ident) => {
                if $pass_id as usize >= workspace.pass_uniforms.len() {
                    workspace.pass_uniforms.push(Tensor::scalar(
                        backend,
                        SortUniforms {
                            shift: $pass_id * 4,
                        },
                        BufferUsages::STORAGE | BufferUsages::UNIFORM,
                    )?);
                }

                let uniforms_buffer = &workspace.pass_uniforms[$pass_id as usize];

                self.count.call(
                    pass,
                    &workspace.num_wgs,
                    uniforms_buffer,
                    n_sort,
                    $src,
                    &mut workspace.count_buf,
                )?;

                self.reduce.call(
                    pass,
                    &workspace.num_reduce_wgs,
                    n_sort,
                    &workspace.count_buf,
                    &mut workspace.reduced_buf,
                )?;

                self.scan
                    .call(pass, [1u32, num_batches, 1], n_sort, &mut workspace.reduced_buf)?;

                self.scan_add.call(
                    pass,
                    &workspace.num_reduce_wgs,
                    n_sort,
                    &workspace.reduced_buf,
                    &mut workspace.count_buf,
                )?;

                self.scatter.call(
                    pass,
                    &workspace.num_wgs,
                    uniforms_buffer,
                    n_sort,
                    $src,
                    $values,
                    &workspace.count_buf,
                    output_keys,
                    output_values,
                )?;
            };
        }

        if num_passes > 0 {
            run_pass!(0, input_keys, input_values);

            let mut cur_keys = output_keys;
            let mut cur_vals = output_values;
            output_keys = output_keys_pong;
            output_values = output_values_pong;

            for pass_id in 1..num_passes {
                run_pass!(pass_id, cur_keys, cur_vals);
                std::mem::swap(&mut cur_keys, &mut output_keys);
                std::mem::swap(&mut cur_vals, &mut output_values);
            }
        }
        Ok(())
    }
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use crate::utils::RadixSort;
    use crate::utils::radix_sort::RadixSortWorkspace;
    use khal::{BufferUsages, Shader};
    use khal::backend::{Backend, Encoder, GpuBackend, WebGpu};
    use vortx::tensor::Tensor;

    pub fn cpu_argsort<T: Ord>(data: &[T]) -> Vec<usize> {
        let mut indices = (0..data.len()).collect::<Vec<_>>();
        indices.sort_by_key(|&i| &data[i]);
        indices
    }

    #[futures_test::test]
    #[serial_test::serial]
    async fn test_sorting() {
        let gpu = GpuBackend::WebGpu(WebGpu::default().await.unwrap());
        let sort = RadixSort::from_backend(&gpu).unwrap();
        let mut workspace = RadixSortWorkspace::new(&gpu);

        for i in 0u32..128 {
            let keys_inp = [
                5 + i * 4,
                i,
                6,
                123,
                74657,
                123,
                999,
                2u32.pow(24) + 123,
                6,
                7,
                8,
                0,
                i * 2,
                16 + i,
                128 * i,
            ];

            let values_inp: Vec<_> = keys_inp.iter().copied().map(|x| x * 2 + 5).collect();

            let input_usages = BufferUsages::STORAGE;
            let output_usages = BufferUsages::STORAGE | BufferUsages::COPY_SRC;

            let mut keys = Tensor::vector(&gpu, keys_inp, input_usages).unwrap();
            let mut values = Tensor::vector(&gpu, &values_inp, input_usages).unwrap();
            let mut out_keys = Tensor::vector(&gpu, keys_inp, output_usages).unwrap();
            let mut out_values = Tensor::vector(&gpu, &values_inp, output_usages).unwrap();
            let num_points =
                Tensor::scalar(&gpu, keys_inp.len() as u32, BufferUsages::STORAGE).unwrap();

            let mut encoder = gpu.begin_encoding();
            let mut pass = encoder.begin_pass("test", None);
            sort.dispatch(
                &gpu,
                &mut pass,
                &mut workspace,
                &mut keys,
                &mut values,
                &num_points,
                32,
                1,
                &mut out_keys,
                &mut out_values,
            ).unwrap();
            drop(pass);
            gpu.submit(encoder);

            let result_keys = gpu.slow_read_vec(out_keys.buffer()).await.unwrap();
            let result_values = gpu.slow_read_vec(out_values.buffer()).await.unwrap();

            let inds = cpu_argsort(&keys_inp);
            let ref_keys: Vec<u32> = inds.iter().map(|&i| keys_inp[i]).collect();
            let ref_values: Vec<u32> = inds.iter().map(|&i| values_inp[i]).collect();

            assert_eq!(ref_keys, result_keys);
            assert_eq!(ref_values, result_values);
        }
    }

    #[futures_test::test]
    #[serial_test::serial]
    async fn test_sorting_big() {
        use rand::Rng;

        let gpu = GpuBackend::WebGpu(WebGpu::default().await.unwrap());
        let sort = RadixSort::from_backend(&gpu).unwrap();
        let mut workspace = RadixSortWorkspace::new(&gpu);

        // Simulate some data as one might find for a bunch of gaussians.
        let mut rng = rand::rng();
        let mut keys_inp = Vec::new();
        for i in 0..10000 {
            let start = rng.random_range(i..i + 150);
            let end = rng.random_range(start..start + 250);

            for j in start..end {
                if rng.random::<f32>() < 0.5 {
                    keys_inp.push(j);
                }
            }
        }
        let values_inp: Vec<_> = keys_inp.iter().map(|&x| x * 2 + 5).collect();

        let input_usages = BufferUsages::STORAGE;
        let output_usages = BufferUsages::STORAGE | BufferUsages::COPY_SRC;

        let mut keys = Tensor::vector(&gpu, &keys_inp, input_usages).unwrap();
        let mut values = Tensor::vector(&gpu, &values_inp, input_usages).unwrap();
        let mut out_keys = Tensor::vector(&gpu, &keys_inp, output_usages).unwrap();
        let mut out_values = Tensor::vector(&gpu, &values_inp, output_usages).unwrap();
        let num_points =
            Tensor::scalar(&gpu, keys_inp.len() as u32, BufferUsages::STORAGE).unwrap();

        let mut encoder = gpu.begin_encoding();
        let mut pass = encoder.begin_pass("test", None);
        sort.dispatch(
            &gpu,
            &mut pass,
            &mut workspace,
            &mut keys,
            &mut values,
            &num_points,
            32,
            1,
            &mut out_keys,
            &mut out_values,
        );
        drop(pass);
        gpu.submit(encoder);

        let result_keys = gpu.slow_read_vec(out_keys.buffer()).await.unwrap();
        let result_values = gpu.slow_read_vec(out_values.buffer()).await.unwrap();

        let inds = cpu_argsort(&keys_inp);
        let ref_keys: Vec<u32> = inds.iter().map(|&i| keys_inp[i]).collect();
        let ref_values: Vec<u32> = inds.iter().map(|&i| values_inp[i]).collect();

        assert_eq!(ref_keys, result_keys);
        assert_eq!(ref_values, result_values);
    }
}
