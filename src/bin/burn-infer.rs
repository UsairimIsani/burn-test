use burn_autodiff::ADBackendDecorator;
use burn_dataset::Dataset;
use burn_dataset::source::huggingface::MNISTDataset;
use burn_wgpu::{AutoGraphicsApi, WgpuBackend, WgpuDevice};
use burn_test::model::infer;

fn main() {
    type MyBackend = WgpuBackend<AutoGraphicsApi, f32, i32>;
    type MyAutodiffBackend = ADBackendDecorator<MyBackend>;

    let device = WgpuDevice::default();

    infer::<MyAutodiffBackend>(
        "/tmp/guide",device, MNISTDataset::test().get(1).unwrap())
}
