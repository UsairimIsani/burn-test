use burn::optim::AdamConfig;
use burn_autodiff::ADBackendDecorator;
use burn_wgpu::{AutoGraphicsApi, WgpuBackend, WgpuDevice};
use burn_test::model::ModelConfig;
use burn_test::training::{train, TrainingConfig};

fn main() {
    type MyBackend = WgpuBackend<AutoGraphicsApi, f32, i32>;
    type MyAutodiffBackend = ADBackendDecorator<MyBackend>;

    let device = WgpuDevice::default();
    train::<MyAutodiffBackend>(
        "/tmp/guide",
       TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
        device.clone(),
    );
}
