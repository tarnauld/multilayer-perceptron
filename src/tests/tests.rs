use std::os::raw::c_void;
use core::externs::*;

#[test]
fn should_complete_a_xor() {
    unsafe{
        let neuron_layers = Box::into_raw(Box::new([2, 3, 1])) as *mut c_void;
        let mlp = create_model(neuron_layers, 3);
        let data_set = Box::into_raw(Box::new([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0])) as *mut c_void;
        train_weights(mlp, 10000, data_set, 12);
        assert!(classification(mlp, [1., 1.]) > 0.9 && classification(mlp, [1., 1.]) < 1.);
        assert!(classification(mlp, [-1., -1.]) > 0.9 && classification(mlp, [-1., -1.]) < 1.);
        assert!(classification(mlp, [1., -1.]) < -0.9 && classification(mlp, [1., -1.]) > -1.);
        assert!(classification(mlp, [-1., 1.]) < -0.9 && classification(mlp, [-1., 1.]) > -1.);
    }
}
