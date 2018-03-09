use std::os::raw::c_void;
use core::externs::*;

#[test]
fn should_complete_a_xor_classification() {
    unsafe{
        let neuron_layers = Box::into_raw(Box::new([2, 3, 1])) as *mut c_void;
        let mlp = create_model(neuron_layers, 3, true);
        let data_set = Box::into_raw(Box::new([1.0, 1.0, 1.0,
                                               1.0, -1.0, -1.0,
                                              -1.0, 1.0, -1.0,
                                              -1.0, -1.0, 1.0])) as *mut c_void;
        train_weights(mlp, 10000, data_set, 12);
        println!("{:?}", prediction(mlp, [1., 1.]));
        println!("{:?}", prediction(mlp, [-1., -1.]));
        println!("{:?}", prediction(mlp, [1., -1.]));
        println!("{:?}", prediction(mlp, [-1., 1.]));
        assert!(prediction(mlp, [1., 1.]) > 0.9 && prediction(mlp, [1., 1.]) < 1.);
        assert!(prediction(mlp, [-1., -1.]) > 0.9 && prediction(mlp, [-1., -1.]) < 1.);
        assert!(prediction(mlp, [1., -1.]) < -0.9 && prediction(mlp, [1., -1.]) > -1.);
        assert!(prediction(mlp, [-1., 1.]) < -0.9 && prediction(mlp, [-1., 1.]) > -1.);
    }
}

#[test]
fn should_complete_a_xor2_classification() {
    unsafe{
        let neuron_layers = Box::into_raw(Box::new([2, 3, 3, 3, 1])) as *mut c_void;
        let mlp = create_model(neuron_layers, 5, true);
        let data_set = Box::into_raw(Box::new([5.0, 2.0, 5.0,
                                               5.0, -2.0, -5.0,
                                               -5.0, 2.0, -5.0,
                                               -5.0, -2.0, 5.0])) as *mut c_void;
        train_weights(mlp, 10000, data_set, 12);
        println!("{:?}", prediction(mlp, [5., 5.]));
        println!("{:?}", prediction(mlp, [-5., -5.]));
        println!("{:?}", prediction(mlp, [5., -5.]));
        println!("{:?}", prediction(mlp, [-5., 5.]));
        assert!(prediction(mlp, [5., 5.]) > 0.9 && prediction(mlp, [5., 5.]) < 1.);
        assert!(prediction(mlp, [-5., -5.]) > 0.9 && prediction(mlp, [-5., -5.]) < 1.);
        assert!(prediction(mlp, [5., -5.]) < -0.9 && prediction(mlp, [5., -5.]) > -1.);
        assert!(prediction(mlp, [-5., 5.]) < -0.9 && prediction(mlp, [-5., 5.]) > -1.);
    }
}

#[test]
fn should_complete_a_xor_regression() {
    unsafe{
        let neuron_layers = Box::into_raw(Box::new([2, 3, 1])) as *mut c_void;
        let mlp = create_model(neuron_layers, 3, false);
        let data_set = Box::into_raw(Box::new([1.0, 1.0, 1.0,
                                               1.0, -1.0, -1.0,
                                              -1.0, 1.0, -1.0,
                                              -1.0, -1.0, 1.0])) as *mut c_void;
        train_weights(mlp, 10000, data_set, 12);
        println!("{:?}", prediction(mlp, [1., 1.]));
        println!("{:?}", prediction(mlp, [-1., -1.]));
        println!("{:?}", prediction(mlp, [1., -1.]));
        println!("{:?}", prediction(mlp, [-1., 1.]));
        /*assert!(prediction(mlp, [1., 1.]) > 0.9 && prediction(mlp, [1., 1.]) < 1.);
        assert!(prediction(mlp, [-1., -1.]) > 0.9 && prediction(mlp, [-1., -1.]) < 1.);
        assert!(prediction(mlp, [1., -1.]) < -0.9 && prediction(mlp, [1., -1.]) > -1.);
        assert!(prediction(mlp, [-1., 1.]) < -0.9 && prediction(mlp, [-1., 1.]) > -1.);*/
    }
}

#[test]
fn should_complete_a_xor2_regression() {
    unsafe{
        let neuron_layers = Box::into_raw(Box::new([2, 3, 3, 3, 1])) as *mut c_void;
        let mlp = create_model(neuron_layers, 5, false);
        let data_set = Box::into_raw(Box::new([5.0, 2.0, 5.0,
                                               5.0, -2.0, -5.0,
                                               -5.0, 2.0, -5.0,
                                               -5.0, -2.0, 5.0])) as *mut c_void;
        train_weights(mlp, 10000, data_set, 12);
        println!("{:?}", prediction(mlp, [5., 5.]));
        println!("{:?}", prediction(mlp, [-5., -5.]));
        println!("{:?}", prediction(mlp, [5., -5.]));
        println!("{:?}", prediction(mlp, [-5., 5.]));
        /*assert!(prediction(mlp, [5., 5.]) > 0.9 && prediction(mlp, [5., 5.]) < 1.);
        assert!(prediction(mlp, [-5., -5.]) > 0.9 && prediction(mlp, [-5., -5.]) < 1.);
        assert!(prediction(mlp, [5., -5.]) < -0.9 && prediction(mlp, [5., -5.]) > -1.);
        assert!(prediction(mlp, [-5., 5.]) < -0.9 && prediction(mlp, [-5., 5.]) > -1.);*/
    }
}
