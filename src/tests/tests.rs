use core::mlp;
use std;
use std::os::raw::c_void;

#[test]
fn should_generate_random_weight(){
    let entries = Box::into_raw(Box::new([2, 3, 1])) as *mut c_void;
    let mut mlp = mlp::create_model(entries, 3);
}
