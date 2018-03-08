use rand;
use rand::Rng;
use std::os::raw::c_void;
use std::slice;

pub struct MLP {
    weights: Vec<Vec<Vec<f64>>>,
    values: Vec<Vec<f64>>,
    input: i32,
    output: i32
}

#[no_mangle]
pub extern fn create_model(entries: *mut c_void, length: usize) -> *mut MLP{
    let mlp: &[i32];
    unsafe{
        mlp = slice::from_raw_parts(entries as *mut i32, length);
    }
    Box::into_raw(Box::new(MLP::new(mlp)))
}

impl MLP{
    pub fn new(entries: &[i32]) -> MLP{
        MLP{
            weights: MLP::init_weigths(entries),
            values: Vec::new(),
            input: entries[0],
            output: entries[entries.len() - 1]
        }
    }

    pub fn init_weigths(entries: &[i32]) -> Vec<Vec<Vec<f64>>> {
        let mut v : Vec<Vec<Vec<f64>>> = Vec::new();
        for i in 0..entries.len(){
            let mut v1 : Vec<Vec<f64>> = Vec::new();
            for j in 0..entries[i]{
                let mut v2 : Vec<f64> = Vec::new();
                for k in 0..entries[j as usize]{
                    v2.push(rand::thread_rng().gen_range(-1., 1.));
                }
                v1.push(v2);
            }
            v.push(v1);
        }
        v
    }

    pub fn generate_weight(){
        rand::thread_rng().gen_range(-1., 1.);
    }
}
