use rand;
use rand::Rng;
use std::os::raw::c_void;
use std::slice;

pub struct MLP {
    pub weights: Vec<Vec<Vec<f64>>>,
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
        let mut v1 : Vec<Vec<Vec<f64>>> = Vec::new();//Layers vector

        for i in 0..entries.len(){
            let mut v2 : Vec<Vec<f64>> = Vec::new();//Neurals vector
            for _j in 0..entries[i as usize] + 1{
                let mut v3 : Vec<f64> = Vec::new();//weights vector
                if i != 0{
                    for _k in 0..entries[i - 1] + 1{
                        v3.push(MLP::generate_weight());
                    }
                }
                v2.push(v3);
            }
            v1.push(v2);
        }
        v1
    }

    pub fn activation_function(vi : f64) -> f64{
        //1./(1. + (-1. * vi).exp())
        -((-vi).exp() - vi.exp()) / ((-vi).exp() + vi.exp())
    }

    pub fn generate_weight() -> f64{
        rand::thread_rng().gen_range(-1., 1.)
    }
}
