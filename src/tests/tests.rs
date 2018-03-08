use core::mlp;
use std::os::raw::c_void;

#[test]
fn should_generate_random_weight(){
    let entries = Box::into_raw(Box::new([2, 3, 1])) as *mut c_void;
    let mlp = mlp::create_model(entries, 3);
    let mut nbtotal = 0;

    unsafe{
        println!("{:?} layers.", (*mlp).weights.len());
        for (i, v1) in (*mlp).weights.iter().enumerate(){
            println!("{:?} neurals for layer {:?}", (*mlp).weights[i].len(), i);
            for (j, v2) in v1.iter().enumerate(){
                let mut nbweight = 0;
                for (_k, v3) in v2.iter().enumerate(){
                    nbweight += 1;
                    nbtotal += 1;
                }
                println!("{:?} weights for layer {:?}", nbweight, i);
            }
            println!();
        }
    }
    assert!(nbtotal == 20);
}

/*

2, 3, 1

Three Layers
1st layer -> 2 Neurals + 1 step = 3 Neurals
2nd layer -> 3 Neurals + 1 step = 4 Neurals
3rd layer -> 1 Neurals + 1 step = 2 Neurals

*/
