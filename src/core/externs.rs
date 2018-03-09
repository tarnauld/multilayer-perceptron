use std;
use std::os::raw::c_void;
use core::structs::MLP;
use core::points::array_to_point;
use core::mlp::*;

#[no_mangle]
pub extern fn create_model(entries: *mut c_void, length: i32) -> *mut MLP {
    let mlp: &[i32];
    unsafe{
        mlp = std::slice::from_raw_parts(entries as *mut i32, length as usize);
    }
    let mut max: i32 = 0;
    let mut neurals = mlp.to_vec();
    for nb in neurals.iter_mut() {
        *nb += 1;
        if *nb > max {
            max = *nb;
        }
    }
    let weights = init_weights(&neurals, max);
    let output = init_all(&neurals);
    let delta = init_all(&neurals);
    let m = MLP {
        weights,
        output,
        neurals: neurals,
        delta
    };
    Box::into_raw(Box::new(m))
}

#[no_mangle]
pub unsafe extern fn train_weights(mlp: *mut MLP, nb: i32, pts: *mut c_void, length: i32) {
    let points = array_to_point(std::slice::from_raw_parts(pts as *mut f64, length as usize));
    for _i in 0..nb{
        for j in 0..points.len(){
            let point = &points[j as usize];
            train_neural(mlp, &point);
        }
    }
}

#[no_mangle]
pub unsafe extern fn classification(mlp: *mut MLP, point: [f64; 2]) -> f64 {
    default_bottom(mlp);
    (*mlp).output[0][1] = point[0];
    (*mlp).output[0][2] = point[1];
    for i in 1..(*mlp).neurals.len() {
        let nb = (*mlp).neurals[i];
        for j in 1..nb {
            (*mlp).output[i][j as usize] = calculate_output_classification(mlp, i as i32,j);
        }
    }
    (*mlp).output[(*mlp).neurals.len() - 1][1]
}
