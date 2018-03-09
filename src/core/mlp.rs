use rand;
use rand::Rng;
use core::structs::Point;
use core::structs::MLP;
use std::f64;

pub fn init_weights(neurals: &[i32], max: i32) -> Vec<Vec<Vec<f64>>> {
    let mut weights: Vec<Vec<Vec<f64>>> = Vec::new();
    for _i in 0..neurals.len() {
        let mut v1 = Vec::new();
        for _j in 0..max {
            let mut v2 = Vec::new();
            for _k in 0..max {
                v2.push(rand::thread_rng().gen_range(-1.0, 1.0));
            }
            v1.push(v2);
        }
        weights.push(v1);
    }
    weights
}

pub unsafe fn default_bottom(mlp: *mut MLP) {
    for i in 0..(*mlp).neurals.len() {
        (*mlp).output[i][0] = 1.0;
    }
}

pub fn init_all(neurals: &[i32]) -> Vec<Vec<f64>> {
    let mut v1: Vec<Vec<f64>> = Vec::new();
    for nb in neurals {
        let mut v2: Vec<f64> = Vec::new();
        for _ in 0..*nb {
            v2.push(0.0);
        }
        v1.push(v2);
    }
    v1
}

pub unsafe fn train_neural(mlp: *mut MLP, point: &Point) {
    default_bottom(mlp);
    (*mlp).output[0][1] = point.x;
    (*mlp).output[0][2] = point.z;
    for i in 1..(*mlp).neurals.len() {
        let nb_neurons_for_layer = (*mlp).neurals[i];
        for j in 1..nb_neurons_for_layer {
            (*mlp).output[i][j as usize] = calculate_output_prediction(mlp, i as i32,j);
        }
    }
    get_delta(mlp, point.y as f64);
    gradient_retropropagation(mlp);
    update_weights(mlp);
}

unsafe fn update_weights(mlp: *mut MLP) {
    for i in 1..(*mlp).neurals.len() {
        let neurons_in_layer = (*mlp).neurals[i];
        let neurons_in_previous_layer = (*mlp).neurals[i-1];
        for j in 0..neurons_in_previous_layer {
            for k in 0..neurons_in_layer {
                let left = (*mlp).weights[i][j as usize][k as usize];
                let right = 0.1 * (*mlp).output[i-1][j as usize] * (*mlp).delta[i][k as usize];
                (*mlp).weights[i][j as usize][k as usize] = left - right;
            }
        }
    }
}

unsafe fn gradient_retropropagation(mlp: *mut MLP) {
    for i in (1..((*mlp).neurals.len() - 1)).rev() {
        for j in 0..(*mlp).neurals[i] {
            let left = 1. - (*mlp).output[i][j as usize].powf(2.0);
            let mut right = 0.;
            for k in 1..(*mlp).neurals[i+1] {
                right+= (*mlp).weights[i+1][j as usize][k as usize] * (*mlp).delta[i+1][k as usize];
            }
            (*mlp).delta[i][j as usize] = left * right;
        }
    }
}

unsafe fn get_delta(mlp: *mut MLP, y: f64) {
    let layer = (*mlp).neurals.len() - 1;
    for i in 0..*(*mlp).neurals.last().unwrap() {
        let o = (*mlp).output[layer][i as usize];
        if (*mlp).classification == true{
            (*mlp).delta[layer][i as usize] = (1. - o.powf(2.0)) * (o - y);
        }else{
            (*mlp).delta[layer][i as usize] = o - y;
        }
    }
}

pub unsafe fn calculate_output_prediction(mlp: *mut MLP, layer_number: i32, index: i32) -> f64 {
    let layer_number = layer_number;
    let nb = (*mlp).neurals[(layer_number -1) as usize];
    let mut x = 0.0;
    for i in 0..nb {
        let weight = (*mlp).weights[layer_number as usize][i as usize][index as usize];
        let x_i = (*mlp).output[(layer_number -1) as usize][i as usize];
        x += weight * x_i;
    }
    if (*mlp).classification == true{
        return activation_function_tanh(x);
    }else{
        if layer_number as usize == ((*mlp).neurals.len() - 1){
                x
        }
        else {
            activation_function_tanh(x)
        }
    }
}

fn activation_function_tanh(x: f64) -> f64{
    (1. - (-2. * x).exp()) / (1. + (-2. * x).exp())
}
