pub struct Point{
    pub x: f64,
    pub y: f64,
    pub z: f64
}

pub struct MLP {
    pub weights: Vec<Vec<Vec<f64>>>,
    pub output: Vec<Vec<f64>>,
    pub delta: Vec<Vec<f64>>,
    pub neurals: Vec<i32>
}
