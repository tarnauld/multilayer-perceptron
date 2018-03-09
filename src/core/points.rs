use core::structs::Point;

fn new(x: f64, y: f64, z: f64) -> Point{
    Point{
        x: x,
        y: y,
        z: z
    }
}
pub fn array_to_point(points: &[f64]) -> Vec<Point>{
    let mut v : Vec<Point> = Vec::new();
    let mut i = 0;
    loop{
        if i > points.len() - 1 {break;}
        v.push(new(points[i as usize], points[(i + 1) as usize], points[(i + 2) as usize]));
        i += 3;
    }
    v
}
