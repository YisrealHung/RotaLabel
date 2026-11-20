use crate::model::Annotation;

pub const SPLITS: [&str; 3] = ["train", "valid", "test"];

pub fn clipped_corners(ann: &Annotation, width: u32, height: u32) -> [[f64; 2]; 4] {
    let mut pts = ann.corners();
    let fw = width as f64;
    let fh = height as f64;
    for p in &mut pts {
        p[0] = p[0].clamp(0.0, fw);
        p[1] = p[1].clamp(0.0, fh);
    }
    pts
}

pub fn clipped_aabb(ann: &Annotation, width: u32, height: u32) -> [f64; 4] {
    let [x0, y0, x1, y1] = ann.aabb();
    let fw = width as f64;
    let fh = height as f64;
    [x0.max(0.0), y0.max(0.0), x1.min(fw), y1.min(fh)]
}

pub fn fmt(v: f64) -> String {
    if v == 0.0 {
        return "0".to_string();
    }
    let s = format!("{:.6}", v);
    let trimmed = s.trim_end_matches('0').trim_end_matches('.');
    if trimmed.is_empty() || trimmed == "-0" {
        "0".to_string()
    } else {
        trimmed.to_string()
    }
}
