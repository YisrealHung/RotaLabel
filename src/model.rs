//! Core data model: every box is stored as `(cx, cy, w, h, angle)` in pixels.

use serde::{Deserialize, Serialize};

use crate::geometry;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum BoxKind {
    #[default]
    Bbox,
    Obb,
}

impl BoxKind {
    pub fn as_str(self) -> &'static str {
        match self {
            BoxKind::Bbox => "bbox",
            BoxKind::Obb => "obb",
        }
    }
}

pub const DEFAULT_COLORS: [&str; 20] = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0",
    "#f032e6", "#bcf60c", "#fabebe", "#008080", "#e6beff", "#9a6324", "#fffac8",
    "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000075", "#808080",
];

pub fn default_color(index: usize) -> String {
    DEFAULT_COLORS[index % DEFAULT_COLORS.len()].to_string()
}

fn round4(v: f64) -> f64 {
    let r = (v * 10000.0).round() / 10000.0;
    if r == 0.0 { 0.0 } else { r }
}

fn serialize_round4<S: serde::Serializer>(v: &f64, s: S) -> Result<S::Ok, S::Error> {
    s.serialize_f64(round4(*v))
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Annotation {
    pub cls: usize,
    #[serde(serialize_with = "serialize_round4")]
    pub cx: f64,
    #[serde(serialize_with = "serialize_round4")]
    pub cy: f64,
    #[serde(serialize_with = "serialize_round4")]
    pub w: f64,
    #[serde(serialize_with = "serialize_round4")]
    pub h: f64,
    #[serde(default, serialize_with = "serialize_round4")]
    pub angle: f64,
    #[serde(default)]
    pub kind: BoxKind,
}

impl Annotation {
    pub fn new(cls: usize, cx: f64, cy: f64, w: f64, h: f64, angle: f64, kind: BoxKind) -> Self {
        Self { cls, cx, cy, w, h, angle, kind }
    }

    pub fn corners(&self) -> [[f64; 2]; 4] {
        geometry::box_corners(self.cx, self.cy, self.w, self.h, self.angle)
    }

    pub fn aabb(&self) -> [f64; 4] {
        if self.kind == BoxKind::Bbox || self.angle == 0.0 {
            [
                self.cx - self.w / 2.0,
                self.cy - self.h / 2.0,
                self.cx + self.w / 2.0,
                self.cy + self.h / 2.0,
            ]
        } else {
            geometry::aabb(&self.corners())
        }
    }

    pub fn to_bbox(&self) -> Self {
        let [x0, y0, x1, y1] = self.aabb();
        Self::new(
            self.cls,
            (x0 + x1) / 2.0,
            (y0 + y1) / 2.0,
            x1 - x0,
            y1 - y0,
            0.0,
            BoxKind::Bbox,
        )
    }

    pub fn to_obb(&self) -> Self {
        Self::new(self.cls, self.cx, self.cy, self.w, self.h, self.angle, BoxKind::Obb)
    }

    pub fn from_corners(cls: usize, pts: &[[f64; 2]], kind: BoxKind) -> Self {
        if kind == BoxKind::Bbox {
            let [x0, y0, x1, y1] = geometry::aabb(pts);
            return Self::new(
                cls,
                (x0 + x1) / 2.0,
                (y0 + y1) / 2.0,
                x1 - x0,
                y1 - y0,
                0.0,
                BoxKind::Bbox,
            );
        }
        let hint = if pts.len() >= 2 {
            let d = [pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]];
            Some(d[1].atan2(d[0]).to_degrees())
        } else {
            None
        };
        let (cx, cy, w, h, a) = geometry::fit_obb(pts, hint);
        Self::new(cls, cx, cy, w, h, a, BoxKind::Obb)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClassDef {
    pub name: String,
    #[serde(default)]
    pub color: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ImageAnnotations {
    #[serde(default)]
    pub width: u32,
    #[serde(default)]
    pub height: u32,
    #[serde(default)]
    pub annotations: Vec<Annotation>,
}
