//! Annotation canvas: zoom/pan, drawing bbox/rotated boxes, move, resize, rotate, undo.

use egui::{
    Align2, Color32, CornerRadius, CursorIcon, FontId, Key, PointerButton, Pos2, Rect, Sense,
    Shape, Stroke, TextureHandle, Vec2,
};

use crate::geometry::norm180;
use crate::model::{default_color, Annotation, BoxKind, ClassDef};
use crate::ui::theme::{color_lightness, parse_hex_color, Palette};

pub const HANDLE_PX: f32 = 9.0;
pub const ROT_OFFSET_PX: f32 = 28.0;
pub const MIN_BOX: f64 = 2.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LabelMode {
    All,
    Selected,
    None,
    Hidden,
}

impl LabelMode {
    pub const ALL_MODES: [LabelMode; 4] = [
        LabelMode::All,
        LabelMode::Selected,
        LabelMode::None,
        LabelMode::Hidden,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            LabelMode::All => "all",
            LabelMode::Selected => "selected",
            LabelMode::None => "none",
            LabelMode::Hidden => "hidden",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "selected" => LabelMode::Selected,
            "none" => LabelMode::None,
            "hidden" => LabelMode::Hidden,
            _ => LabelMode::All,
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            LabelMode::All => "全部",
            LabelMode::Selected => "選取",
            LabelMode::None => "只有框",
            LabelMode::Hidden => "隱藏",
        }
    }

    pub fn long_label(self) -> &'static str {
        match self {
            LabelMode::All => "全部顯示（框＋標籤）",
            LabelMode::Selected => "只顯示選取框的標籤",
            LabelMode::None => "只顯示框（隱藏標籤）",
            LabelMode::Hidden => "全部隱藏（只看圖片）",
        }
    }

    pub fn cycle(self) -> Self {
        match self {
            LabelMode::All => LabelMode::Selected,
            LabelMode::Selected => LabelMode::None,
            LabelMode::None => LabelMode::Hidden,
            LabelMode::Hidden => LabelMode::All,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolMode {
    Select,
    Bbox,
    Obb,
}

impl ToolMode {
    pub fn label(self) -> &'static str {
        match self {
            ToolMode::Select => "選取",
            ToolMode::Bbox => "BBox",
            ToolMode::Obb => "旋轉框",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandleId {
    Tl,
    T,
    Tr,
    R,
    Br,
    B,
    Bl,
    L,
    Rot,
}

impl HandleId {
    pub const RESIZE_HANDLES: [(HandleId, f64, f64); 8] = [
        (HandleId::Tl, -1.0, -1.0),
        (HandleId::T, 0.0, -1.0),
        (HandleId::Tr, 1.0, -1.0),
        (HandleId::R, 1.0, 0.0),
        (HandleId::Br, 1.0, 1.0),
        (HandleId::B, 0.0, 1.0),
        (HandleId::Bl, -1.0, 1.0),
        (HandleId::L, -1.0, 0.0),
    ];

    pub fn signs(self) -> (f64, f64) {
        match self {
            HandleId::Tl => (-1.0, -1.0),
            HandleId::T => (0.0, -1.0),
            HandleId::Tr => (1.0, -1.0),
            HandleId::R => (1.0, 0.0),
            HandleId::Br => (1.0, 1.0),
            HandleId::B => (0.0, 1.0),
            HandleId::Bl => (-1.0, 1.0),
            HandleId::L => (-1.0, 0.0),
            HandleId::Rot => (0.0, -1.0),
        }
    }
}

#[derive(Debug, Clone)]
pub enum DragState {
    Pan {
        start_pointer: [f32; 2],
        start_pan: [f32; 2],
    },
    Draw {
        start_img: [f64; 2],
        end_img: [f64; 2],
    },
    Move {
        idx: usize,
        start_img: [f64; 2],
        orig: Annotation,
        before: Vec<Annotation>,
    },
    Resize {
        idx: usize,
        handle: HandleId,
        orig: Annotation,
        before: Vec<Annotation>,
    },
    Rotate {
        idx: usize,
        orig: Annotation,
        before: Vec<Annotation>,
    },
}

#[derive(Debug, Clone)]
pub struct UndoEntry {
    pub before: Vec<Annotation>,
    pub after: Vec<Annotation>,
    pub text: &'static str,
}

#[derive(Debug, Clone)]
pub struct CanvasState {
    pub has_image: bool,
    pub img_w: u32,
    pub img_h: u32,
    pub items: Vec<Annotation>,
    pub selected: Option<usize>,
    pub hovered: Option<usize>,
    pub clipboard: Vec<Annotation>,
    pub undo_stack: Vec<UndoEntry>,
    pub undo_pos: usize,
    pub mode: ToolMode,
    pub label_mode: LabelMode,
    pub current_class: usize,
    pub zoom: f32,
    pub pan: [f32; 2],
    pub auto_fit: bool,
    pub last_vp_size: [f32; 2],
    pub space_held: bool,
    pub cursor_img: Option<[f64; 2]>,
    pub drag: Option<DragState>,
}

impl Default for CanvasState {
    fn default() -> Self {
        Self {
            has_image: false,
            img_w: 0,
            img_h: 0,
            items: Vec::new(),
            selected: None,
            hovered: None,
            clipboard: Vec::new(),
            undo_stack: Vec::new(),
            undo_pos: 0,
            mode: ToolMode::Bbox,
            label_mode: LabelMode::All,
            current_class: 0,
            zoom: 1.0,
            pan: [0.0, 0.0],
            auto_fit: true,
            last_vp_size: [0.0, 0.0],
            space_held: false,
            cursor_img: None,
            drag: None,
        }
    }
}

impl CanvasState {
    pub fn clear(&mut self) {
        self.drag = None;
        self.has_image = false;
        self.img_w = 0;
        self.img_h = 0;
        self.items.clear();
        self.selected = None;
        self.hovered = None;
        self.undo_stack.clear();
        self.undo_pos = 0;
    }

    pub fn load_image(&mut self, w: u32, h: u32, anns: Vec<Annotation>, ok: bool) {
        self.clear();
        if ok && w > 0 && h > 0 {
            self.has_image = true;
            self.img_w = w;
            self.img_h = h;
            self.items = anns;
            self.auto_fit = true;
            if self.last_vp_size[0] > 0.0 && self.last_vp_size[1] > 0.0 {
                self.fit_view(self.last_vp_size);
            }
        }
    }

    pub fn fit_view(&mut self, vp_size: [f32; 2]) {
        if !self.has_image || self.img_w == 0 || self.img_h == 0 {
            return;
        }
        let vw = vp_size[0].max(1.0);
        let vh = vp_size[1].max(1.0);
        let iw = self.img_w as f32;
        let ih = self.img_h as f32;
        let scale = (vw / iw).min(vh / ih).clamp(0.02, 60.0);
        self.zoom = scale;
        self.pan = [(vw - iw * scale) * 0.5, (vh - ih * scale) * 0.5];
        self.auto_fit = true;
    }

    pub fn box_visible(&self, idx: usize) -> bool {
        self.label_mode != LabelMode::Hidden || self.selected == Some(idx)
    }

    pub fn label_visible(&self, idx: usize) -> bool {
        match self.label_mode {
            LabelMode::All => true,
            LabelMode::Selected => self.selected == Some(idx) || self.hovered == Some(idx),
            LabelMode::None | LabelMode::Hidden => false,
        }
    }

    pub fn px_per_screen_px(&self) -> f64 {
        1.0 / (self.zoom.max(1e-5) as f64)
    }

    pub fn select(&mut self, idx: Option<usize>) {
        self.selected = idx.filter(|&i| i < self.items.len());
    }

    pub fn can_undo(&self) -> bool {
        self.undo_pos > 0
    }

    pub fn can_redo(&self) -> bool {
        self.undo_pos < self.undo_stack.len()
    }

    pub fn undo_text(&self) -> Option<&'static str> {
        self.can_undo().then(|| self.undo_stack[self.undo_pos - 1].text)
    }

    pub fn redo_text(&self) -> Option<&'static str> {
        self.can_redo().then(|| self.undo_stack[self.undo_pos].text)
    }

    pub fn commit(&mut self, before: Vec<Annotation>, text: &'static str) {
        self.undo_stack.truncate(self.undo_pos);
        self.undo_stack.push(UndoEntry {
            before,
            after: self.items.clone(),
            text,
        });
        self.undo_pos = self.undo_stack.len();
    }

    pub fn undo(&mut self) -> bool {
        if !self.can_undo() {
            return false;
        }
        self.drag = None;
        self.undo_pos -= 1;
        self.items = self.undo_stack[self.undo_pos].before.clone();
        self.selected = None;
        self.hovered = None;
        true
    }

    pub fn redo(&mut self) -> bool {
        if !self.can_redo() {
            return false;
        }
        self.drag = None;
        self.items = self.undo_stack[self.undo_pos].after.clone();
        self.undo_pos += 1;
        self.selected = None;
        self.hovered = None;
        true
    }

    pub fn delete_selected(&mut self) -> bool {
        let Some(idx) = self.selected else {
            return false;
        };
        if idx >= self.items.len() {
            self.selected = None;
            return false;
        }
        self.drag = None;
        let before = self.items.clone();
        self.items.remove(idx);
        self.hovered = None;
        if self.items.is_empty() {
            self.selected = None;
        } else {
            self.selected = Some(idx.min(self.items.len() - 1));
        }
        self.commit(before, "刪除");
        true
    }

    pub fn cancel_or_delete(&mut self) -> bool {
        if matches!(self.drag, Some(DragState::Draw { .. })) {
            self.drag = None;
            return false;
        }
        self.delete_selected()
    }

    pub fn rotate_selected(&mut self, delta: f64) -> bool {
        let Some(idx) = self.selected else {
            return false;
        };
        if idx >= self.items.len() {
            self.selected = None;
            return false;
        }
        let before = self.items.clone();
        let a = &mut self.items[idx];
        if a.kind == BoxKind::Bbox {
            a.kind = BoxKind::Obb;
        }
        a.angle = norm180(a.angle + delta);
        self.commit(before, "旋轉");
        true
    }

    pub fn set_selected_class(&mut self, cls: usize) -> bool {
        let Some(idx) = self.selected else {
            return false;
        };
        if idx >= self.items.len() || self.items[idx].cls == cls {
            return false;
        }
        let before = self.items.clone();
        self.items[idx].cls = cls;
        self.commit(before, "變更類別");
        true
    }

    pub fn toggle_selected_kind(&mut self) -> bool {
        let Some(idx) = self.selected else {
            return false;
        };
        if idx >= self.items.len() {
            self.selected = None;
            return false;
        }
        let before = self.items.clone();
        let new = if self.items[idx].kind == BoxKind::Obb {
            self.items[idx].to_bbox()
        } else {
            self.items[idx].to_obb()
        };
        self.items[idx] = new;
        self.commit(before, "切換框類型");
        true
    }

    pub fn nudge_selected(&mut self, dx: f64, dy: f64) -> bool {
        let Some(idx) = self.selected else {
            return false;
        };
        if idx >= self.items.len() {
            self.selected = None;
            return false;
        }
        let before = self.items.clone();
        self.items[idx].cx += dx;
        self.items[idx].cy += dy;
        self.commit(before, "移動");
        true
    }

    pub fn copy_selected(&mut self) {
        if let Some(idx) = self.selected {
            if let Some(a) = self.items.get(idx) {
                self.clipboard = vec![a.clone()];
            }
        }
    }

    pub fn paste(&mut self, num_classes: usize) -> bool {
        if self.clipboard.is_empty() || !self.has_image {
            return false;
        }
        let before = self.items.clone();
        let mut last_idx = None;
        for a in &self.clipboard {
            let mut copy = a.clone();
            if copy.cls >= num_classes {
                copy.cls = self.current_class;
            }
            self.items.push(copy);
            last_idx = Some(self.items.len() - 1);
        }
        self.selected = last_idx;
        self.commit(before, "貼上");
        true
    }

    pub fn clamp(&self, pt: [f64; 2]) -> [f64; 2] {
        [
            pt[0].clamp(0.0, self.img_w as f64),
            pt[1].clamp(0.0, self.img_h as f64),
        ]
    }

    pub fn to_local(a: &Annotation, pt: [f64; 2]) -> [f64; 2] {
        let dx = pt[0] - a.cx;
        let dy = pt[1] - a.cy;
        let (s, c) = a.angle.to_radians().sin_cos();
        [dx * c + dy * s, -dx * s + dy * c]
    }

    pub fn handle_at(&self, idx: usize, pt: [f64; 2]) -> Option<HandleId> {
        let a = self.items.get(idx)?;
        let lp = Self::to_local(a, pt);
        let px = self.px_per_screen_px();
        let tol = (HANDLE_PX as f64) * px;
        if a.kind == BoxKind::Obb {
            let rh_y = -a.h / 2.0 - (ROT_OFFSET_PX as f64) * px;
            if lp[0].hypot(lp[1] - rh_y) <= tol {
                return Some(HandleId::Rot);
            }
        }
        let w2 = a.w / 2.0;
        let h2 = a.h / 2.0;
        for (hid, sx, sy) in HandleId::RESIZE_HANDLES {
            if (lp[0] - sx * w2).abs() <= tol * 0.75 && (lp[1] - sy * h2).abs() <= tol * 0.75 {
                return Some(hid);
            }
        }
        None
    }

    pub fn item_at(&self, pt: [f64; 2]) -> Option<usize> {
        let mut best: Option<usize> = None;
        let mut best_area = f64::INFINITY;
        for (i, a) in self.items.iter().enumerate() {
            if !self.box_visible(i) {
                continue;
            }
            let lp = Self::to_local(a, pt);
            if lp[0].abs() <= a.w / 2.0 && lp[1].abs() <= a.h / 2.0 {
                let area = a.w * a.h;
                if best.is_none() || area < best_area {
                    best = Some(i);
                    best_area = area;
                }
            }
        }
        best
    }

    pub fn apply_resize(
        &mut self,
        idx: usize,
        handle: HandleId,
        orig: &Annotation,
        sp: [f64; 2],
    ) {
        let Some(a) = self.items.get_mut(idx) else {
            return;
        };
        let (sx, sy) = handle.signs();
        let (s, c) = orig.angle.to_radians().sin_cos();
        let vx = sp[0] - orig.cx;
        let vy = sp[1] - orig.cy;
        let mut qx = vx * c + vy * s;
        let mut qy = -vx * s + vy * c;
        let ax = -sx * orig.w / 2.0;
        let ay = -sy * orig.h / 2.0;
        if orig.kind == BoxKind::Bbox {
            qx = qx.clamp(-orig.cx, self.img_w as f64 - orig.cx);
            qy = qy.clamp(-orig.cy, self.img_h as f64 - orig.cy);
        }
        let (w, lcx) = if sx != 0.0 {
            ((qx - ax).abs().max(MIN_BOX), (qx + ax) / 2.0)
        } else {
            (orig.w, 0.0)
        };
        let (h, lcy) = if sy != 0.0 {
            ((qy - ay).abs().max(MIN_BOX), (qy + ay) / 2.0)
        } else {
            (orig.h, 0.0)
        };
        a.w = w;
        a.h = h;
        a.cx = orig.cx + lcx * c - lcy * s;
        a.cy = orig.cy + lcx * s + lcy * c;
    }

    pub fn finish_drag(&mut self) -> bool {
        let Some(d) = self.drag.take() else {
            return false;
        };
        match d {
            DragState::Draw { start_img, end_img } => {
                let w = (end_img[0] - start_img[0]).abs();
                let h = (end_img[1] - start_img[1]).abs();
                if w >= MIN_BOX && h >= MIN_BOX {
                    let before = self.items.clone();
                    let kind = if self.mode == ToolMode::Obb {
                        BoxKind::Obb
                    } else {
                        BoxKind::Bbox
                    };
                    let ann = Annotation::new(
                        self.current_class,
                        (start_img[0] + end_img[0]) / 2.0,
                        (start_img[1] + end_img[1]) / 2.0,
                        w,
                        h,
                        0.0,
                        kind,
                    );
                    self.items.push(ann);
                    self.selected = Some(self.items.len() - 1);
                    self.commit(before, "新增框");
                    return true;
                }
                false
            }
            DragState::Move { idx, orig, before, .. } => {
                if let Some(cur) = self.items.get(idx) {
                    if *cur != orig {
                        self.commit(before, "移動");
                        return true;
                    }
                }
                false
            }
            DragState::Resize { idx, orig, before, .. } => {
                if let Some(cur) = self.items.get(idx) {
                    if *cur != orig {
                        self.commit(before, "縮放");
                        return true;
                    }
                }
                false
            }
            DragState::Rotate { idx, orig, before } => {
                if let Some(cur) = self.items.get(idx) {
                    if *cur != orig {
                        self.commit(before, "旋轉");
                        return true;
                    }
                }
                false
            }
            DragState::Pan { .. } => false,
        }
    }
}

fn rotated_cursor(handle: HandleId, angle: f64) -> CursorIcon {
    let (sx, sy) = handle.signs();
    let base = sy.atan2(sx).to_degrees() + angle;
    let a = base.rem_euclid(180.0);
    if a < 22.5 || a >= 157.5 {
        CursorIcon::ResizeHorizontal
    } else if a < 67.5 {
        CursorIcon::ResizeNwSe
    } else if a < 112.5 {
        CursorIcon::ResizeVertical
    } else {
        CursorIcon::ResizeNeSw
    }
}

fn draw_dashed_line(
    painter: &egui::Painter,
    p0: Pos2,
    p1: Pos2,
    stroke: Stroke,
    dash: f32,
    gap: f32,
) {
    let d = p1 - p0;
    let len = d.length();
    if len <= 1e-3 {
        return;
    }
    let dir = d / len;
    let mut cur = 0.0;
    while cur < len {
        let end = (cur + dash).min(len);
        painter.line_segment([p0 + dir * cur, p0 + dir * end], stroke);
        cur += dash + gap;
    }
}

fn draw_dashed_rect(painter: &egui::Painter, rect: Rect, stroke: Stroke) {
    let tl = rect.left_top();
    let tr = rect.right_top();
    let br = rect.right_bottom();
    let bl = rect.left_bottom();
    draw_dashed_line(painter, tl, tr, stroke, 6.0, 4.0);
    draw_dashed_line(painter, tr, br, stroke, 6.0, 4.0);
    draw_dashed_line(painter, br, bl, stroke, 6.0, 4.0);
    draw_dashed_line(painter, bl, tl, stroke, 6.0, 4.0);
}

pub struct CanvasOutcome {
    pub changed: bool,
    pub selection_changed: Option<usize>,
    pub prev_selected: Option<usize>,
}

/// Render the annotation canvas and handle mouse/wheel/drag interactions.
pub fn render_canvas(
    ui: &mut egui::Ui,
    state: &mut CanvasState,
    classes: &[ClassDef],
    texture: Option<&TextureHandle>,
    pal: &Palette,
    modal_open: bool,
) -> CanvasOutcome {
    let prev_selected = state.selected;
    let mut changed = false;

    let avail = ui.available_size();
    let (rect, response) = ui.allocate_exact_size(avail, Sense::click_and_drag());
    let vp_size = [rect.width(), rect.height()];
    let resized = (vp_size[0] - state.last_vp_size[0]).abs() > 0.5
        || (vp_size[1] - state.last_vp_size[1]).abs() > 0.5;
    state.last_vp_size = vp_size;
    if resized && state.auto_fit && state.has_image {
        state.fit_view(vp_size);
    }

    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, CornerRadius::ZERO, pal.canvas);

    let img_to_screen = |pt: [f64; 2], st: &CanvasState| -> Pos2 {
        Pos2::new(
            rect.min.x + st.pan[0] + (pt[0] as f32) * st.zoom,
            rect.min.y + st.pan[1] + (pt[1] as f32) * st.zoom,
        )
    };
    let screen_to_img = |pos: Pos2, st: &CanvasState| -> [f64; 2] {
        [
            ((pos.x - rect.min.x - st.pan[0]) / st.zoom.max(1e-5)) as f64,
            ((pos.y - rect.min.y - st.pan[1]) / st.zoom.max(1e-5)) as f64,
        ]
    };

    // Input handling
    if !modal_open {
        let pointer_pos = ui.input(|i| i.pointer.hover_pos());
        let in_canvas = pointer_pos.map(|p| rect.contains(p)).unwrap_or(false);
        let mods = ui.input(|i| i.modifiers);
        let space_down = ui.input(|i| i.key_down(Key::Space));
        state.space_held = space_down;

        if let Some(pos) = pointer_pos {
            if in_canvas {
                state.cursor_img = Some(screen_to_img(pos, state));
            } else {
                state.cursor_img = None;
            }
        } else {
            state.cursor_img = None;
        }

        // Scroll & Ctrl+wheel zoom
        if in_canvas && state.has_image {
            let scroll = ui.input(|i| i.raw_scroll_delta);
            if scroll.x != 0.0 || scroll.y != 0.0 {
                if mods.ctrl {
                    let factor = 1.0025_f32.powf(scroll.y);
                    let cur = state.zoom;
                    let new_zoom = (cur * factor).clamp(0.02, 60.0);
                    if let Some(pos) = pointer_pos {
                        let vp_p = [pos.x - rect.min.x, pos.y - rect.min.y];
                        let before_x = (vp_p[0] - state.pan[0]) / cur;
                        let before_y = (vp_p[1] - state.pan[1]) / cur;
                        state.zoom = new_zoom;
                        state.pan = [vp_p[0] - before_x * new_zoom, vp_p[1] - before_y * new_zoom];
                        state.auto_fit = false;
                    }
                } else {
                    state.auto_fit = false;
                    let (dx, dy) = if mods.shift {
                        (if scroll.x != 0.0 { scroll.x } else { scroll.y }, 0.0)
                    } else {
                        (scroll.x, scroll.y)
                    };
                    state.pan[0] += dx;
                    state.pan[1] += dy;
                }
            }
        }

        // Mouse presses
        let primary_pressed = ui.input(|i| i.pointer.button_pressed(PointerButton::Primary));
        let middle_pressed = ui.input(|i| i.pointer.button_pressed(PointerButton::Middle));
        let secondary_pressed = ui.input(|i| i.pointer.button_pressed(PointerButton::Secondary));
        let double_clicked = response.double_clicked();

        if in_canvas {
            if let Some(pos) = pointer_pos {
                let sp = screen_to_img(pos, state);
                if middle_pressed || (primary_pressed && state.space_held) {
                    state.auto_fit = false;
                    state.drag = Some(DragState::Pan {
                        start_pointer: [pos.x, pos.y],
                        start_pan: state.pan,
                    });
                } else if state.has_image && secondary_pressed {
                    state.selected = state.item_at(sp);
                } else if state.has_image && double_clicked {
                    if let Some(hit) = state.item_at(sp) {
                        state.selected = Some(hit);
                    }
                } else if state.has_image && primary_pressed {
                    let mut started = false;
                    if let Some(sel) = state.selected {
                        if let Some(h) = state.handle_at(sel, sp) {
                            let orig = state.items[sel].clone();
                            let before = state.items.clone();
                            state.drag = Some(if h == HandleId::Rot {
                                DragState::Rotate {
                                    idx: sel,
                                    orig,
                                    before,
                                }
                            } else {
                                DragState::Resize {
                                    idx: sel,
                                    handle: h,
                                    orig,
                                    before,
                                }
                            });
                            started = true;
                        }
                    }
                    if !started && (state.mode == ToolMode::Select || mods.ctrl) {
                        let hit = state.item_at(sp);
                        state.selected = hit;
                        if let Some(idx) = hit {
                            let orig = state.items[idx].clone();
                            let before = state.items.clone();
                            state.drag = Some(DragState::Move {
                                idx,
                                start_img: sp,
                                orig,
                                before,
                            });
                        }
                        started = true;
                    }
                    if !started {
                        let p = state.clamp(sp);
                        state.selected = None;
                        state.drag = Some(DragState::Draw {
                            start_img: p,
                            end_img: p,
                        });
                    }
                }
            }
        }

        // Ongoing drag update
        if let Some(pos) = pointer_pos {
            let sp = screen_to_img(pos, state);
            if let Some(mut d) = state.drag.take() {
                let still_down = ui.input(|i| {
                    i.pointer.button_down(PointerButton::Primary)
                        || i.pointer.button_down(PointerButton::Middle)
                });
                match &mut d {
                    DragState::Pan {
                        start_pointer,
                        start_pan,
                    } => {
                        state.pan = [
                            start_pan[0] + (pos.x - start_pointer[0]),
                            start_pan[1] + (pos.y - start_pointer[1]),
                        ];
                    }
                    DragState::Draw { end_img, .. } => {
                        *end_img = state.clamp(sp);
                    }
                    DragState::Move {
                        idx,
                        start_img,
                        orig,
                        ..
                    } => {
                        if let Some(a) = state.items.get_mut(*idx) {
                            let dx = sp[0] - start_img[0];
                            let dy = sp[1] - start_img[1];
                            a.cx = orig.cx + dx;
                            a.cy = orig.cy + dy;
                            if a.kind == BoxKind::Bbox {
                                let fw = state.img_w as f64;
                                let fh = state.img_h as f64;
                                if a.w <= fw {
                                    a.cx = a.cx.clamp(a.w / 2.0, fw - a.w / 2.0);
                                }
                                if a.h <= fh {
                                    a.cy = a.cy.clamp(a.h / 2.0, fh - a.h / 2.0);
                                }
                            }
                        }
                    }
                    DragState::Resize {
                        idx,
                        handle,
                        orig,
                        ..
                    } => {
                        state.apply_resize(*idx, *handle, orig, sp);
                    }
                    DragState::Rotate { idx, .. } => {
                        if let Some(a) = state.items.get_mut(*idx) {
                            let mut ang = (sp[1] - a.cy).atan2(sp[0] - a.cx).to_degrees() + 90.0;
                            if mods.shift {
                                ang = (ang / 15.0).round() * 15.0;
                            }
                            a.angle = norm180(ang);
                        }
                    }
                }
                state.drag = Some(d);
                if !still_down {
                    changed |= state.finish_drag();
                }
            } else if in_canvas {
                state.hovered = if state.mode == ToolMode::Select && state.has_image {
                    state.item_at(sp)
                } else {
                    None
                };
            }
        } else if !ui.input(|i| i.pointer.any_down()) && state.drag.is_some() {
            changed |= state.finish_drag();
        }

        // Cursor icon
        if in_canvas {
            let icon = if state.space_held
                || matches!(state.drag, Some(DragState::Pan { .. }))
            {
                if state.drag.is_some() {
                    CursorIcon::Grabbing
                } else {
                    CursorIcon::Grab
                }
            } else if let (Some(sel), Some(sp)) = (state.selected, state.cursor_img) {
                match state.handle_at(sel, sp) {
                    Some(HandleId::Rot) => CursorIcon::PointingHand,
                    Some(h) => {
                        let ang = state.items.get(sel).map(|a| a.angle).unwrap_or(0.0);
                        rotated_cursor(h, ang)
                    }
                    None => {
                        if state.mode == ToolMode::Select {
                            if state.item_at(sp).is_some() {
                                CursorIcon::Move
                            } else {
                                CursorIcon::Default
                            }
                        } else {
                            CursorIcon::Crosshair
                        }
                    }
                }
            } else if state.mode == ToolMode::Select {
                if state.cursor_img.and_then(|sp| state.item_at(sp)).is_some() {
                    CursorIcon::Move
                } else {
                    CursorIcon::Default
                }
            } else {
                CursorIcon::Crosshair
            };
            ui.ctx().set_cursor_icon(icon);
        }
    }

    // Rendering
    if !state.has_image {
        let c = rect.center();
        let box_rect = Rect::from_center_size(c, Vec2::new(340.0, 180.0));
        draw_dashed_rect(&painter, box_rect, Stroke::new(2.0_f32, pal.border));
        painter.text(
            Pos2::new(c.x, c.y - 26.0),
            Align2::CENTER_CENTER,
            "開始標註",
            FontId::proportional(20.0),
            pal.text,
        );
        painter.text(
            Pos2::new(c.x, c.y + 24.0),
            Align2::CENTER_CENTER,
            "按 Ctrl+O 開啟圖片資料夾\nW 畫 BBox · E 畫旋轉框 · F1 快捷鍵",
            FontId::proportional(13.0),
            pal.muted,
        );
        return CanvasOutcome {
            changed,
            selection_changed: state.selected,
            prev_selected,
        };
    }

    let img_tl = img_to_screen([0.0, 0.0], state);
    let img_br = img_to_screen([state.img_w as f64, state.img_h as f64], state);
    let img_rect = Rect::from_min_max(img_tl, img_br);
    if let Some(tex) = texture {
        painter.image(
            tex.id(),
            img_rect,
            Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
            Color32::WHITE,
        );
    }
    painter.rect_stroke(
        img_rect,
        CornerRadius::ZERO,
        Stroke::new(1.0_f32, Color32::from_rgb(150, 150, 150)),
        egui::StrokeKind::Outside,
    );

    // Draw boxes (non-selected first, selected on top)
    let order = (0..state.items.len()).filter(|&i| state.selected != Some(i));
    let indices: Vec<usize> = order.chain(state.selected.into_iter()).collect();

    for i in indices {
        if i >= state.items.len() || !state.box_visible(i) {
            continue;
        }
        let a = &state.items[i];
        let hex = classes
            .get(a.cls)
            .map(|c| c.color.as_str())
            .unwrap_or("#ff00ff");
        let color = if a.cls < classes.len() {
            parse_hex_color(hex)
        } else {
            parse_hex_color(&default_color(a.cls))
        };
        let is_selected = state.selected == Some(i);
        let is_hovered = state.hovered == Some(i);
        let alpha = if is_selected {
            70
        } else if is_hovered {
            45
        } else {
            18
        };
        let fill = Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), alpha);
        let stroke = Stroke::new(if is_selected { 3.0_f32 } else { 2.0_f32 }, color);

        let corners = a.corners();
        let sc: Vec<Pos2> = corners.iter().map(|&p| img_to_screen(p, state)).collect();
        painter.add(Shape::convex_polygon(sc.clone(), fill, stroke));

        if a.kind == BoxKind::Obb {
            let top_mid = Pos2::new((sc[0].x + sc[1].x) * 0.5, (sc[0].y + sc[1].y) * 0.5);
            let bot_mid = Pos2::new((sc[2].x + sc[3].x) * 0.5, (sc[2].y + sc[3].y) * 0.5);
            let dir = (bot_mid - top_mid).normalized();
            let tick_len = ((a.h as f32 * state.zoom) * 0.5).min(10.0);
            painter.line_segment(
                [top_mid, top_mid + dir * tick_len],
                Stroke::new(2.0_f32, color),
            );
        }

        if state.label_visible(i) {
            let cname = classes
                .get(a.cls)
                .map(|c| c.name.clone())
                .unwrap_or_else(|| format!("#{}", a.cls));
            let label_text = if a.kind == BoxKind::Obb {
                format!("{cname}  {:.1}°", a.angle)
            } else {
                cname
            };
            let font = FontId::proportional(12.0);
            let galley = painter.layout_no_wrap(
                label_text,
                font,
                if color_lightness(color) > 140.0 {
                    Color32::BLACK
                } else {
                    Color32::WHITE
                },
            );
            let anchor = sc[0];
            let pad_x = 4.0;
            let th = galley.size().y + 3.0;
            let tw = galley.size().x + pad_x * 2.0;
            let bg_rect = Rect::from_min_size(Pos2::new(anchor.x, anchor.y - th), Vec2::new(tw, th));
            let bg = Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 210);
            painter.rect_filled(bg_rect, CornerRadius::ZERO, bg);
            painter.galley(
                Pos2::new(anchor.x + pad_x, anchor.y - th + 1.5),
                galley,
                Color32::WHITE,
            );
        }

        if is_selected {
            let px = state.px_per_screen_px();
            if a.kind == BoxKind::Obb {
                let (s, c) = a.angle.to_radians().sin_cos();
                let lx = 0.0_f64;
                let ly = -a.h / 2.0 - (ROT_OFFSET_PX as f64) * px;
                let rh_img = [lx * c - ly * s + a.cx, lx * s + ly * c + a.cy];
                let rh = img_to_screen(rh_img, state);
                let top_mid = Pos2::new((sc[0].x + sc[1].x) * 0.5, (sc[0].y + sc[1].y) * 0.5);
                painter.line_segment([top_mid, rh], Stroke::new(1.0_f32, color));
                painter.circle(
                    rh,
                    HANDLE_PX * 0.7,
                    parse_hex_color("#ffcc00"),
                    Stroke::new(1.0_f32, Color32::WHITE),
                );
            }
            let (s, c) = a.angle.to_radians().sin_cos();
            let hw = a.w / 2.0;
            let hh = a.h / 2.0;
            for (_, sx, sy) in HandleId::RESIZE_HANDLES {
                let lx = sx * hw;
                let ly = sy * hh;
                let hp_img = [lx * c - ly * s + a.cx, lx * s + ly * c + a.cy];
                let hp = img_to_screen(hp_img, state);
                let hrect = Rect::from_center_size(hp, Vec2::splat(HANDLE_PX));
                painter.rect(
                    hrect,
                    CornerRadius::ZERO,
                    color,
                    Stroke::new(1.0_f32, Color32::WHITE),
                    egui::StrokeKind::Middle,
                );
            }
        }
    }

    // Draw preview rectangle
    if let Some(DragState::Draw { start_img, end_img }) = state.drag {
        let p0 = img_to_screen(start_img, state);
        let p1 = img_to_screen(end_img, state);
        let drect = Rect::from_two_pos(p0, p1);
        let col = classes
            .get(state.current_class)
            .map(|c| parse_hex_color(&c.color))
            .unwrap_or_else(|| parse_hex_color(&default_color(state.current_class)));
        draw_dashed_rect(&painter, drect, Stroke::new(2.0_f32, col));
    }

    // Draw crosshair in draw mode
    if state.mode != ToolMode::Select
        && !matches!(state.drag, Some(DragState::Pan { .. }))
    {
        if let Some(c) = state.cursor_img {
            if c[0] >= 0.0
                && c[0] <= state.img_w as f64
                && c[1] >= 0.0
                && c[1] <= state.img_h as f64
            {
                let cp = img_to_screen(c, state);
                let stroke = Stroke::new(1.0_f32, Color32::from_rgba_unmultiplied(255, 255, 255, 160));
                draw_dashed_line(
                    &painter,
                    Pos2::new(img_rect.min.x, cp.y),
                    Pos2::new(img_rect.max.x, cp.y),
                    stroke,
                    5.0,
                    4.0,
                );
                draw_dashed_line(
                    &painter,
                    Pos2::new(cp.x, img_rect.min.y),
                    Pos2::new(cp.x, img_rect.max.y),
                    stroke,
                    5.0,
                    4.0,
                );
            }
        }
    }

    CanvasOutcome {
        changed,
        selection_changed: state.selected,
        prev_selected,
    }
}
