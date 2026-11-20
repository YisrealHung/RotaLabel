use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use egui::{
    Align2, Color32, ColorImage, CornerRadius, FontId, Key, Pos2, Rect, ScrollArea, Stroke,
    TextureHandle, TextureOptions, Vec2,
};
use image::RgbImage;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::augment::mosaic::{
    build_pools, generate_mosaic, make_mosaic, mosaic_splits, MosaicConfig,
};
use crate::augment::pipeline::AugConfig;
use crate::augment::version::{
    default_workers, generate_version, make_sample, next_version_dir, source_names, split_copies,
    split_items, VersionConfig, VersionStats, MAX_MULTIPLIER, TEST_TARGET,
};
use crate::exporters::FORMAT_OPTIONS;
use crate::imageio;
use crate::model::{default_color, Annotation, ClassDef};
use crate::project_io::Project;
use crate::ui::theme::{color_lightness, parse_hex_color, Palette};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TabKind {
    Normal,
    Mosaic,
}

#[derive(Debug, Clone)]
struct ParamRow {
    enabled: bool,
    value: f64,
}

impl ParamRow {
    fn from_val(v: f64, default: f64) -> Self {
        Self {
            enabled: v > 0.0,
            value: if v > 0.0 { v } else { default },
        }
    }
    fn effective(&self) -> f64 {
        if self.enabled {
            self.value
        } else {
            0.0
        }
    }
}

struct WorkerProgress {
    done: usize,
    total: usize,
    current_name: String,
    finished: Option<Result<VersionStats, String>>,
    out_dir: PathBuf,
}

struct PreviewTile {
    tex: TextureHandle,
    img_w: u32,
    img_h: u32,
    boxes: Vec<Annotation>,
}

const GEOMETRIC_SPEC: [(&str, &str, &str, f64, f64, usize); 6] = [
    ("rotate", "任意角度旋轉", "± 度", 180.0, 15.0, 0),
    ("shear", "剪切 Shear", "± 度", 45.0, 10.0, 0),
    ("scale", "縮放", "± %", 50.0, 20.0, 0),
    ("crop", "隨機裁切 (放大)", "最多 %", 80.0, 20.0, 0),
    ("translate_x", "左右平移", "± px", 4096.0, 50.0, 0),
    ("translate_y", "上下平移", "± px", 4096.0, 50.0, 0),
];

const PHOTOMETRIC_SPEC: [(&str, &str, &str, f64, f64, usize); 8] = [
    ("brightness", "亮度", "± %", 100.0, 25.0, 0),
    ("exposure", "曝光", "± %", 100.0, 15.0, 0),
    ("contrast", "對比", "± %", 100.0, 20.0, 0),
    ("saturation", "飽和度", "± %", 100.0, 25.0, 0),
    ("hue", "色相", "± 度", 180.0, 15.0, 0),
    ("grayscale", "灰階", "機率 %", 100.0, 15.0, 0),
    ("blur", "模糊", "最多 px", 10.0, 1.5, 1),
    ("noise", "雜訊", "最多 % 像素", 25.0, 2.0, 1),
];

pub struct VersionDialog {
    tab: TabKind,
    // Normal fields
    train: f64,
    valid: f64,
    test: f64,
    seed: u64,
    skip_unannotated: bool,
    resize_mode: String,
    resize_w: u32,
    resize_h: u32,
    grayscale: bool,
    flip_h: bool,
    flip_v: bool,
    rot90: bool,
    translate_allow_out: bool,
    geo_params: BTreeMap<&'static str, ParamRow>,
    photo_params: BTreeMap<&'static str, ParamRow>,
    multiplier: usize,
    min_visibility: f64,
    formats: BTreeMap<&'static str, bool>,
    workers: usize,

    // Mosaic fields
    canvas_w: u32,
    canvas_h: u32,
    count_train: usize,
    count_valid: usize,
    count_test: usize,
    patches: usize,
    patches_jitter: usize,
    allow_overlap: bool,
    min_fit_scale: f64,
    padding: i32,
    gap: i32,
    m_flip_h: bool,
    m_flip_v: bool,
    m_rot90: bool,
    m_rotate: ParamRow,
    m_scale: ParamRow,
    m_photo: BTreeMap<&'static str, ParamRow>,
    photometric_scope: String,

    // Preview state
    preview_seed: u64,
    preview_name: Option<String>,
    preview_img: Option<RgbImage>,
    pool_cache: Option<(i32, Vec<(RgbImage, Vec<Annotation>)>)>,
    preview_tiles: Vec<PreviewTile>,
    preview_dirty: bool,

    // Generation worker
    worker: Option<Arc<Mutex<WorkerProgress>>>,
    cancel_flag: Arc<AtomicBool>,
    result_popup: Option<(Result<VersionStats, String>, PathBuf)>,
    warn_msg: Option<String>,
}

impl VersionDialog {
    pub fn new(project: &Project) -> Self {
        let vcfg: VersionConfig = project
            .settings
            .get("version_config")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();
        let mcfg: MosaicConfig = project
            .settings
            .get("mosaic_config")
            .map(MosaicConfig::from_json_value)
            .unwrap_or_default();
        let tab = match project.settings.get("version_tab").and_then(|v| v.as_str()) {
            Some("mosaic") => TabKind::Mosaic,
            _ => TabKind::Normal,
        };

        let mut geo_params = BTreeMap::new();
        geo_params.insert("rotate", ParamRow::from_val(vcfg.aug.rotate, 15.0));
        geo_params.insert("shear", ParamRow::from_val(vcfg.aug.shear, 10.0));
        geo_params.insert("scale", ParamRow::from_val(vcfg.aug.scale, 20.0));
        geo_params.insert("crop", ParamRow::from_val(vcfg.aug.crop, 20.0));
        geo_params.insert("translate_x", ParamRow::from_val(vcfg.aug.translate_x, 50.0));
        geo_params.insert("translate_y", ParamRow::from_val(vcfg.aug.translate_y, 50.0));

        let mut photo_params = BTreeMap::new();
        let mut m_photo = BTreeMap::new();
        for (key, _, _, _, def, _) in PHOTOMETRIC_SPEC {
            let vv = aug_field(&vcfg.aug, key);
            let mv = aug_field(&mcfg.aug, key);
            photo_params.insert(key, ParamRow::from_val(vv, def));
            m_photo.insert(key, ParamRow::from_val(mv, def));
        }

        let mut formats = BTreeMap::new();
        for (key, _) in FORMAT_OPTIONS {
            formats.insert(key, vcfg.formats.iter().any(|f| f == key));
        }

        let mut dlg = Self {
            tab,
            train: vcfg.train,
            valid: vcfg.valid,
            test: vcfg.test,
            seed: vcfg.seed,
            skip_unannotated: vcfg.skip_unannotated,
            resize_mode: vcfg.resize_mode,
            resize_w: vcfg.resize_w,
            resize_h: vcfg.resize_h,
            grayscale: vcfg.grayscale,
            flip_h: vcfg.aug.flip_h,
            flip_v: vcfg.aug.flip_v,
            rot90: vcfg.aug.rot90,
            translate_allow_out: vcfg.aug.translate_allow_out,
            geo_params,
            photo_params,
            multiplier: vcfg.multiplier.clamp(1, MAX_MULTIPLIER),
            min_visibility: vcfg.min_visibility,
            formats,
            workers: vcfg.workers,

            canvas_w: mcfg.canvas_w,
            canvas_h: mcfg.canvas_h,
            count_train: mcfg.count_train,
            count_valid: mcfg.count_valid,
            count_test: mcfg.count_test,
            patches: mcfg.patches,
            patches_jitter: mcfg.patches_jitter,
            allow_overlap: mcfg.allow_overlap,
            min_fit_scale: mcfg.min_fit_scale,
            padding: mcfg.padding,
            gap: mcfg.gap,
            m_flip_h: mcfg.flip_h,
            m_flip_v: mcfg.flip_v,
            m_rot90: mcfg.rot90,
            m_rotate: ParamRow::from_val(mcfg.rotate, 15.0),
            m_scale: ParamRow::from_val(mcfg.scale, 20.0),
            m_photo,
            photometric_scope: mcfg.photometric_scope,

            preview_seed: 0,
            preview_name: None,
            preview_img: None,
            pool_cache: None,
            preview_tiles: Vec::new(),
            preview_dirty: true,

            worker: None,
            cancel_flag: Arc::new(AtomicBool::new(false)),
            result_popup: None,
            warn_msg: None,
        };
        dlg.next_preview_image(project);
        dlg
    }

    pub fn build_version_config(&self) -> VersionConfig {
        let mut aug = AugConfig {
            flip_h: self.flip_h,
            flip_v: self.flip_v,
            rot90: self.rot90,
            rotate: self.geo_params["rotate"].effective(),
            shear: self.geo_params["shear"].effective(),
            scale: self.geo_params["scale"].effective(),
            crop: self.geo_params["crop"].effective(),
            translate_x: self.geo_params["translate_x"].effective(),
            translate_y: self.geo_params["translate_y"].effective(),
            translate_allow_out: self.translate_allow_out,
            ..Default::default()
        };
        for (k, row) in &self.photo_params {
            set_aug_field(&mut aug, k, row.effective());
        }
        let formats: Vec<String> = FORMAT_OPTIONS
            .iter()
            .filter(|(k, _)| *self.formats.get(k).unwrap_or(&false))
            .map(|(k, _)| (*k).to_string())
            .collect();
        VersionConfig {
            train: self.train,
            valid: self.valid,
            test: self.test,
            seed: self.seed,
            skip_unannotated: self.skip_unannotated,
            resize_mode: self.resize_mode.clone(),
            resize_w: self.resize_w,
            resize_h: self.resize_h,
            grayscale: self.grayscale,
            multiplier: self.multiplier.clamp(1, MAX_MULTIPLIER),
            min_visibility: self.min_visibility,
            formats,
            workers: self.workers,
            aug,
        }
    }

    pub fn build_mosaic_config(&self) -> MosaicConfig {
        let mut aug = AugConfig::default();
        for (k, row) in &self.m_photo {
            set_aug_field(&mut aug, k, row.effective());
        }
        MosaicConfig {
            canvas_w: self.canvas_w,
            canvas_h: self.canvas_h,
            count_train: self.count_train,
            count_valid: self.count_valid,
            count_test: self.count_test,
            patches: self.patches,
            patches_jitter: self.patches_jitter,
            padding: self.padding,
            gap: self.gap,
            allow_overlap: self.allow_overlap,
            min_fit_scale: self.min_fit_scale,
            flip_h: self.m_flip_h,
            flip_v: self.m_flip_v,
            rot90: self.m_rot90,
            rotate: self.m_rotate.effective(),
            scale: self.m_scale.effective(),
            photometric_scope: self.photometric_scope.clone(),
            aug,
        }
    }

    pub fn save_settings(&self, project: &mut Project) {
        if let Ok(v) = serde_json::to_value(self.build_version_config()) {
            project.settings.insert("version_config".to_string(), v);
        }
        if let Ok(m) = serde_json::to_value(self.build_mosaic_config()) {
            project.settings.insert("mosaic_config".to_string(), m);
        }
        project.settings.insert(
            "version_tab".to_string(),
            serde_json::Value::String(
                if self.tab == TabKind::Mosaic {
                    "mosaic"
                } else {
                    "normal"
                }
                .to_string(),
            ),
        );
        project.save_meta();
    }

    pub fn is_running(&self) -> bool {
        if let Some(ref w) = self.worker {
            if let Ok(guard) = w.lock() {
                return guard.finished.is_none();
            }
        }
        false
    }

    fn next_preview_image(&mut self, project: &Project) {
        let mut names: Vec<&String> = project
            .images
            .iter()
            .filter(|n| project.box_count(n) > 0)
            .collect();
        if names.is_empty() {
            names = project.images.iter().collect();
        }
        if names.is_empty() {
            return;
        }
        let mut rng = rand::thread_rng();
        let pick = names[rng.gen_range(0..names.len())].clone();
        self.preview_img = imageio::imread(&project.image_path(&pick));
        self.preview_name = Some(pick);
        self.preview_dirty = true;
    }

    fn render_preview(&mut self, ctx: &egui::Context, project: &Project) {
        self.preview_tiles.clear();
        self.preview_dirty = false;
        if self.tab == TabKind::Mosaic {
            let mcfg = self.build_mosaic_config();
            let need_rebuild = self
                .pool_cache
                .as_ref()
                .map(|(pad, _)| *pad != mcfg.padding)
                .unwrap_or(true);
            if need_rebuild {
                let mut names: Vec<String> = project
                    .images
                    .iter()
                    .filter(|n| project.box_count(n) > 0)
                    .cloned()
                    .collect();
                if names.len() > 60 {
                    names.truncate(60);
                }
                let mut sp = std::collections::HashMap::new();
                sp.insert("train".to_string(), names);
                let pool = build_pools(project, &sp, mcfg.padding)
                    .remove("train")
                    .unwrap_or_default();
                self.pool_cache = Some((mcfg.padding, pool));
            }
            let pool = &self.pool_cache.as_ref().unwrap().1;
            if pool.is_empty() {
                return;
            }
            let mut rng = ChaCha8Rng::seed_from_u64(self.preview_seed);
            let (img, boxes) = make_mosaic(pool, &mcfg, &mut rng, self.grayscale);
            let tex = rgb_to_texture(ctx, "mosaic_prev", &img);
            self.preview_tiles.push(PreviewTile {
                tex,
                img_w: img.width(),
                img_h: img.height(),
                boxes,
            });
            return;
        }

        let Some(ref pname) = self.preview_name else {
            return;
        };
        let Some(ref pimg) = self.preview_img else {
            return;
        };
        let cfg = self.build_version_config();
        let anns = project
            .get(pname)
            .map(|ia| ia.annotations.as_slice())
            .unwrap_or(&[]);
        let mut rng = ChaCha8Rng::seed_from_u64(self.preview_seed);
        let aug_on = cfg.aug.any_enabled();
        for k in 0..4 {
            let (img, boxes) = make_sample(pimg, anns, &cfg, &mut rng, k > 0 && aug_on);
            let tex = rgb_to_texture(ctx, &format!("prev_{k}"), &img);
            self.preview_tiles.push(PreviewTile {
                tex,
                img_w: img.width(),
                img_h: img.height(),
                boxes,
            });
        }
    }

    fn start_generate(&mut self, ctx: &egui::Context, project: &mut Project) {
        let cfg = self.build_version_config();
        if cfg.formats.is_empty() {
            self.warn_msg = Some("請至少選擇一種匯出格式".to_string());
            return;
        }
        if cfg.train + cfg.valid + cfg.test <= 0.0 {
            self.warn_msg = Some("切分比例總和不可為 0".to_string());
            return;
        }
        self.save_settings(project);
        let out_dir = next_version_dir(project);
        let is_mosaic = self.tab == TabKind::Mosaic;
        let mcfg = self.build_mosaic_config();
        if is_mosaic {
            let sp = mosaic_splits(project, &cfg);
            let counts = mcfg.counts();
            let has_any = ["train", "valid", "test"]
                .iter()
                .any(|&s| !sp.get(s).map(|v| v.is_empty()).unwrap_or(true) && counts[s] > 0);
            if !has_any {
                self.warn_msg =
                    Some("沒有可拼接的切分：請確認有已標註的圖片，且張數不全為 0".to_string());
                return;
            }
        }

        self.cancel_flag.store(false, Ordering::Relaxed);
        let cancel = self.cancel_flag.clone();
        let prog = Arc::new(Mutex::new(WorkerProgress {
            done: 0,
            total: 1,
            current_name: String::new(),
            finished: None,
            out_dir: out_dir.clone(),
        }));
        self.worker = Some(prog.clone());

        let proj_clone = project.clone();
        let ctx_clone = ctx.clone();
        std::thread::spawn(move || {
            let cb = |done: usize, total: usize, name: &str| {
                if let Ok(mut g) = prog.lock() {
                    g.done = done;
                    g.total = total;
                    g.current_name = name.to_string();
                }
                ctx_clone.request_repaint();
            };
            let res = if is_mosaic {
                generate_mosaic(&proj_clone, &cfg, &mcfg, &out_dir, Some(&cb), Some(&cancel))
            } else {
                generate_version(&proj_clone, &cfg, &out_dir, Some(&cb), Some(&cancel))
            };
            if let Ok(mut g) = prog.lock() {
                g.finished = Some(res.map_err(|e| e.to_string()));
            }
            ctx_clone.request_repaint();
        });
    }

    /// Returns `false` if the dialog should close.
    pub fn show(&mut self, ctx: &egui::Context, project: &mut Project, pal: &Palette) -> bool {
        let mut keep_open = true;

        // Check worker completion
        if let Some(ref w) = self.worker {
            let done_opt = w.lock().ok().and_then(|g| {
                g.finished
                    .as_ref()
                    .map(|res| (res.clone(), g.out_dir.clone()))
            });
            if let Some((res, out_dir)) = done_opt {
                self.worker = None;
                self.result_popup = Some((res, out_dir));
            }
        }

        let running = self.is_running();
        if ctx.input(|i| i.key_pressed(Key::Escape)) {
            if self.result_popup.is_some() {
                self.result_popup = None;
            } else if self.warn_msg.is_some() {
                self.warn_msg = None;
            } else if running {
                self.cancel_flag.store(true, Ordering::Relaxed);
            } else {
                self.save_settings(project);
                return false;
            }
        }

        if self.preview_dirty {
            self.render_preview(ctx, project);
        }

        let screen = ctx.screen_rect();
        let win_w = (screen.width() - 60.0).clamp(800.0, 1280.0);
        let win_h = (screen.height() - 60.0).clamp(560.0, 860.0);

        let mut close_clicked = false;
        egui::Window::new("產生資料集版本")
            .collapsible(false)
            .resizable(true)
            .default_size([win_w, win_h])
            .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
            .show(ctx, |ui| {
                let mut changed = false;

                // Top area: left settings + right preview
                let avail_h = ui.available_height() - 44.0;
                ui.horizontal(|ui| {
                    // Left column
                    ui.allocate_ui_with_layout(
                        Vec2::new(540.0, avail_h),
                        egui::Layout::top_down(egui::Align::Min),
                        |ui| {
                            ui.horizontal(|ui| {
                                if ui
                                    .selectable_label(self.tab == TabKind::Normal, "一般擴增")
                                    .clicked()
                                {
                                    self.tab = TabKind::Normal;
                                    changed = true;
                                }
                                if ui
                                    .selectable_label(self.tab == TabKind::Mosaic, "拼接擴增")
                                    .clicked()
                                {
                                    self.tab = TabKind::Mosaic;
                                    changed = true;
                                }
                            });
                            ui.separator();

                            ScrollArea::vertical()
                                .id_salt("version_left_scroll")
                                .max_height(avail_h - 32.0)
                                .show(ui, |ui| {
                                    ui.add_enabled_ui(!running, |ui| {
                                        if self.tab == TabKind::Normal {
                                            changed |= self.ui_normal_tab(ui, pal);
                                        } else {
                                            changed |= self.ui_mosaic_tab(ui, pal);
                                        }
                                        ui.add_space(6.0);
                                        changed |= self.ui_output_group(ui, pal);
                                    });
                                });
                        },
                    );

                    ui.separator();

                    // Right column: preview
                    let rem_w = ui.available_width().max(200.0);
                    ui.allocate_ui_with_layout(
                        Vec2::new(rem_w, avail_h),
                        egui::Layout::top_down(egui::Align::Min),
                        |ui| {
                            ui.horizontal(|ui| {
                                ui.vertical(|ui| {
                                    ui.strong("擴增預覽");
                                    let hint = if self.tab == TabKind::Mosaic {
                                        "隨機拼接樣本（使用所有已標註圖片的圖塊）"
                                    } else {
                                        "左上為前處理後原圖，其餘為隨機擴增樣本"
                                    };
                                    ui.label(egui::RichText::new(hint).small().color(pal.muted));
                                });
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        if ui.button("重新隨機").clicked() {
                                            self.preview_seed += 1;
                                            self.preview_dirty = true;
                                        }
                                        if self.tab == TabKind::Normal && ui.button("換一張圖片").clicked()
                                        {
                                            self.next_preview_image(project);
                                        }
                                    },
                                );
                            });
                            ui.add_space(6.0);
                            self.ui_preview_canvas(ui, project, pal);
                        },
                    );
                });

                ui.separator();

                // Bottom bar
                ui.horizontal(|ui| {
                    if let Some(ref w) = self.worker {
                        if let Ok(g) = w.lock() {
                            let frac = if g.total > 0 {
                                g.done as f32 / g.total as f32
                            } else {
                                0.0
                            };
                            ui.add(
                                egui::ProgressBar::new(frac)
                                    .desired_width((ui.available_width() - 180.0).max(120.0))
                                    .text(format!("{}/{}  {}", g.done, g.total, g.current_name)),
                            );
                        }
                    } else {
                        let summary = self.summary_text(project);
                        ui.label(egui::RichText::new(summary).strong());
                    }

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let gen_btn = ui.add_enabled(
                            !running,
                            egui::Button::new(
                                egui::RichText::new("產生版本")
                                    .color(pal.accent_text)
                                    .strong(),
                            )
                            .fill(if running { pal.border } else { pal.accent }),
                        );
                        if gen_btn.clicked() {
                            self.start_generate(ctx, project);
                        }
                        if running {
                            if ui.button("取消").clicked() {
                                self.cancel_flag.store(true, Ordering::Relaxed);
                            }
                        } else if ui.button("關閉").clicked() {
                            close_clicked = true;
                        }
                    });
                });

                if changed {
                    self.preview_dirty = true;
                }
            });

        if close_clicked {
            self.save_settings(project);
            keep_open = false;
        }

        if let Some(msg) = self.warn_msg.clone() {
            egui::Window::new("產生版本")
                .collapsible(false)
                .resizable(false)
                .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
                .show(ctx, |ui| {
                    ui.label(msg);
                    if ui.button("確定").clicked() {
                        self.warn_msg = None;
                    }
                });
        }

        if let Some((res, out_dir)) = self.result_popup.clone() {
            let mut dismiss = false;
            egui::Window::new(if res.is_ok() { "產生版本完成" } else { "產生版本失敗" })
                .collapsible(false)
                .resizable(false)
                .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
                .show(ctx, |ui| match &res {
                    Ok(stats) if stats.cancelled => {
                        ui.label(format!("已取消（部分檔案留在 {}）", out_dir.display()));
                        if ui.button("確定").clicked() {
                            dismiss = true;
                        }
                    }
                    Ok(stats) => {
                        ui.label(format!(
                            "Train {} 張、Valid {} 張、Test {} 張，共 {} 個框。\n\n輸出位置：\n{}",
                            stats.train,
                            stats.valid,
                            stats.test,
                            stats.boxes,
                            out_dir.display()
                        ));
                        ui.add_space(6.0);
                        ui.horizontal(|ui| {
                            if ui.button("開啟資料夾").clicked() {
                                open_folder_in_os(&out_dir);
                                dismiss = true;
                            }
                            if ui.button("關閉").clicked() {
                                dismiss = true;
                            }
                        });
                    }
                    Err(e) => {
                        ui.label(e);
                        if ui.button("確定").clicked() {
                            dismiss = true;
                        }
                    }
                });
            if dismiss {
                self.result_popup = None;
            }
        }

        keep_open
    }

    fn ui_normal_tab(&mut self, ui: &mut egui::Ui, pal: &Palette) -> bool {
        let mut changed = false;

        ui.group(|ui| {
            ui.strong("1. 切分 Train / Valid / Test");
            ui.horizontal(|ui| {
                ui.label("比例 %");
                ui.label("Train");
                changed |= ui
                    .add(egui::DragValue::new(&mut self.train).range(0.0..=100.0))
                    .changed();
                ui.label("Valid");
                changed |= ui
                    .add(egui::DragValue::new(&mut self.valid).range(0.0..=100.0))
                    .changed();
                ui.label("Test");
                changed |= ui
                    .add(egui::DragValue::new(&mut self.test).range(0.0..=100.0))
                    .changed();
            });
            ui.horizontal(|ui| {
                ui.label("亂數種子");
                changed |= ui
                    .add(egui::DragValue::new(&mut self.seed).range(0..=999999))
                    .changed();
            });
            changed |= ui
                .checkbox(&mut self.skip_unannotated, "略過沒有標註的圖片")
                .changed();
            ui.label(
                egui::RichText::new(format!(
                    "有啟用擴增時：Train / Valid 依倍數擴增；Test 固定輸出 {TEST_TARGET} 張（不足時以擴增補足，超過時只取前 {TEST_TARGET} 張）。"
                ))
                .small()
                .color(pal.muted),
            );
        });

        ui.add_space(4.0);
        ui.group(|ui| {
            ui.strong("2. 前處理（所有切分）");
            ui.horizontal(|ui| {
                ui.label("Resize");
                let cur_label = match self.resize_mode.as_str() {
                    "stretch" => "拉伸至",
                    "fit" => "等比例縮放 + 黑邊補齊至",
                    _ => "不調整大小",
                };
                egui::ComboBox::from_id_salt("resize_mode_cmb")
                    .selected_text(cur_label)
                    .show_ui(ui, |ui| {
                        for (k, text) in [
                            ("none", "不調整大小"),
                            ("stretch", "拉伸至"),
                            ("fit", "等比例縮放 + 黑邊補齊至"),
                        ] {
                            if ui.selectable_value(&mut self.resize_mode, k.to_string(), text).changed() {
                                changed = true;
                            }
                        }
                    });
                let resize_on = self.resize_mode != "none";
                ui.add_enabled_ui(resize_on, |ui| {
                    changed |= ui
                        .add(egui::DragValue::new(&mut self.resize_w).range(16..=8192))
                        .changed();
                    ui.label("×");
                    changed |= ui
                        .add(egui::DragValue::new(&mut self.resize_h).range(16..=8192))
                        .changed();
                });
            });
            changed |= ui.checkbox(&mut self.grayscale, "轉灰階").changed();
            ui.label(
                egui::RichText::new("EXIF 方向一律自動校正 (auto-orient)。")
                    .small()
                    .color(pal.muted),
            );
        });

        ui.add_space(4.0);
        ui.group(|ui| {
            ui.strong("3. 幾何擴增 · 框會跟著轉換");
            changed |= ui.checkbox(&mut self.flip_h, "水平翻轉").changed();
            changed |= ui.checkbox(&mut self.flip_v, "垂直翻轉").changed();
            changed |= ui
                .checkbox(&mut self.rot90, "90° 旋轉（隨機 0/90/180/270）")
                .changed();
            egui::Grid::new("geo_aug_grid").num_columns(3).show(ui, |ui| {
                for (key, text, unit, mx, _def, dec) in GEOMETRIC_SPEC {
                    let row = self.geo_params.get_mut(key).unwrap();
                    changed |= ui.checkbox(&mut row.enabled, text).changed();
                    ui.add_enabled_ui(row.enabled, |ui| {
                        let lo = if dec > 0 { 0.1 } else { 1.0 };
                        changed |= ui
                            .add(
                                egui::DragValue::new(&mut row.value)
                                    .range(lo..=mx)
                                    .speed(if dec > 0 { 0.2 } else { 1.0 })
                                    .fixed_decimals(dec),
                            )
                            .changed();
                    });
                    ui.label(unit);
                    ui.end_row();
                }
            });
            let tx_on = self.geo_params["translate_x"].enabled || self.geo_params["translate_y"].enabled;
            ui.add_enabled_ui(tx_on, |ui| {
                changed |= ui
                    .checkbox(
                        &mut self.translate_allow_out,
                        "允許框超出邊界（超出部分裁掉，依「框保留門檻」決定是否保留）",
                    )
                    .changed();
            });
            ui.label(
                egui::RichText::new("不勾選時會自動限制平移量，讓所有框完整留在圖內。平移以前處理後的圖片像素計算。")
                    .small()
                    .color(pal.muted),
            );
        });

        ui.add_space(4.0);
        ui.group(|ui| {
            ui.strong("4. 色彩 / 像素擴增");
            egui::Grid::new("photo_aug_grid").num_columns(3).show(ui, |ui| {
                for (key, text, unit, mx, _def, dec) in PHOTOMETRIC_SPEC {
                    let row = self.photo_params.get_mut(key).unwrap();
                    changed |= ui.checkbox(&mut row.enabled, text).changed();
                    ui.add_enabled_ui(row.enabled, |ui| {
                        let lo = if dec > 0 { 0.1 } else { 1.0 };
                        changed |= ui
                            .add(
                                egui::DragValue::new(&mut row.value)
                                    .range(lo..=mx)
                                    .speed(if dec > 0 { 0.2 } else { 1.0 })
                                    .fixed_decimals(dec),
                            )
                            .changed();
                    });
                    ui.label(unit);
                    ui.end_row();
                }
            });
        });

        changed
    }

    fn ui_mosaic_tab(&mut self, ui: &mut egui::Ui, pal: &Palette) -> bool {
        let mut changed = false;

        ui.group(|ui| {
            ui.strong("1. 背景範圍與張數");
            ui.horizontal(|ui| {
                ui.label("背景範圍 寬 × 高");
                changed |= ui
                    .add(egui::DragValue::new(&mut self.canvas_w).range(64..=8192))
                    .changed();
                ui.label("×");
                changed |= ui
                    .add(egui::DragValue::new(&mut self.canvas_h).range(64..=8192))
                    .changed();
                ui.label("px");
            });
            ui.horizontal(|ui| {
                ui.label("產生張數");
                ui.label("Train");
                changed |= ui
                    .add(egui::DragValue::new(&mut self.count_train).range(0..=100000))
                    .changed();
                ui.label("Valid");
                changed |= ui
                    .add(egui::DragValue::new(&mut self.count_valid).range(0..=100000))
                    .changed();
                ui.label("Test");
                changed |= ui
                    .add(egui::DragValue::new(&mut self.count_test).range(0..=100000))
                    .changed();
            });
            ui.horizontal(|ui| {
                ui.label("每張拼接張數");
                changed |= ui
                    .add(egui::DragValue::new(&mut self.patches).range(1..=500))
                    .changed();
                ui.label("±");
                changed |= ui
                    .add(egui::DragValue::new(&mut self.patches_jitter).range(0..=500))
                    .changed();
                let lo = self.patches.saturating_sub(self.patches_jitter).max(1);
                let hi = lo.max(self.patches.saturating_add(self.patches_jitter));
                let range_text = if lo == hi {
                    format!("→ 每張 {lo} 張")
                } else {
                    format!("→ 每張 {lo} ~ {hi} 張隨機")
                };
                ui.label(range_text);
            });
            ui.label(
                egui::RichText::new("原圖不夠時會重複使用同一張，但每份複本的角度都不同（至少差 30°）。")
                    .small()
                    .color(pal.muted),
            );
            changed |= ui
                .checkbox(&mut self.allow_overlap, "允許圖塊背景互相重疊（標註框不會被蓋住）")
                .changed();
            ui.horizontal(|ui| {
                ui.label("最小縮放");
                changed |= ui
                    .add(egui::DragValue::new(&mut self.min_fit_scale).range(1.0..=100.0).fixed_decimals(0))
                    .changed();
                ui.label("%（放不下時最多縮到這麼小；100 = 不縮小）");
            });
            ui.horizontal(|ui| {
                ui.label("裁切外擴 px");
                changed |= ui
                    .add(egui::DragValue::new(&mut self.padding).range(0..=1000))
                    .changed();
                ui.label("間距 px");
                changed |= ui
                    .add(egui::DragValue::new(&mut self.gap).range(0..=1000))
                    .changed();
            });
            ui.label(
                egui::RichText::new("切分比例、亂數種子、轉灰階沿用「一般擴增」分頁。")
                    .small()
                    .color(pal.muted),
            );
        });

        ui.add_space(4.0);
        ui.group(|ui| {
            ui.strong("2. 圖塊幾何變化 · 框會跟著轉換");
            changed |= ui.checkbox(&mut self.m_flip_h, "水平翻轉").changed();
            changed |= ui.checkbox(&mut self.m_flip_v, "垂直翻轉").changed();
            changed |= ui
                .checkbox(&mut self.m_rot90, "90° 旋轉（隨機 0/90/180/270）")
                .changed();
            egui::Grid::new("m_geo_grid").num_columns(3).show(ui, |ui| {
                changed |= ui.checkbox(&mut self.m_rotate.enabled, "任意角度旋轉").changed();
                ui.add_enabled_ui(self.m_rotate.enabled, |ui| {
                    changed |= ui
                        .add(egui::DragValue::new(&mut self.m_rotate.value).range(1.0..=180.0).fixed_decimals(0))
                        .changed();
                });
                ui.label("± 度");
                ui.end_row();

                changed |= ui.checkbox(&mut self.m_scale.enabled, "縮放").changed();
                ui.add_enabled_ui(self.m_scale.enabled, |ui| {
                    changed |= ui
                        .add(egui::DragValue::new(&mut self.m_scale.value).range(1.0..=90.0).fixed_decimals(0))
                        .changed();
                });
                ui.label("± %");
                ui.end_row();
            });
        });

        ui.add_space(4.0);
        ui.group(|ui| {
            ui.strong("3. 色彩 / 像素擴增");
            egui::Grid::new("m_photo_grid").num_columns(3).show(ui, |ui| {
                for (key, text, unit, mx, _def, dec) in PHOTOMETRIC_SPEC {
                    let row = self.m_photo.get_mut(key).unwrap();
                    changed |= ui.checkbox(&mut row.enabled, text).changed();
                    ui.add_enabled_ui(row.enabled, |ui| {
                        let lo = if dec > 0 { 0.1 } else { 1.0 };
                        changed |= ui
                            .add(
                                egui::DragValue::new(&mut row.value)
                                    .range(lo..=mx)
                                    .speed(if dec > 0 { 0.2 } else { 1.0 })
                                    .fixed_decimals(dec),
                            )
                            .changed();
                    });
                    ui.label(unit);
                    ui.end_row();
                }
            });
            ui.horizontal(|ui| {
                ui.label("套用方式");
                let cur = if self.photometric_scope == "canvas" {
                    "整張拼接圖一起"
                } else {
                    "每個圖塊各自隨機"
                };
                egui::ComboBox::from_id_salt("m_scope_cmb")
                    .selected_text(cur)
                    .show_ui(ui, |ui| {
                        for (k, text) in [("patch", "每個圖塊各自隨機"), ("canvas", "整張拼接圖一起")]
                        {
                            if ui
                                .selectable_value(&mut self.photometric_scope, k.to_string(), text)
                                .changed()
                            {
                                changed = true;
                            }
                        }
                    });
            });
        });

        changed
    }

    fn ui_output_group(&mut self, ui: &mut egui::Ui, pal: &Palette) -> bool {
        let mut changed = false;
        ui.group(|ui| {
            ui.strong("5. 輸出");
            if self.tab == TabKind::Normal {
                ui.horizontal(|ui| {
                    ui.label("每張 Train / Valid 圖輸出數量（含原圖）");
                    changed |= ui
                        .add(egui::DragValue::new(&mut self.multiplier).range(1..=MAX_MULTIPLIER))
                        .changed();
                });
                ui.horizontal(|ui| {
                    ui.label("框保留門檻（擴增後至少保留 % 面積）");
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut self.min_visibility)
                                .range(0.0..=100.0)
                                .fixed_decimals(0),
                        )
                        .changed();
                });
            }
            ui.label("匯出格式");
            for (key, text) in FORMAT_OPTIONS {
                let checked = self.formats.get_mut(key).unwrap();
                changed |= ui.checkbox(checked, text).changed();
            }
            ui.horizontal(|ui| {
                ui.label("平行處理數");
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut self.workers)
                            .range(0..=256)
                            .custom_formatter(|v, _| {
                                if v == 0.0 {
                                    "自動".to_string()
                                } else {
                                    format!("{}", v as usize)
                                }
                            }),
                    )
                    .changed();
                let total_threads = std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1);
                ui.label(format!(
                    "（本機 {total_threads} 執行緒；自動 = {}）",
                    default_workers()
                ));
            });
            ui.label(
                egui::RichText::new("YOLO(bbox) 會把旋轉框轉成外接矩形；YOLO-OBB / DOTA / COCO-OBB 會把 bbox 視為 0° 旋轉框。")
                    .small()
                    .color(pal.muted),
            );
        });
        changed
    }

    fn ui_preview_canvas(&self, ui: &mut egui::Ui, project: &Project, pal: &Palette) {
        let avail = ui.available_size();
        let (rect, _) = ui.allocate_exact_size(avail, egui::Sense::hover());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, CornerRadius::same(10), pal.canvas);

        if self.preview_tiles.is_empty() {
            let msg = if self.tab == TabKind::Mosaic {
                "沒有已標註的圖片可以拼接"
            } else {
                "沒有可預覽的圖片"
            };
            painter.text(
                rect.center(),
                Align2::CENTER_CENTER,
                msg,
                FontId::proportional(14.0),
                pal.muted,
            );
            return;
        }

        if self.preview_tiles.len() == 1 {
            self.paint_tile(&painter, rect.shrink(6.0), &self.preview_tiles[0], &project.classes);
            return;
        }

        let gap = 6.0;
        let tile_w = ((rect.width() - gap * 3.0) * 0.5).max(10.0);
        let tile_h = ((rect.height() - gap * 3.0) * 0.5).max(10.0);
        for (i, tile) in self.preview_tiles.iter().enumerate() {
            let col = (i % 2) as f32;
            let row = (i / 2) as f32;
            let sub = Rect::from_min_size(
                Pos2::new(
                    rect.min.x + gap + col * (tile_w + gap),
                    rect.min.y + gap + row * (tile_h + gap),
                ),
                Vec2::new(tile_w, tile_h),
            );
            self.paint_tile(&painter, sub, tile, &project.classes);
        }
    }

    fn paint_tile(
        &self,
        painter: &egui::Painter,
        area: Rect,
        tile: &PreviewTile,
        classes: &[ClassDef],
    ) {
        if tile.img_w == 0 || tile.img_h == 0 {
            return;
        }
        let scale = (area.width() / tile.img_w as f32).min(area.height() / tile.img_h as f32);
        let dw = tile.img_w as f32 * scale;
        let dh = tile.img_h as f32 * scale;
        let img_rect = Rect::from_center_size(area.center(), Vec2::new(dw, dh));
        painter.image(
            tile.tex.id(),
            img_rect,
            Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
            Color32::WHITE,
        );
        for a in &tile.boxes {
            let hex = classes
                .get(a.cls)
                .map(|c| c.color.as_str())
                .unwrap_or("#00ff00");
            let col = if a.cls < classes.len() {
                parse_hex_color(hex)
            } else {
                parse_hex_color(&default_color(a.cls))
            };
            let sc: Vec<Pos2> = a
                .corners()
                .iter()
                .map(|[x, y]| {
                    Pos2::new(
                        img_rect.min.x + (*x as f32) * scale,
                        img_rect.min.y + (*y as f32) * scale,
                    )
                })
                .collect();
            painter.add(egui::Shape::closed_line(sc.clone(), Stroke::new(1.5_f32, col)));
            let label = classes
                .get(a.cls)
                .map(|c| c.name.as_str())
                .unwrap_or("?");
            let text_col = if color_lightness(col) > 140.0 {
                Color32::BLACK
            } else {
                Color32::WHITE
            };
            let pos = Pos2::new(sc[0].x, (sc[0].y - 12.0).max(img_rect.min.y));
            let galley = painter.layout_no_wrap(label.to_string(), FontId::proportional(10.0), text_col);
            let bg_rect = Rect::from_min_size(pos, galley.size() + Vec2::new(4.0, 2.0));
            painter.rect_filled(bg_rect, CornerRadius::ZERO, col);
            painter.galley(pos + Vec2::new(2.0, 1.0), galley, text_col);
        }
    }

    fn summary_text(&self, project: &Project) -> String {
        let cfg = self.build_version_config();
        if self.tab == TabKind::Mosaic {
            let m = self.build_mosaic_config();
            let sp = mosaic_splits(project, &cfg);
            let nt = if sp.get("train").map(|v| v.is_empty()).unwrap_or(true) { 0 } else { m.count_train };
            let nv = if sp.get("valid").map(|v| v.is_empty()).unwrap_or(true) { 0 } else { m.count_valid };
            let nte = if sp.get("test").map(|v| v.is_empty()).unwrap_or(true) { 0 } else { m.count_test };
            return format!(
                "拼接 {}×{} → Train {nt}、Valid {nv}、Test {nte} 張（圖塊來源：Train {}、Valid {}、Test {} 張原圖）",
                m.canvas_w,
                m.canvas_h,
                sp.get("train").map(|v| v.len()).unwrap_or(0),
                sp.get("valid").map(|v| v.len()).unwrap_or(0),
                sp.get("test").map(|v| v.len()).unwrap_or(0),
            );
        }
        let names = source_names(project, &cfg);
        let sp = split_items(&names, &cfg);
        let cp = split_copies(&sp, &cfg);
        let mult = if cfg.aug.any_enabled() {
            cfg.multiplier
        } else {
            1
        };
        format!(
            "來源 {} 張 → Train {}×{mult} = {}，Valid {}×{mult} = {}，Test {} → {}",
            names.len(),
            sp["train"].len(),
            cp["train"].iter().sum::<usize>(),
            sp["valid"].len(),
            cp["valid"].iter().sum::<usize>(),
            sp["test"].len(),
            cp["test"].iter().sum::<usize>(),
        )
    }
}

pub fn rgb_to_texture(ctx: &egui::Context, name: &str, img: &RgbImage) -> TextureHandle {
    let size = [img.width() as usize, img.height() as usize];
    let ci = ColorImage::from_rgb(size, img.as_raw());
    ctx.load_texture(name, ci, TextureOptions::LINEAR)
}

pub fn open_folder_in_os(path: &std::path::Path) {
    #[cfg(target_os = "windows")]
    {
        let _ = std::process::Command::new("explorer").arg(path).spawn();
    }
    #[cfg(target_os = "macos")]
    {
        let _ = std::process::Command::new("open").arg(path).spawn();
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let _ = std::process::Command::new("xdg-open").arg(path).spawn();
    }
}

fn aug_field(a: &AugConfig, key: &str) -> f64 {
    match key {
        "brightness" => a.brightness,
        "exposure" => a.exposure,
        "contrast" => a.contrast,
        "saturation" => a.saturation,
        "hue" => a.hue,
        "grayscale" => a.grayscale,
        "blur" => a.blur,
        "noise" => a.noise,
        _ => 0.0,
    }
}

fn set_aug_field(a: &mut AugConfig, key: &str, v: f64) {
    match key {
        "brightness" => a.brightness = v,
        "exposure" => a.exposure = v,
        "contrast" => a.contrast = v,
        "saturation" => a.saturation = v,
        "hue" => a.hue = v,
        "grayscale" => a.grayscale = v,
        "blur" => a.blur = v,
        "noise" => a.noise = v,
        _ => {}
    }
}
