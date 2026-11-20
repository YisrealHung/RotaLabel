use std::collections::HashMap;
use std::path::{Path, PathBuf};

use egui::{
    Align2, Color32, CornerRadius, Key, RichText, ScrollArea, Stroke, TextureHandle, Vec2,
};

use crate::imageio;
use crate::importers;
use crate::model::BoxKind;
use crate::project_io::{clean_path, Project, LABEL_FORMATS};
use crate::ui::canvas::{render_canvas, CanvasState, LabelMode, ToolMode};
use crate::ui::fonts::install_cjk_fonts;
use crate::ui::settings::AppSettings;
use crate::ui::theme::{apply_theme, format_hex_color, parse_hex_color, Palette};
use crate::ui::version_dialog::{open_folder_in_os, rgb_to_texture, VersionDialog};

const HELP_ROWS: [(&str, &str); 19] = [
    ("W", "畫 BBox（拖曳）"),
    ("E", "畫旋轉框（畫完拖黃色把手旋轉）"),
    ("V", "選取 / 編輯模式"),
    ("Ctrl + 左鍵 / 右鍵 / 雙擊", "在繪製模式下直接選取框"),
    ("Z / X", "選取框逆 / 順時針旋轉 1°（Shift：15°）"),
    ("拖曳旋轉把手 + Shift", "以 15° 為單位吸附"),
    ("T", "選取框在 BBox / 旋轉框之間切換"),
    ("1 – 9", "切換類別（若有選取框則同時變更其類別）"),
    ("Delete / Backspace", "刪除選取框（畫框途中可取消繪製）"),
    ("方向鍵", "微調位置 1px（Shift：10px）"),
    ("Ctrl+C / Ctrl+V", "複製 / 貼上框（可跨圖片）"),
    ("Ctrl+Z / Ctrl+Y", "復原 / 重做"),
    ("A / D", "上一張 / 下一張"),
    ("F", "圖片符合視窗"),
    ("滾輪 / Shift+滾輪", "上下 / 左右捲動"),
    ("Ctrl + 滾輪", "縮放"),
    ("中鍵或空白鍵拖曳", "平移"),
    ("L", "切換標註顯示模式"),
    ("Esc", "取消選取"),
];

enum ClassModal {
    Add { text: String, err: Option<String> },
    Rename { idx: usize, text: String, err: Option<String> },
    Recolor { idx: usize, color: Color32 },
    DeleteConfirm { idx: usize },
}

struct ImportModal {
    dir: PathBuf,
}

struct LabelOutputModal {
    fmt: String,
    dir: String,
    err: Option<String>,
}

pub struct RotaLabelApp {
    project: Option<Project>,
    current_idx: Option<usize>,
    canvas: CanvasState,
    texture: Option<TextureHandle>,
    settings: AppSettings,
    dark_theme: bool,
    pal: Palette,
    filter_text: String,
    status_msg: String,
    status_until: f64,
    class_modal: Option<ClassModal>,
    import_modal: Option<ImportModal>,
    label_out_modal: Option<LabelOutputModal>,
    version_dialog: Option<VersionDialog>,
    show_help: bool,
    info_dialog: Option<(String, String)>,
}

impl RotaLabelApp {
    pub fn new(cc: &eframe::CreationContext<'_>, initial_folder: Option<PathBuf>) -> Self {
        install_cjk_fonts(&cc.egui_ctx);
        let settings = AppSettings::load();
        let dark = settings.theme != "light";
        let pal = apply_theme(&cc.egui_ctx, dark);
        let label_mode = LabelMode::from_str(&settings.label_mode);

        let mut canvas = CanvasState::default();
        canvas.label_mode = label_mode;

        let mut app = Self {
            project: None,
            current_idx: None,
            canvas,
            texture: None,
            settings,
            dark_theme: dark,
            pal,
            filter_text: String::new(),
            status_msg: String::new(),
            status_until: 0.0,
            class_modal: None,
            import_modal: None,
            label_out_modal: None,
            version_dialog: None,
            show_help: false,
            info_dialog: None,
        };

        if let Some(dir) = initial_folder {
            app.open_project(&cc.egui_ctx, &dir);
        } else if let Some(last) = app.settings.last_project.clone() {
            let p = PathBuf::from(&last);
            if p.is_dir() {
                app.open_project(&cc.egui_ctx, &p);
            }
        }
        app
    }

    fn set_status(&mut self, ctx: &egui::Context, msg: impl Into<String>, secs: f64) {
        self.status_msg = msg.into();
        self.status_until = ctx.input(|i| i.time) + secs;
    }

    fn current_name(&self) -> Option<&str> {
        let p = self.project.as_ref()?;
        let idx = self.current_idx?;
        p.images.get(idx).map(|s| s.as_str())
    }

    fn flush(&mut self) {
        let Some(name) = self.current_name().map(|s| s.to_string()) else {
            return;
        };
        if !self.canvas.has_image {
            return;
        }
        let Some(ref mut p) = self.project else {
            return;
        };
        {
            let ia = p.get_mut(&name);
            ia.annotations = self.canvas.items.clone();
            ia.width = self.canvas.img_w;
            ia.height = self.canvas.img_h;
        }
        p.save_annotations(&name, false);
    }

    fn open_project(&mut self, ctx: &egui::Context, folder: &Path) {
        self.flush();
        let proj = match Project::open(folder) {
            Ok(p) => p,
            Err(e) => {
                self.info_dialog = Some(("開啟失敗".to_string(), e.to_string()));
                return;
            }
        };
        self.canvas.clear();
        self.texture = None;
        self.current_idx = None;
        self.settings.last_project = Some(proj.root.to_string_lossy().to_string());
        self.settings.save();

        let (n_img, n_box) = proj.auto_imported;
        if n_img > 0 {
            let out_txt = Self::output_text_for(&proj);
            self.set_status(
                ctx,
                format!("已載入資料夾內的標註：{n_img} 張圖片、{n_box} 個框（{out_txt}）"),
                15.0,
            );
        }

        let start = proj
            .settings
            .get("last_image")
            .and_then(|v| v.as_str())
            .and_then(|last| proj.images.iter().position(|n| n == last))
            .or_else(|| (!proj.images.is_empty()).then_some(0));

        self.project = Some(proj);
        self.update_title(ctx);
        if let Some(idx) = start {
            self.load_index(ctx, idx);
        }
    }

    fn load_index(&mut self, ctx: &egui::Context, idx: usize) {
        let Some(ref p) = self.project else {
            return;
        };
        if idx >= p.images.len() {
            return;
        }
        if self.current_idx != Some(idx) {
            self.flush();
        }
        self.current_idx = Some(idx);

        let name = self.project.as_ref().unwrap().images[idx].clone();
        let img_path = self.project.as_ref().unwrap().image_path(&name);
        let anns = self
            .project
            .as_ref()
            .unwrap()
            .get(&name)
            .map(|ia| ia.annotations.clone())
            .unwrap_or_default();

        if let Some(img) = imageio::imread(&img_path) {
            let w = img.width();
            let h = img.height();
            self.texture = Some(rgb_to_texture(ctx, "canvas_img", &img));
            self.canvas.load_image(w, h, anns, true);
            if let Some(ref mut p) = self.project {
                let ia = p.get_mut(&name);
                ia.width = w;
                ia.height = h;
                p.settings.insert(
                    "last_image".to_string(),
                    serde_json::Value::String(name),
                );
            }
        } else {
            self.texture = None;
            self.canvas.load_image(0, 0, Vec::new(), false);
            self.set_status(ctx, format!("無法讀取：{name}（此圖的標註保留不變）"), 8.0);
        }
    }

    fn reload_current(&mut self, ctx: &egui::Context) {
        let idx = self.current_idx;
        self.current_idx = None;
        if let Some(i) = idx {
            self.load_index(ctx, i);
        }
    }

    fn step(&mut self, ctx: &egui::Context, delta: isize) {
        let Some(ref p) = self.project else {
            return;
        };
        let Some(cur) = self.current_idx else {
            return;
        };
        let filter = self.filter_text.to_lowercase();
        let mut r = cur as isize + delta;
        while r >= 0 && (r as usize) < p.images.len() {
            let name = &p.images[r as usize];
            if filter.is_empty() || name.to_lowercase().contains(&filter) {
                self.load_index(ctx, r as usize);
                return;
            }
            r += delta;
        }
    }

    fn output_text_for(p: &Project) -> String {
        let fmt_label = LABEL_FORMATS
            .iter()
            .find(|(k, _)| *k == p.label_format())
            .map(|(_, l)| *l)
            .unwrap_or("DOTA");
        let where_str = match p.output_dir() {
            None => "圖片旁".to_string(),
            Some(out) => {
                if let Ok(rel) = out.strip_prefix(&p.root) {
                    format!("{}/", rel.to_string_lossy().replace('\\', "/"))
                } else {
                    out.to_string_lossy().to_string()
                }
            }
        };
        format!("{fmt_label} → {where_str}")
    }

    fn update_title(&self, ctx: &egui::Context) {
        let title = if let Some(ref p) = self.project {
            format!(
                "RotaLabel — {}  [標註輸出：{}]",
                p.root.display(),
                Self::output_text_for(p)
            )
        } else {
            "RotaLabel 標註工具".to_string()
        };
        ctx.send_viewport_cmd(egui::ViewportCommand::Title(title));
    }

    fn set_label_mode(&mut self, ctx: &egui::Context, mode: LabelMode) {
        self.canvas.label_mode = mode;
        self.settings.label_mode = mode.as_str().to_string();
        self.settings.save();
        self.set_status(ctx, format!("標註顯示：{}", mode.long_label()), 2.0);
    }

    fn toggle_theme(&mut self, ctx: &egui::Context) {
        self.dark_theme = !self.dark_theme;
        self.pal = apply_theme(ctx, self.dark_theme);
        self.settings.theme = if self.dark_theme { "dark" } else { "light" }.to_string();
        self.settings.save();
    }

    fn pick_class(&mut self, i: usize) -> bool {
        let Some(ref p) = self.project else {
            return false;
        };
        if i >= p.classes.len() {
            return false;
        }
        self.canvas.current_class = i;
        if self.canvas.set_selected_class(i) {
            self.flush();
            return true;
        }
        false
    }

    fn sync_selection_to_class(&mut self) {
        if let Some(idx) = self.canvas.selected {
            if let Some(a) = self.canvas.items.get(idx) {
                self.canvas.current_class = a.cls;
            }
        }
    }

    fn has_modal_open(&self) -> bool {
        self.class_modal.is_some()
            || self.import_modal.is_some()
            || self.label_out_modal.is_some()
            || self.version_dialog.is_some()
            || self.show_help
            || self.info_dialog.is_some()
    }

    fn handle_shortcuts(&mut self, ctx: &egui::Context) {
        if self.has_modal_open() || ctx.wants_keyboard_input() {
            return;
        }
        let mut modified = false;
        let mut open_picker = false;
        let mut open_version = false;
        let mut do_flush = false;
        let mut step_delta = 0isize;
        let mut do_fit = false;
        let mut do_cycle_labels = false;

        ctx.input(|i| {
            let ctrl = i.modifiers.ctrl;
            let shift = i.modifiers.shift;

            if i.key_pressed(Key::F1) {
                self.show_help = true;
            }
            if ctrl && i.key_pressed(Key::O) {
                open_picker = true;
            }
            if ctrl && i.key_pressed(Key::S) {
                do_flush = true;
            }
            if ctrl && i.key_pressed(Key::E) {
                open_version = true;
            }
            if ctrl && i.key_pressed(Key::Z) && !shift {
                if self.canvas.undo() {
                    modified = true;
                }
            }
            if ctrl && (i.key_pressed(Key::Y) || (shift && i.key_pressed(Key::Z))) {
                if self.canvas.redo() {
                    modified = true;
                }
            }
            if ctrl && i.key_pressed(Key::C) {
                self.canvas.copy_selected();
            }
            if ctrl && i.key_pressed(Key::V) {
                let n = self.project.as_ref().map(|p| p.classes.len()).unwrap_or(0);
                if self.canvas.paste(n) {
                    modified = true;
                }
            }
            if !ctrl {
                if i.key_pressed(Key::A) {
                    step_delta = -1;
                }
                if i.key_pressed(Key::D) {
                    step_delta = 1;
                }
                if i.key_pressed(Key::V) {
                    self.canvas.mode = ToolMode::Select;
                }
                if i.key_pressed(Key::W) {
                    self.canvas.mode = ToolMode::Bbox;
                }
                if i.key_pressed(Key::E) {
                    self.canvas.mode = ToolMode::Obb;
                }
                if i.key_pressed(Key::F) {
                    do_fit = true;
                }
                if i.key_pressed(Key::L) {
                    do_cycle_labels = true;
                }
                if i.key_pressed(Key::Z) {
                    if self.canvas.rotate_selected(if shift { -15.0 } else { -1.0 }) {
                        modified = true;
                    }
                }
                if i.key_pressed(Key::X) {
                    if self.canvas.rotate_selected(if shift { 15.0 } else { 1.0 }) {
                        modified = true;
                    }
                }
                if i.key_pressed(Key::T) {
                    if self.canvas.toggle_selected_kind() {
                        modified = true;
                    }
                }
                if i.key_pressed(Key::Delete) || i.key_pressed(Key::Backspace) {
                    if self.canvas.cancel_or_delete() {
                        modified = true;
                    }
                }
                if i.key_pressed(Key::Escape) {
                    if matches!(self.canvas.drag, Some(crate::ui::canvas::DragState::Draw { .. })) {
                        self.canvas.drag = None;
                    } else {
                        self.canvas.select(None);
                    }
                }
                let step_px = if shift { 10.0 } else { 1.0 };
                if i.key_pressed(Key::ArrowLeft) && self.canvas.nudge_selected(-step_px, 0.0) {
                    modified = true;
                }
                if i.key_pressed(Key::ArrowRight) && self.canvas.nudge_selected(step_px, 0.0) {
                    modified = true;
                }
                if i.key_pressed(Key::ArrowUp) && self.canvas.nudge_selected(0.0, -step_px) {
                    modified = true;
                }
                if i.key_pressed(Key::ArrowDown) && self.canvas.nudge_selected(0.0, step_px) {
                    modified = true;
                }
                let num_keys = [
                    Key::Num1, Key::Num2, Key::Num3, Key::Num4, Key::Num5, Key::Num6, Key::Num7,
                    Key::Num8, Key::Num9,
                ];
                for (idx, k) in num_keys.into_iter().enumerate() {
                    if i.key_pressed(k) {
                        self.pick_class(idx);
                    }
                }
            }
        });

        if open_picker {
            self.choose_folder(ctx);
        }
        if do_flush {
            self.flush();
            self.set_status(ctx, "已儲存", 1.5);
        }
        if open_version && self.project.is_some() {
            self.flush();
            let dlg = VersionDialog::new(self.project.as_ref().unwrap());
            self.version_dialog = Some(dlg);
        }
        if step_delta != 0 {
            self.step(ctx, step_delta);
        }
        if do_fit {
            self.canvas.fit_view(self.canvas.last_vp_size);
        }
        if do_cycle_labels {
            let next = self.canvas.label_mode.cycle();
            self.set_label_mode(ctx, next);
        }
        if modified {
            self.flush();
            self.sync_selection_to_class();
        }
    }

    fn choose_folder(&mut self, ctx: &egui::Context) {
        let mut dlg = rfd::FileDialog::new().set_title("選擇圖片資料夾");
        if let Some(ref last) = self.settings.last_project {
            let p = PathBuf::from(last);
            if p.is_dir() {
                dlg = dlg.set_directory(&p);
            }
        }
        if let Some(dir) = dlg.pick_folder() {
            self.open_project(ctx, &dir);
        }
    }

    fn choose_import_folder(&mut self) {
        let Some(ref p) = self.project else {
            return;
        };
        if let Some(dir) = rfd::FileDialog::new()
            .set_title("選擇標註資料夾（.txt，檔名需與圖片相同）")
            .set_directory(&p.root)
            .pick_folder()
        {
            self.import_modal = Some(ImportModal { dir });
        }
    }

    fn ui_top_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("menubar_and_toolbar").show(ctx, |ui| {
            // Menu bar
            egui::menu::bar(ui, |ui| {
                ui.menu_button("檔案", |ui| {
                    if ui.button("開啟資料夾 (Ctrl+O)").clicked() {
                        ui.close_menu();
                        self.choose_folder(ctx);
                    }
                    if ui
                        .add_enabled(self.project.is_some(), egui::Button::new("儲存 (Ctrl+S)"))
                        .clicked()
                    {
                        ui.close_menu();
                        self.flush();
                    }
                    if ui
                        .add_enabled(
                            self.project.is_some(),
                            egui::Button::new("匯入標註 (YOLO / YOLO-OBB / DOTA)…"),
                        )
                        .clicked()
                    {
                        ui.close_menu();
                        self.choose_import_folder();
                    }
                    if ui
                        .add_enabled(
                            self.project.is_some(),
                            egui::Button::new("標註輸出設定…"),
                        )
                        .clicked()
                    {
                        ui.close_menu();
                        if let Some(ref p) = self.project {
                            self.label_out_modal = Some(LabelOutputModal {
                                fmt: p.label_format().to_string(),
                                dir: p
                                    .output_dir()
                                    .unwrap_or_else(|| p.root.clone())
                                    .to_string_lossy()
                                    .to_string(),
                                err: None,
                            });
                        }
                    }
                    ui.separator();
                    if ui
                        .add_enabled(
                            self.project.is_some(),
                            egui::Button::new("產生資料集 (Ctrl+E)"),
                        )
                        .clicked()
                    {
                        ui.close_menu();
                        self.flush();
                        if let Some(ref p) = self.project {
                            self.version_dialog = Some(VersionDialog::new(p));
                        }
                    }
                    if ui
                        .add_enabled(
                            self.project.is_some(),
                            egui::Button::new("開啟版本資料夾"),
                        )
                        .clicked()
                    {
                        ui.close_menu();
                        if let Some(ref p) = self.project {
                            let _ = std::fs::create_dir_all(&p.versions_dir);
                            open_folder_in_os(&p.versions_dir);
                        }
                    }
                    ui.separator();
                    if ui.button("結束").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });

                ui.menu_button("編輯", |ui| {
                    if ui
                        .add_enabled(
                            self.canvas.can_undo(),
                            egui::Button::new("復原 (Ctrl+Z)"),
                        )
                        .clicked()
                    {
                        ui.close_menu();
                        if self.canvas.undo() {
                            self.flush();
                        }
                    }
                    if ui
                        .add_enabled(
                            self.canvas.can_redo(),
                            egui::Button::new("重做 (Ctrl+Y)"),
                        )
                        .clicked()
                    {
                        ui.close_menu();
                        if self.canvas.redo() {
                            self.flush();
                        }
                    }
                    ui.separator();
                    if ui
                        .selectable_label(self.canvas.mode == ToolMode::Select, "選取 (V)")
                        .clicked()
                    {
                        ui.close_menu();
                        self.canvas.mode = ToolMode::Select;
                    }
                    if ui
                        .selectable_label(self.canvas.mode == ToolMode::Bbox, "BBox (W)")
                        .clicked()
                    {
                        ui.close_menu();
                        self.canvas.mode = ToolMode::Bbox;
                    }
                    if ui
                        .selectable_label(self.canvas.mode == ToolMode::Obb, "旋轉框 (E)")
                        .clicked()
                    {
                        ui.close_menu();
                        self.canvas.mode = ToolMode::Obb;
                    }
                });

                ui.menu_button("檢視", |ui| {
                    if ui.button("符合視窗 (F)").clicked() {
                        ui.close_menu();
                        self.canvas.fit_view(self.canvas.last_vp_size);
                    }
                    if ui.button("切換深色 / 淺色主題").clicked() {
                        ui.close_menu();
                        self.toggle_theme(ctx);
                    }
                    ui.separator();
                    ui.menu_button("標註顯示", |ui| {
                        for m in LabelMode::ALL_MODES {
                            if ui
                                .selectable_label(self.canvas.label_mode == m, m.long_label())
                                .clicked()
                            {
                                ui.close_menu();
                                self.set_label_mode(ctx, m);
                            }
                        }
                    });
                });

                ui.menu_button("說明", |ui| {
                    if ui.button("快捷鍵說明 (F1)").clicked() {
                        ui.close_menu();
                        self.show_help = true;
                    }
                });
            });

            ui.separator();

            // Toolbar
            ui.horizontal(|ui| {
                if ui.button("開啟資料夾").on_hover_text("開啟資料夾 (Ctrl+O)").clicked() {
                    self.choose_folder(ctx);
                }
                if ui
                    .add_enabled(self.project.is_some(), egui::Button::new("儲存"))
                    .on_hover_text("儲存 (Ctrl+S)")
                    .clicked()
                {
                    self.flush();
                    self.set_status(ctx, "已儲存", 1.5);
                }
                ui.separator();
                if ui
                    .add_enabled(self.project.is_some(), egui::Button::new("◀ 上一張"))
                    .on_hover_text("上一張 (A)")
                    .clicked()
                {
                    self.step(ctx, -1);
                }
                if ui
                    .add_enabled(self.project.is_some(), egui::Button::new("下一張 ▶"))
                    .on_hover_text("下一張 (D)")
                    .clicked()
                {
                    self.step(ctx, 1);
                }
                ui.separator();
                for (mode, label, tip) in [
                    (ToolMode::Select, "選取", "選取 / 編輯框 (V)"),
                    (ToolMode::Bbox, "BBox", "畫 BBox (W)"),
                    (ToolMode::Obb, "旋轉框", "畫旋轉框 (E)"),
                ] {
                    if ui
                        .selectable_label(self.canvas.mode == mode, label)
                        .on_hover_text(tip)
                        .clicked()
                    {
                        self.canvas.mode = mode;
                    }
                }
                ui.separator();
                if ui
                    .add_enabled(self.canvas.can_undo(), egui::Button::new("↶"))
                    .on_hover_text("復原 (Ctrl+Z)")
                    .clicked()
                {
                    if self.canvas.undo() {
                        self.flush();
                    }
                }
                if ui
                    .add_enabled(self.canvas.can_redo(), egui::Button::new("↷"))
                    .on_hover_text("重做 (Ctrl+Y)")
                    .clicked()
                {
                    if self.canvas.redo() {
                        self.flush();
                    }
                }
                if ui.button("⛶").on_hover_text("符合視窗 (F)").clicked() {
                    self.canvas.fit_view(self.canvas.last_vp_size);
                }
                ui.separator();
                ui.menu_button(format!("顯示：{}", self.canvas.label_mode.short_label()), |ui| {
                    for m in LabelMode::ALL_MODES {
                        if ui
                            .selectable_label(self.canvas.label_mode == m, m.long_label())
                            .clicked()
                        {
                            ui.close_menu();
                            self.set_label_mode(ctx, m);
                        }
                    }
                });

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let gen_enabled = self.project.is_some();
                    let gen_btn = ui
                        .add_enabled(
                            gen_enabled,
                            egui::Button::new(
                                RichText::new("產生資料集")
                                    .color(if gen_enabled {
                                        self.pal.accent_text
                                    } else {
                                        self.pal.muted
                                    })
                                    .strong(),
                            )
                            .fill(if gen_enabled {
                                self.pal.accent
                            } else {
                                self.pal.panel2
                            }),
                        )
                        .on_hover_text("產生資料集版本：切分、擴增、匯出 (Ctrl+E)");
                    if gen_btn.clicked() {
                        self.flush();
                        if let Some(ref p) = self.project {
                            self.version_dialog = Some(VersionDialog::new(p));
                        }
                    }
                    if ui.button("?").on_hover_text("快捷鍵說明 (F1)").clicked() {
                        self.show_help = true;
                    }
                    if ui.button("◐").on_hover_text("切換深色 / 淺色主題").clicked() {
                        self.toggle_theme(ctx);
                    }
                });
            });
        });
    }

    fn ui_left_panel(&mut self, ctx: &egui::Context) {
        let mut clicked_idx: Option<usize> = None;
        egui::SidePanel::left("images_panel")
            .frame(
                egui::Frame::side_top_panel(&ctx.style())
                    .fill(self.pal.panel)
                    .inner_margin(egui::Margin::same(8)),
            )
            .resizable(true)
            .default_width(250.0)
            .min_width(220.0)
            .show(ctx, |ui| {
                ui.strong("圖片");
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.add(
                        egui::TextEdit::singleline(&mut self.filter_text)
                            .hint_text("🔍  搜尋檔名…")
                            .desired_width(ui.available_width() - 28.0),
                    );
                    if !self.filter_text.is_empty() && ui.button("×").clicked() {
                        self.filter_text.clear();
                    }
                });
                ui.add_space(4.0);

                if let Some(ref p) = self.project {
                    let filter = self.filter_text.to_lowercase();
                    let visible_indices: Vec<usize> = p
                        .images
                        .iter()
                        .enumerate()
                        .filter(|(_, n)| filter.is_empty() || n.to_lowercase().contains(&filter))
                        .map(|(i, _)| i)
                        .collect();

                    let list_h = (ui.available_height() - 48.0).max(80.0);
                    ScrollArea::vertical()
                        .id_salt("img_list_scroll")
                        .max_height(list_h)
                        .show_rows(ui, 28.0, visible_indices.len(), |ui, row_range| {
                            for r in row_range {
                                let img_idx = visible_indices[r];
                                let name = &p.images[img_idx];
                                let count = p.box_count(name);
                                let selected = self.current_idx == Some(img_idx);
                                let (rect, resp) = ui.allocate_exact_size(
                                    Vec2::new(ui.available_width(), 28.0),
                                    egui::Sense::click(),
                                );
                                if resp.clicked() {
                                    clicked_idx = Some(img_idx);
                                }
                                let painter = ui.painter_at(rect);
                                if selected {
                                    painter.rect_filled(
                                        rect.shrink(1.0),
                                        CornerRadius::same(6),
                                        self.pal.selection,
                                    );
                                    painter.rect_filled(
                                        egui::Rect::from_min_size(
                                            egui::Pos2::new(rect.min.x + 2.0, rect.min.y + 5.0),
                                            Vec2::new(3.0, rect.height() - 10.0),
                                        ),
                                        CornerRadius::same(2),
                                        self.pal.accent,
                                    );
                                } else if resp.hovered() {
                                    painter.rect_filled(
                                        rect.shrink(1.0),
                                        CornerRadius::same(6),
                                        self.pal.panel2,
                                    );
                                }

                                let dot_col = if count > 0 { self.pal.ok } else { self.pal.border };
                                painter.circle_filled(
                                    egui::Pos2::new(rect.min.x + 14.0, rect.center().y),
                                    4.0,
                                    dot_col,
                                );

                                let mut right_x = rect.max.x - 8.0;
                                if count > 0 {
                                    let pill_text = count.to_string();
                                    let g = painter.layout_no_wrap(
                                        pill_text,
                                        egui::FontId::proportional(11.0),
                                        self.pal.text,
                                    );
                                    let pw = g.size().x + 10.0;
                                    let ph = g.size().y + 2.0;
                                    let prect = egui::Rect::from_min_size(
                                        egui::Pos2::new(right_x - pw, rect.center().y - ph * 0.5),
                                        Vec2::new(pw, ph),
                                    );
                                    let pbg = Color32::from_rgba_unmultiplied(
                                        self.pal.accent.r(),
                                        self.pal.accent.g(),
                                        self.pal.accent.b(),
                                        60,
                                    );
                                    painter.rect_filled(prect, CornerRadius::same((ph * 0.5) as u8), pbg);
                                    painter.galley(
                                        egui::Pos2::new(prect.min.x + 5.0, prect.min.y + 1.0),
                                        g,
                                        self.pal.text,
                                    );
                                    right_x = prect.min.x - 6.0;
                                }

                                let text_col = if count > 0 { self.pal.text } else { self.pal.muted };
                                let max_w = (right_x - rect.min.x - 26.0).max(20.0);
                                let galley = painter.layout(
                                    name.clone(),
                                    egui::FontId::proportional(13.0),
                                    text_col,
                                    max_w,
                                );
                                painter.galley(
                                    egui::Pos2::new(
                                        rect.min.x + 26.0,
                                        rect.center().y - galley.size().y * 0.5,
                                    ),
                                    galley,
                                    text_col,
                                );
                            }
                        });

                    let total = p.images.len();
                    let done = p.images.iter().filter(|n| p.box_count(n) > 0).count();
                    let boxes: usize = p.images.iter().map(|n| p.box_count(n)).sum();
                    let pct = if total > 0 { done * 100 / total } else { 0 };
                    ui.with_layout(egui::Layout::bottom_up(egui::Align::Min), |ui| {
                        let frac = if total > 0 { done as f32 / total as f32 } else { 0.0 };
                        ui.add(
                            egui::ProgressBar::new(frac)
                                .desired_height(6.0)
                                .fill(self.pal.accent),
                        );
                        ui.label(
                            RichText::new(format!("已標註 {done} / {total} 張（{pct}%）· {boxes} 個框"))
                                .small()
                                .color(self.pal.muted),
                        );
                    });
                } else {
                    ui.label(RichText::new("尚未開啟資料夾").color(self.pal.muted));
                }
            });

        if let Some(idx) = clicked_idx {
            self.load_index(ctx, idx);
        }
    }

    fn ui_right_panel(&mut self, ctx: &egui::Context) {
        let mut pick_cls: Option<usize> = None;
        let mut open_rename: Option<(usize, String)> = None;
        let mut open_recolor: Option<(usize, Color32)> = None;
        let mut open_delete: Option<usize> = None;
        let mut select_ann: Option<usize> = None;

        egui::SidePanel::right("right_panel")
            .frame(
                egui::Frame::side_top_panel(&ctx.style())
                    .fill(self.pal.panel)
                    .inner_margin(egui::Margin::same(8)),
            )
            .resizable(true)
            .default_width(295.0)
            .min_width(260.0)
            .show(ctx, |ui| {
                let total_h = ui.available_height();
                let top_h = (total_h * 0.44).max(160.0);

                // Top: Classes
                ui.allocate_ui_with_layout(
                    Vec2::new(ui.available_width(), top_h),
                    egui::Layout::top_down(egui::Align::Min),
                    |ui| {
                        ui.strong("類別");
                        ui.add_space(4.0);
                        let mut counts: HashMap<usize, usize> = HashMap::new();
                        for a in &self.canvas.items {
                            *counts.entry(a.cls).or_insert(0) += 1;
                        }
                        let classes_len = self.project.as_ref().map(|p| p.classes.len()).unwrap_or(0);
                        let list_h = (top_h - 56.0).max(80.0);
                        if let Some(ref p) = self.project {
                            ScrollArea::vertical()
                                .id_salt("class_list_scroll")
                                .max_height(list_h)
                                .show(ui, |ui| {
                                    for (i, c) in p.classes.iter().enumerate() {
                                        let selected = self.canvas.current_class == i;
                                        let (rect, resp) = ui.allocate_exact_size(
                                            Vec2::new(ui.available_width(), 30.0),
                                            egui::Sense::click(),
                                        );
                                        if resp.double_clicked() {
                                            open_rename = Some((i, c.name.clone()));
                                        } else if resp.clicked() {
                                            pick_cls = Some(i);
                                        }
                                        let painter = ui.painter_at(rect);
                                        if selected {
                                            painter.rect_filled(
                                                rect.shrink(1.0),
                                                CornerRadius::same(6),
                                                self.pal.selection,
                                            );
                                        } else if resp.hovered() {
                                            painter.rect_filled(
                                                rect.shrink(1.0),
                                                CornerRadius::same(6),
                                                self.pal.panel2,
                                            );
                                        }
                                        let swatch = parse_hex_color(&c.color);
                                        painter.rect_filled(
                                            egui::Rect::from_min_size(
                                                egui::Pos2::new(rect.min.x + 10.0, rect.center().y - 7.0),
                                                Vec2::splat(14.0),
                                            ),
                                            CornerRadius::same(4),
                                            swatch,
                                        );
                                        let mut right_x = rect.max.x - 8.0;
                                        if i < 9 {
                                            let key_s = (i + 1).to_string();
                                            let g = painter.layout_no_wrap(
                                                key_s,
                                                egui::FontId::proportional(11.0),
                                                self.pal.muted,
                                            );
                                            let pw = g.size().x + 10.0;
                                            let ph = g.size().y + 2.0;
                                            let prect = egui::Rect::from_min_size(
                                                egui::Pos2::new(right_x - pw, rect.center().y - ph * 0.5),
                                                Vec2::new(pw, ph),
                                            );
                                            painter.rect_filled(
                                                prect,
                                                CornerRadius::same((ph * 0.5) as u8),
                                                self.pal.panel2,
                                            );
                                            painter.galley(
                                                egui::Pos2::new(prect.min.x + 5.0, prect.min.y + 1.0),
                                                g,
                                                self.pal.muted,
                                            );
                                            right_x = prect.min.x - 6.0;
                                        }
                                        if let Some(&cnt) = counts.get(&i) {
                                            if cnt > 0 {
                                                let g = painter.layout_no_wrap(
                                                    cnt.to_string(),
                                                    egui::FontId::proportional(11.0),
                                                    self.pal.text,
                                                );
                                                let pw = g.size().x + 10.0;
                                                let ph = g.size().y + 2.0;
                                                let prect = egui::Rect::from_min_size(
                                                    egui::Pos2::new(right_x - pw, rect.center().y - ph * 0.5),
                                                    Vec2::new(pw, ph),
                                                );
                                                let pbg = Color32::from_rgba_unmultiplied(
                                                    self.pal.accent.r(),
                                                    self.pal.accent.g(),
                                                    self.pal.accent.b(),
                                                    60,
                                                );
                                                painter.rect_filled(
                                                    prect,
                                                    CornerRadius::same((ph * 0.5) as u8),
                                                    pbg,
                                                );
                                                painter.galley(
                                                    egui::Pos2::new(prect.min.x + 5.0, prect.min.y + 1.0),
                                                    g,
                                                    self.pal.text,
                                                );
                                            }
                                        }
                                        let g = painter.layout_no_wrap(
                                            c.name.clone(),
                                            egui::FontId::proportional(13.0),
                                            self.pal.text,
                                        );
                                        painter.galley(
                                            egui::Pos2::new(
                                                rect.min.x + 32.0,
                                                rect.center().y - g.size().y * 0.5,
                                            ),
                                            g,
                                            self.pal.text,
                                        );
                                    }
                                });
                        }
                        ui.horizontal(|ui| {
                            let has_proj = self.project.is_some();
                            if ui.add_enabled(has_proj, egui::Button::new("＋ 新增")).clicked() {
                                self.class_modal = Some(ClassModal::Add {
                                    text: String::new(),
                                    err: None,
                                });
                            }
                            let cur = self.canvas.current_class;
                            let valid_cls = has_proj && cur < classes_len;
                            if ui.add_enabled(valid_cls, egui::Button::new("改名")).clicked() {
                                let cname = self.project.as_ref().unwrap().classes[cur].name.clone();
                                open_rename = Some((cur, cname));
                            }
                            if ui.add_enabled(valid_cls, egui::Button::new("顏色")).clicked() {
                                let cstr = &self.project.as_ref().unwrap().classes[cur].color;
                                open_recolor = Some((cur, parse_hex_color(cstr)));
                            }
                            if ui.add_enabled(valid_cls, egui::Button::new("刪除")).clicked() {
                                open_delete = Some(cur);
                            }
                        });
                    },
                );

                ui.separator();

                // Bottom: Current image annotations
                let n_anns = self.canvas.items.len();
                let header = if n_anns > 0 {
                    format!("此圖標註（{n_anns}）")
                } else {
                    "此圖標註".to_string()
                };
                ui.strong(header);
                ui.add_space(4.0);
                if n_anns == 0 {
                    ui.centered_and_justified(|ui| {
                        ui.label(
                            RichText::new("尚無標註\n按 W 畫 BBox，按 E 畫旋轉框")
                                .color(self.pal.muted),
                        );
                    });
                } else {
                    let classes = self
                        .project
                        .as_ref()
                        .map(|p| p.classes.as_slice())
                        .unwrap_or(&[]);
                    ScrollArea::vertical()
                        .id_salt("ann_list_scroll")
                        .show_rows(ui, 32.0, n_anns, |ui, range| {
                            for i in range {
                                let a = &self.canvas.items[i];
                                let selected = self.canvas.selected == Some(i);
                                let (rect, resp) = ui.allocate_exact_size(
                                    Vec2::new(ui.available_width(), 32.0),
                                    egui::Sense::click(),
                                );
                                if resp.clicked() {
                                    select_ann = Some(i);
                                }
                                let painter = ui.painter_at(rect);
                                if selected {
                                    painter.rect_filled(
                                        rect.shrink(1.0),
                                        CornerRadius::same(6),
                                        self.pal.selection,
                                    );
                                    painter.rect_filled(
                                        egui::Rect::from_min_size(
                                            egui::Pos2::new(rect.min.x + 2.0, rect.min.y + 6.0),
                                            Vec2::new(3.0, rect.height() - 12.0),
                                        ),
                                        CornerRadius::same(2),
                                        self.pal.accent,
                                    );
                                } else if resp.hovered() {
                                    painter.rect_filled(
                                        rect.shrink(1.0),
                                        CornerRadius::same(6),
                                        self.pal.panel2,
                                    );
                                }

                                let col = classes
                                    .get(a.cls)
                                    .map(|c| parse_hex_color(&c.color))
                                    .unwrap_or_else(|| Color32::from_rgb(255, 0, 255));
                                let cname = classes
                                    .get(a.cls)
                                    .map(|c| c.name.as_str())
                                    .unwrap_or("?");
                                painter.text(
                                    egui::Pos2::new(rect.min.x + 16.0, rect.center().y),
                                    Align2::CENTER_CENTER,
                                    (i + 1).to_string(),
                                    egui::FontId::proportional(11.0),
                                    self.pal.muted,
                                );
                                painter.rect_stroke(
                                    egui::Rect::from_center_size(
                                        egui::Pos2::new(rect.min.x + 34.0, rect.center().y),
                                        Vec2::new(12.0, 9.0),
                                    ),
                                    CornerRadius::ZERO,
                                    Stroke::new(2.0_f32, col),
                                    egui::StrokeKind::Middle,
                                );

                                let mut right_x = rect.max.x - 8.0;
                                let (tag, tag_bg) = if a.kind == BoxKind::Obb {
                                    (format!("旋轉 {:.1}°", a.angle), self.pal.accent)
                                } else {
                                    ("BBox".to_string(), self.pal.muted)
                                };
                                let tg = painter.layout_no_wrap(
                                    tag,
                                    egui::FontId::proportional(11.0),
                                    self.pal.text,
                                );
                                let pw = tg.size().x + 10.0;
                                let ph = tg.size().y + 2.0;
                                let prect = egui::Rect::from_min_size(
                                    egui::Pos2::new(right_x - pw, rect.center().y - ph * 0.5),
                                    Vec2::new(pw, ph),
                                );
                                painter.rect_filled(
                                    prect,
                                    CornerRadius::same((ph * 0.5) as u8),
                                    Color32::from_rgba_unmultiplied(tag_bg.r(), tag_bg.g(), tag_bg.b(), 55),
                                );
                                painter.galley(
                                    egui::Pos2::new(prect.min.x + 5.0, prect.min.y + 1.0),
                                    tg,
                                    self.pal.text,
                                );
                                right_x = prect.min.x - 6.0;

                                let size_s = format!("{:.0}×{:.0}", a.w, a.h);
                                let sg = painter.layout_no_wrap(
                                    size_s,
                                    egui::FontId::proportional(11.0),
                                    self.pal.muted,
                                );
                                painter.galley(
                                    egui::Pos2::new(
                                        right_x - sg.size().x,
                                        rect.center().y - sg.size().y * 0.5,
                                    ),
                                    sg.clone(),
                                    self.pal.muted,
                                );

                                let ng = painter.layout_no_wrap(
                                    cname.to_string(),
                                    egui::FontId::proportional(13.0),
                                    self.pal.text,
                                );
                                painter.galley(
                                    egui::Pos2::new(
                                        rect.min.x + 48.0,
                                        rect.center().y - ng.size().y * 0.5,
                                    ),
                                    ng,
                                    self.pal.text,
                                );
                            }
                        });
                }
            });

        if let Some(i) = pick_cls {
            self.pick_class(i);
        }
        if let Some((idx, text)) = open_rename {
            self.class_modal = Some(ClassModal::Rename {
                idx,
                text,
                err: None,
            });
        }
        if let Some((idx, color)) = open_recolor {
            self.class_modal = Some(ClassModal::Recolor { idx, color });
        }
        if let Some(idx) = open_delete {
            if let Some(ref p) = self.project {
                if p.classes.len() <= 1 {
                    self.info_dialog =
                        Some(("刪除類別".to_string(), "至少需保留一個類別".to_string()));
                } else {
                    self.class_modal = Some(ClassModal::DeleteConfirm { idx });
                }
            }
        }
        if let Some(idx) = select_ann {
            self.canvas.select(Some(idx));
            self.sync_selection_to_class();
        }
    }

    fn ui_status_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                let badge = egui::Frame::NONE
                    .fill(self.pal.accent)
                    .corner_radius(CornerRadius::same(8))
                    .inner_margin(egui::Margin::symmetric(8, 2));
                badge.show(ui, |ui| {
                    ui.label(
                        RichText::new(self.canvas.mode.label())
                            .color(self.pal.accent_text)
                            .strong()
                            .small(),
                    );
                });

                let now = ctx.input(|i| i.time);
                if !self.status_msg.is_empty() && now < self.status_until {
                    ui.label(RichText::new(&self.status_msg).color(self.pal.text).small());
                } else if let (Some(ref p), Some(idx)) = (&self.project, self.current_idx) {
                    if let Some(name) = p.images.get(idx) {
                        let cname = p
                            .classes
                            .get(self.canvas.current_class)
                            .map(|c| c.name.as_str())
                            .unwrap_or("?");
                        let mut s = format!(
                            "{} / {}   ·   {name}   ·   {}×{}   ·   類別：{cname}",
                            idx + 1,
                            p.images.len(),
                            self.canvas.img_w,
                            self.canvas.img_h
                        );
                        if let Some(sel) = self.canvas.selected {
                            if let Some(a) = self.canvas.items.get(sel) {
                                s.push_str(&format!(
                                    "   ·   選取 cx {:.0}  cy {:.0}  w {:.0}  h {:.0}  ∠ {:.1}°",
                                    a.cx, a.cy, a.w, a.h, a.angle
                                ));
                            }
                        }
                        ui.label(RichText::new(s).color(self.pal.muted).small());
                    }
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(
                        RichText::new(format!("縮放 {:.0}%", self.canvas.zoom * 100.0))
                            .color(self.pal.muted)
                            .small(),
                    );
                    if let Some(c) = self.canvas.cursor_img {
                        ui.label(
                            RichText::new(format!("x {:.0}   y {:.0}", c[0], c[1]))
                                .color(self.pal.muted)
                                .small(),
                        );
                    }
                });
            });
        });
    }

    fn ui_modals(&mut self, ctx: &egui::Context) {
        // Class modal
        if let Some(mut modal) = self.class_modal.take() {
            let mut keep = true;
            match &mut modal {
                ClassModal::Add { text, err } => {
                    egui::Window::new("新增類別")
                        .collapsible(false)
                        .resizable(false)
                        .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
                        .show(ctx, |ui| {
                            ui.label("類別名稱：");
                            let resp = ui.text_edit_singleline(text);
                            resp.request_focus();
                            if let Some(e) = err {
                                ui.label(RichText::new(e.as_str()).color(self.pal.warn));
                            }
                            let submit = ui.input(|i| i.key_pressed(Key::Enter));
                            ui.horizontal(|ui| {
                                if ui.button("確定").clicked() || submit {
                                    let trimmed = text.trim().to_string();
                                    if !trimmed.is_empty() {
                                        if let Some(ref mut p) = self.project {
                                            if p.classes.iter().any(|c| c.name == trimmed) {
                                                *err = Some("類別名稱已存在".to_string());
                                            } else {
                                                let idx = p.add_class(trimmed);
                                                self.canvas.current_class = idx;
                                                keep = false;
                                            }
                                        }
                                    }
                                }
                                if ui.button("取消").clicked()
                                    || ui.input(|i| i.key_pressed(Key::Escape))
                                {
                                    keep = false;
                                }
                            });
                        });
                }
                ClassModal::Rename { idx, text, err } => {
                    egui::Window::new("改名")
                        .collapsible(false)
                        .resizable(false)
                        .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
                        .show(ctx, |ui| {
                            ui.label("類別名稱：");
                            let resp = ui.text_edit_singleline(text);
                            resp.request_focus();
                            if let Some(e) = err {
                                ui.label(RichText::new(e.as_str()).color(self.pal.warn));
                            }
                            let submit = ui.input(|i| i.key_pressed(Key::Enter));
                            ui.horizontal(|ui| {
                                if ui.button("確定").clicked() || submit {
                                    let trimmed = text.trim().to_string();
                                    if let Some(ref mut p) = self.project {
                                        if *idx < p.classes.len() && !trimmed.is_empty() {
                                            if trimmed != p.classes[*idx].name
                                                && p.classes.iter().any(|c| c.name == trimmed)
                                            {
                                                *err = Some("類別名稱已存在".to_string());
                                            } else {
                                                p.classes[*idx].name = trimmed;
                                                p.save_meta();
                                                p.sync_label_files();
                                                keep = false;
                                            }
                                        }
                                    }
                                }
                                if ui.button("取消").clicked()
                                    || ui.input(|i| i.key_pressed(Key::Escape))
                                {
                                    keep = false;
                                }
                            });
                        });
                }
                ClassModal::Recolor { idx, color } => {
                    egui::Window::new("選擇顏色")
                        .collapsible(false)
                        .resizable(false)
                        .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
                        .show(ctx, |ui| {
                            egui::color_picker::color_picker_color32(
                                ui,
                                color,
                                egui::color_picker::Alpha::Opaque,
                            );
                            ui.horizontal(|ui| {
                                if ui.button("確定").clicked() {
                                    if let Some(ref mut p) = self.project {
                                        if *idx < p.classes.len() {
                                            p.classes[*idx].color = format_hex_color(*color);
                                            p.save_meta();
                                        }
                                    }
                                    keep = false;
                                }
                                if ui.button("取消").clicked()
                                    || ui.input(|i| i.key_pressed(Key::Escape))
                                {
                                    keep = false;
                                }
                            });
                        });
                }
                ClassModal::DeleteConfirm { idx } => {
                    let cname = self
                        .project
                        .as_ref()
                        .and_then(|p| p.classes.get(*idx))
                        .map(|c| c.name.clone())
                        .unwrap_or_default();
                    let mut do_delete = false;
                    egui::Window::new("刪除類別")
                        .collapsible(false)
                        .resizable(false)
                        .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
                        .show(ctx, |ui| {
                            ui.label(format!(
                                "刪除類別「{cname}」會一併刪除所有圖片中此類別的框，且無法復原。確定嗎？"
                            ));
                            ui.horizontal(|ui| {
                                if ui.button("確定刪除").clicked() {
                                    do_delete = true;
                                    keep = false;
                                }
                                if ui.button("取消").clicked()
                                    || ui.input(|i| i.key_pressed(Key::Escape))
                                {
                                    keep = false;
                                }
                            });
                        });
                    if do_delete {
                        self.flush();
                        let removed = if let Some(ref mut p) = self.project {
                            p.delete_class(*idx)
                        } else {
                            0
                        };
                        self.canvas.clipboard.clear();
                        if let Some(ref p) = self.project {
                            self.canvas.current_class =
                                self.canvas.current_class.min(p.classes.len().saturating_sub(1));
                        }
                        self.reload_current(ctx);
                        self.info_dialog =
                            Some(("刪除類別".to_string(), format!("已刪除 {removed} 個框")));
                    }
                }
            }
            if keep {
                self.class_modal = Some(modal);
            }
        }

        // Import modal
        if let Some(imp) = self.import_modal.take() {
            let mut keep = true;
            let mut run_choice: Option<bool> = None;
            egui::Window::new("匯入標註")
                .collapsible(false)
                .resizable(false)
                .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
                .show(ctx, |ui| {
                    ui.label("對有對應標註檔的圖片：");
                    ui.horizontal(|ui| {
                        if ui.button("取代現有標註").clicked() {
                            run_choice = Some(true);
                            keep = false;
                        }
                        if ui.button("附加到現有標註").clicked() {
                            run_choice = Some(false);
                            keep = false;
                        }
                        if ui.button("取消").clicked()
                            || ui.input(|i| i.key_pressed(Key::Escape))
                        {
                            keep = false;
                        }
                    });
                });
            if let Some(overwrite) = run_choice {
                self.flush();
                if let Some(ref mut p) = self.project {
                    let (n_img, n_box) = importers::import_labels(p, &imp.dir, overwrite);
                    self.reload_current(ctx);
                    self.update_title(ctx);
                    self.info_dialog = Some((
                        "匯入標註".to_string(),
                        format!("已匯入 {n_img} 張圖片、{n_box} 個框"),
                    ));
                }
            } else if keep {
                self.import_modal = Some(imp);
            }
        }

        // Label output settings modal
        if let Some(mut modal) = self.label_out_modal.take() {
            let mut keep = true;
            let mut apply = false;
            egui::Window::new("標註輸出設定")
                .collapsible(false)
                .resizable(false)
                .default_width(520.0)
                .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("格式");
                        let cur_txt = LABEL_FORMATS
                            .iter()
                            .find(|(k, _)| *k == modal.fmt)
                            .map(|(_, l)| *l)
                            .unwrap_or("DOTA");
                        egui::ComboBox::from_id_salt("label_out_fmt")
                            .selected_text(cur_txt)
                            .show_ui(ui, |ui| {
                                for (k, l) in LABEL_FORMATS {
                                    ui.selectable_value(&mut modal.fmt, k.to_string(), l);
                                }
                            });
                    });
                    ui.horizontal(|ui| {
                        ui.label("資料夾");
                        ui.add(
                            egui::TextEdit::singleline(&mut modal.dir)
                                .desired_width(360.0),
                        );
                        if ui.button("瀏覽…").clicked() {
                            if let Some(d) = rfd::FileDialog::new()
                                .set_title("選擇輸出資料夾")
                                .set_directory(Path::new(&modal.dir))
                                .pick_folder()
                            {
                                modal.dir = d.to_string_lossy().to_string();
                            }
                        }
                    });
                    ui.label(
                        RichText::new("按確定後，所有已標註的圖片會以新設定重新寫出一次。")
                            .small()
                            .color(self.pal.muted),
                    );
                    if let Some(ref e) = modal.err {
                        ui.label(RichText::new(e).color(self.pal.warn));
                    }
                    ui.horizontal(|ui| {
                        if ui.button("確定").clicked() {
                            apply = true;
                        }
                        if ui.button("取消").clicked()
                            || ui.input(|i| i.key_pressed(Key::Escape))
                        {
                            keep = false;
                        }
                    });
                });
            if apply {
                self.flush();
                if let Some(ref mut p) = self.project {
                    let trimmed = modal.dir.trim();
                    let target = if trimmed.is_empty() {
                        p.root.clone()
                    } else {
                        let tp = PathBuf::from(trimmed);
                        if tp.is_absolute() {
                            tp
                        } else {
                            p.root.join(tp)
                        }
                    };
                    match p.set_label_output(&modal.fmt, &target) {
                        Ok(n) => {
                            let out_txt = Self::output_text_for(p);
                            self.update_title(ctx);
                            self.info_dialog = Some((
                                "標註輸出設定".to_string(),
                                format!("已將 {n} 張圖片的標註寫到 {out_txt}"),
                            ));
                            keep = false;
                        }
                        Err(e) => {
                            modal.err = Some(format!("無法套用：{e}"));
                        }
                    }
                }
            }
            if keep {
                self.label_out_modal = Some(modal);
            }
        }

        // Version dialog
        if let Some(mut dlg) = self.version_dialog.take() {
            let mut mode = self.canvas.label_mode;
            let still_open = if let Some(ref mut p) = self.project {
                dlg.show(ctx, p, &self.pal, &mut mode)
            } else {
                false
            };
            if mode != self.canvas.label_mode {
                self.set_label_mode(ctx, mode);
            }
            if still_open {
                self.version_dialog = Some(dlg);
            }
        }

        // Help modal
        if self.show_help {
            let mut open = true;
            egui::Window::new("快捷鍵")
                .open(&mut open)
                .collapsible(false)
                .resizable(false)
                .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
                .show(ctx, |ui| {
                    egui::Grid::new("help_grid").striped(true).show(ui, |ui| {
                        for (k, desc) in HELP_ROWS {
                            ui.strong(k);
                            ui.label(desc);
                            ui.end_row();
                        }
                    });
                    ui.add_space(6.0);
                    ui.label(
                        RichText::new("標註會自動儲存並直接覆蓋原標註檔，專案設定位於 <圖片資料夾>/rotalabel_project/。")
                            .small()
                            .color(self.pal.muted),
                    );
                });
            if ctx.input(|i| i.key_pressed(Key::Escape)) {
                open = false;
            }
            self.show_help = open;
        }

        // Info dialog
        if let Some((title, msg)) = self.info_dialog.take() {
            let mut keep = true;
            egui::Window::new(&title)
                .collapsible(false)
                .resizable(false)
                .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
                .show(ctx, |ui| {
                    ui.label(&msg);
                    if ui.button("確定").clicked()
                        || ui.input(|i| i.key_pressed(Key::Enter) || i.key_pressed(Key::Escape))
                    {
                        keep = false;
                    }
                });
            if keep {
                self.info_dialog = Some((title, msg));
            }
        }
    }
}

impl eframe::App for RotaLabelApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.handle_shortcuts(ctx);
        self.ui_top_bar(ctx);
        self.ui_status_bar(ctx);
        self.ui_left_panel(ctx);
        self.ui_right_panel(ctx);

        let modal_open = self.has_modal_open();
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(self.pal.canvas))
            .show(ctx, |ui| {
                let classes = self
                    .project
                    .as_ref()
                    .map(|p| p.classes.as_slice())
                    .unwrap_or(&[]);
                let out = render_canvas(
                    ui,
                    &mut self.canvas,
                    classes,
                    self.texture.as_ref(),
                    &self.pal,
                    modal_open,
                );
                if out.changed {
                    self.flush();
                }
                if out.selection_changed != out.prev_selected {
                    self.sync_selection_to_class();
                }
            });

        self.ui_modals(ctx);
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.flush();
        if let Some(ref p) = self.project {
            p.save_meta();
        }
    }
}

#[allow(dead_code)]
fn _clean_unused_check(_p: &Path) -> PathBuf {
    clean_path(_p)
}
