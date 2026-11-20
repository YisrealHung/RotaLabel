use egui::{Color32, CornerRadius, Stroke, Visuals};

#[derive(Debug, Clone, Copy)]
pub struct Palette {
    pub bg: Color32,
    pub panel: Color32,
    pub panel2: Color32,
    pub input: Color32,
    pub border: Color32,
    pub text: Color32,
    pub muted: Color32,
    pub accent: Color32,
    pub accent_hover: Color32,
    pub accent_text: Color32,
    pub canvas: Color32,
    pub selection: Color32,
    pub ok: Color32,
    pub warn: Color32,
}

pub fn parse_hex_color(hex: &str) -> Color32 {
    let s = hex.trim().trim_start_matches('#');
    if s.len() >= 6 {
        if let (Ok(r), Ok(g), Ok(b)) = (
            u8::from_str_radix(&s[0..2], 16),
            u8::from_str_radix(&s[2..4], 16),
            u8::from_str_radix(&s[4..6], 16),
        ) {
            return Color32::from_rgb(r, g, b);
        }
    }
    Color32::from_rgb(255, 0, 255)
}

pub fn format_hex_color(c: Color32) -> String {
    format!("#{:02x}{:02x}{:02x}", c.r(), c.g(), c.b())
}

pub fn dark_palette() -> Palette {
    Palette {
        bg: parse_hex_color("#1e1f24"),
        panel: parse_hex_color("#25262c"),
        panel2: parse_hex_color("#2d2f36"),
        input: parse_hex_color("#1a1b20"),
        border: parse_hex_color("#383a43"),
        text: parse_hex_color("#e6e7eb"),
        muted: parse_hex_color("#9095a3"),
        accent: parse_hex_color("#7c5cff"),
        accent_hover: parse_hex_color("#9277ff"),
        accent_text: Color32::WHITE,
        canvas: parse_hex_color("#16171b"),
        selection: parse_hex_color("#3b3270"),
        ok: parse_hex_color("#3ecf8e"),
        warn: parse_hex_color("#f5a524"),
    }
}

pub fn light_palette() -> Palette {
    Palette {
        bg: parse_hex_color("#f4f5f8"),
        panel: Color32::WHITE,
        panel2: parse_hex_color("#eef0f4"),
        input: Color32::WHITE,
        border: parse_hex_color("#dcdfe6"),
        text: parse_hex_color("#1f2329"),
        muted: parse_hex_color("#6b7280"),
        accent: parse_hex_color("#6a4cf5"),
        accent_hover: parse_hex_color("#5a3de0"),
        accent_text: Color32::WHITE,
        canvas: parse_hex_color("#e3e5ea"),
        selection: parse_hex_color("#e6e0ff"),
        ok: parse_hex_color("#16a34a"),
        warn: parse_hex_color("#d97706"),
    }
}

pub fn build_visuals(dark: bool, pal: &Palette) -> Visuals {
    let mut visuals = if dark { Visuals::dark() } else { Visuals::light() };
    visuals.dark_mode = dark;
    visuals.panel_fill = pal.bg;
    visuals.window_fill = pal.panel;
    visuals.window_stroke = Stroke::new(1.0_f32, pal.border);
    visuals.window_corner_radius = CornerRadius::same(8);
    visuals.menu_corner_radius = CornerRadius::same(6);
    visuals.extreme_bg_color = pal.input;
    visuals.faint_bg_color = pal.panel2;
    visuals.code_bg_color = pal.panel2;
    visuals.override_text_color = Some(pal.text);
    visuals.hyperlink_color = pal.accent;
    visuals.warn_fg_color = pal.warn;
    visuals.text_cursor.stroke = Stroke::new(2.0_f32, pal.accent);
    visuals.selection.bg_fill = pal.selection;
    // Selected `SelectableLabel` uses `selection.stroke.color` for text.
    visuals.selection.stroke = Stroke::new(1.0_f32, pal.text);

    let hover_fill = if dark {
        parse_hex_color("#363942")
    } else {
        parse_hex_color("#e2e5ec")
    };
    let cr = CornerRadius::same(4);

    visuals.widgets.noninteractive.bg_fill = pal.panel;
    visuals.widgets.noninteractive.weak_bg_fill = pal.panel;
    visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0_f32, pal.border);
    visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0_f32, pal.text);
    visuals.widgets.noninteractive.corner_radius = cr;

    visuals.widgets.inactive.bg_fill = pal.panel2;
    visuals.widgets.inactive.weak_bg_fill = pal.panel2;
    visuals.widgets.inactive.bg_stroke = Stroke::new(1.0_f32, pal.border);
    visuals.widgets.inactive.fg_stroke = Stroke::new(1.0_f32, pal.text);
    visuals.widgets.inactive.corner_radius = cr;

    visuals.widgets.hovered.bg_fill = hover_fill;
    visuals.widgets.hovered.weak_bg_fill = hover_fill;
    visuals.widgets.hovered.bg_stroke = Stroke::new(1.0_f32, pal.accent);
    visuals.widgets.hovered.fg_stroke = Stroke::new(1.5_f32, pal.text);
    visuals.widgets.hovered.corner_radius = cr;

    visuals.widgets.active.bg_fill = pal.selection;
    visuals.widgets.active.weak_bg_fill = pal.selection;
    visuals.widgets.active.bg_stroke = Stroke::new(1.0_f32, pal.accent);
    visuals.widgets.active.fg_stroke = Stroke::new(1.5_f32, pal.text);
    visuals.widgets.active.corner_radius = cr;

    visuals.widgets.open.bg_fill = pal.panel2;
    visuals.widgets.open.weak_bg_fill = pal.panel2;
    visuals.widgets.open.bg_stroke = Stroke::new(1.0_f32, pal.accent);
    visuals.widgets.open.fg_stroke = Stroke::new(1.0_f32, pal.text);
    visuals.widgets.open.corner_radius = cr;

    visuals
}

pub fn apply_theme(ctx: &egui::Context, dark: bool) -> Palette {
    let dark_pal = dark_palette();
    let light_pal = light_palette();
    ctx.set_visuals_of(egui::Theme::Dark, build_visuals(true, &dark_pal));
    ctx.set_visuals_of(egui::Theme::Light, build_visuals(false, &light_pal));
    ctx.set_theme(if dark {
        egui::Theme::Dark
    } else {
        egui::Theme::Light
    });
    ctx.send_viewport_cmd(egui::ViewportCommand::SetTheme(if dark {
        egui::SystemTheme::Dark
    } else {
        egui::SystemTheme::Light
    }));
    if dark { dark_pal } else { light_pal }
}

pub fn color_lightness(c: Color32) -> f32 {
    0.299 * c.r() as f32 + 0.587 * c.g() as f32 + 0.114 * c.b() as f32
}
