use std::path::Path;

use egui::{FontData, FontDefinitions, FontFamily};

/// Load Windows Traditional Chinese system font into egui so Chinese UI labels and class names
/// render crisply without missing-glyph boxes.
pub fn install_cjk_fonts(ctx: &egui::Context) {
    let candidates = [
        r"C:\Windows\Fonts\msjh.ttc",
        r"C:\Windows\Fonts\msjhl.ttc",
        r"C:\Windows\Fonts\mingliu.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ];
    for path_str in candidates {
        let p = Path::new(path_str);
        if p.exists() {
            if let Ok(bytes) = std::fs::read(p) {
                let mut fonts = FontDefinitions::default();
                fonts
                    .font_data
                    .insert("cjk".to_owned(), FontData::from_owned(bytes).into());
                if let Some(fam) = fonts.families.get_mut(&FontFamily::Proportional) {
                    fam.insert(0, "cjk".to_owned());
                }
                if let Some(fam) = fonts.families.get_mut(&FontFamily::Monospace) {
                    fam.push("cjk".to_owned());
                }
                ctx.set_fonts(fonts);
                return;
            }
        }
    }
}
