#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::path::PathBuf;

use rotalabel::ui::app::RotaLabelApp;

fn main() -> eframe::Result<()> {
    let initial_folder = std::env::args_os().nth(1).map(PathBuf::from);
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1440.0, 900.0])
            .with_min_inner_size([800.0, 560.0])
            .with_title("RotaLabel 標註工具"),
        ..Default::default()
    };
    eframe::run_native(
        "RotaLabel 標註工具",
        options,
        Box::new(move |cc| Ok(Box::new(RotaLabelApp::new(cc, initial_folder)))),
    )
}
