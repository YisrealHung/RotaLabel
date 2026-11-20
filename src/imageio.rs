//! Image read/write with EXIF orientation support and Unicode path safety.

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use image::codecs::jpeg::JpegEncoder;
use image::{DynamicImage, ImageDecoder, ImageReader, RgbImage};

pub const IMAGE_EXTS: &[&str] = &["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"];

pub fn is_image_path(path: &Path) -> bool {
    path.is_file()
        && path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| {
                let low = e.to_ascii_lowercase();
                IMAGE_EXTS.contains(&low.as_str())
            })
            .unwrap_or(false)
}

/// Read an image as RGB8 uint8, applying any EXIF orientation tag.
pub fn imread(path: &Path) -> Option<RgbImage> {
    let file = File::open(path).ok()?;
    let reader = ImageReader::new(BufReader::new(file)).with_guessed_format().ok()?;
    let mut decoder = reader.into_decoder().ok()?;
    let orientation = decoder.orientation().ok();
    let mut dyn_img = DynamicImage::from_decoder(decoder).ok()?;
    if let Some(ori) = orientation {
        dyn_img.apply_orientation(ori);
    }
    Some(dyn_img.to_rgb8())
}

/// Encode an RGB8 image to JPEG bytes.
pub fn encode_jpeg(img: &RgbImage, quality: u8) -> std::io::Result<Vec<u8>> {
    let mut buf = Vec::new();
    let encoder = JpegEncoder::new_with_quality(&mut buf, quality);
    img.write_with_encoder(encoder)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    Ok(buf)
}

/// Save an RGB8 image to disk (uses `quality` for `.jpg` / `.jpeg`).
pub fn imwrite(path: &Path, img: &RgbImage, quality: u8) -> std::io::Result<()> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();
    if ext == "jpg" || ext == "jpeg" {
        let data = encode_jpeg(img, quality)?;
        std::fs::write(path, data)
    } else {
        img.save(path)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }
}
