# RotaLabel (Rust Edition)

高效能純 Rust 圖像標註桌面工具（使用 `eframe` / `egui` GPU 渲染 + `rayon` 平行擴增匯出），支援水平框（BBox）與旋轉框（OBB），並可編譯為單一免安裝執行檔。

## 建置與啟動

需要 Rust 工具鏈（建議 `x86_64-pc-windows-gnu` 或 `x86_64-pc-windows-msvc`）。若使用 Windows GNU 工具鏈且未安裝 MSVC Build Tools，需安裝含完整 `dlltool` / `as` 的 MinGW-w64 MSVCRT 工具鏈（例如 `winget install BrechtSanders.WinLibs.POSIX.MSVCRT`）：

```bash
cargo run --release
```

也可以直接指定圖片資料夾：

```bash
cargo run --release -- D:\data\images
```

編譯單一獨立執行檔（產出位於 `target/release/rotalabel.exe`）：

```bash
cargo build --release
```

執行所有單元與整合測試：

```bash
cargo test
```

## 基本流程

1. 按 **開啟資料夾**（`Ctrl+O`），選擇放圖片的資料夾。
2. 在右側 **類別** 面板新增類別。
3. 按 `W` 畫 BBox，或按 `E` 畫旋轉框，在圖上拖曳出矩形。
4. 用 `A` / `D` 切換上一張 / 下一張。標註會自動儲存並**直接覆蓋原標註檔**。
5. 標完後按 **產生資料集**（`Ctrl+E`），設定切分與擴增，另建版本資料夾匯出訓練資料。

## 標註操作

### 快捷鍵

| 按鍵 | 功能 |
|---|---|
| `W` | 畫 BBox |
| `E` | 畫旋轉框 |
| `V` | 選取 / 編輯模式 |
| `Z` / `X` | 選取的框逆時針 / 順時針旋轉 1°（加 `Shift` 為 15°） |
| `T` | 選取的框在 BBox 與旋轉框之間切換 |
| `1`–`9` | 切換類別（有選取框時會同時變更它的類別） |
| `Delete` / `Backspace` | 刪除選取的框（畫框途中可取消繪製） |
| 方向鍵 | 微調位置 1px（加 `Shift` 為 10px） |
| `Ctrl+C` / `Ctrl+V` | 複製 / 貼上框（可貼到其他圖片） |
| `Ctrl+Z` / `Ctrl+Y` | 復原 / 重做 |
| `A` / `D` | 上一張 / 下一張 |
| `F` | 圖片符合視窗 |
| `L` | 切換標註顯示：全部顯示 → 只顯示選取框的標籤 → 只顯示框 → 全部隱藏 |
| `Esc` | 取消選取 |
| 滾輪 / `Shift` + 滾輪 | 上下 / 左右捲動 |
| `Ctrl` + 滾輪 | 縮放 |
| 中鍵拖曳、空白鍵 + 左鍵拖曳 | 平移 |
| `F1` | 顯示快捷鍵說明 |

## 檔案位置與標註儲存

- 在軟體中修改標註後，會依「**從哪裡讀就寫回哪裡**」直接覆蓋原標註檔（同資料夾直接覆寫圖片旁的 `.txt`，由外部資料夾匯入則直接覆寫原匯入資料夾內的 `.txt`）。
- 專案設定與擴增資料集存放在 `<圖片資料夾>/rotalabel_project/`：
  - `project.json`：類別定義與設定
  - `annotations/`：每張圖一個 `.json`（保留 BBox/OBB 類型與精確座標）
  - `versions/`：擴增資料集輸出目錄（支援 YOLO、YOLO-OBB、DOTA、COCO JSON、COCO-OBB 與拼接擴增 Mosaic）
